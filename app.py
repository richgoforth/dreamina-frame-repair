#!/usr/bin/env python3
"""
Web frontend for dreamina-frame-repair.
Local:      python3 app.py  (opens browser automatically)
Production: gunicorn handles startup via Procfile
"""
from __future__ import annotations

import json
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import webbrowser
from pathlib import Path

from flask import Flask, after_this_request, jsonify, render_template, request, Response, send_file

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB

REPAIR_PY = Path(__file__).parent / "repair.py"

# job_id -> {queue, workdir, input, output, filename, status}
jobs: dict = {}

# Lines we don't surface to the user (internal paths, spec file messages)
_SUPPRESS = re.compile(r"Repair spec saved to:|Use --auto-repair|python repair\.py")


def _clean(line: str) -> str | None:
    """Return a user-facing version of a log line, or None to suppress it."""
    if _SUPPRESS.search(line):
        return None
    # Strip full filesystem paths from the Done line
    if line.strip().startswith("Done."):
        return re.sub(r"\s+/\S+\s+", "  ", line).strip()
    return line


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    # Tell the page whether it's running on a server (hosted) or on the user's
    # own machine (local), so the privacy footer states the truth either way.
    hosted = bool(os.environ.get("RAILWAY_ENVIRONMENT"))
    return render_template("index.html", hosted=hosted)


@app.route("/upload", methods=["POST"])
def upload():
    if "video" not in request.files:
        return jsonify({"error": "No video file"}), 400
    f = request.files["video"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400

    job_id      = str(uuid.uuid4())[:8]
    workdir     = Path(tempfile.mkdtemp(prefix=f"repair_{job_id}_"))
    ext         = Path(f.filename).suffix.lower() or ".mp4"
    input_path  = workdir / f"input{ext}"
    output_path = workdir / "output.mov"

    f.save(str(input_path))

    q = queue.Queue()
    jobs[job_id] = {
        "queue":    q,
        "workdir":  workdir,
        "input":    input_path,
        "output":   output_path,
        "filename": f.filename,
        "status":   "pending",
    }

    threading.Thread(target=_run_repair, args=(job_id,), daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/stream/<job_id>")
def stream(job_id):
    if job_id not in jobs:
        return jsonify({"error": "Job not found"}), 404

    q = jobs[job_id]["queue"]

    def generate():
        while True:
            try:
                msg = q.get(timeout=90)
            except queue.Empty:
                yield 'data: {"type":"ping"}\n\n'
                continue
            if msg is None:
                break
            yield f"data: {json.dumps(msg)}\n\n"

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/download/<job_id>")
def download(job_id):
    job = jobs.get(job_id)
    if not job or not job["output"].exists():
        return jsonify({"error": "File not found"}), 404

    dl_name = Path(job["filename"]).stem + "_repaired.mov"

    @after_this_request
    def cleanup(response):
        def _delete():
            time.sleep(10)
            shutil.rmtree(str(job["workdir"]), ignore_errors=True)
            jobs.pop(job_id, None)
        threading.Thread(target=_delete, daemon=True).start()
        return response

    return send_file(str(job["output"]), as_attachment=True, download_name=dl_name)


# ---------------------------------------------------------------------------
# Repair worker
# ---------------------------------------------------------------------------

def _run_repair(job_id: str) -> None:
    job = jobs[job_id]
    q   = job["queue"]
    jobs[job_id]["status"] = "running"

    cmd = [
        sys.executable, str(REPAIR_PY),
        str(job["input"]),
        "--detect", "--auto-repair",
        "--output", str(job["output"]),
    ]

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        done_size = ""
        for raw in proc.stdout:
            line = raw.rstrip()
            if not line:
                continue
            clean = _clean(line)
            if clean is not None:
                q.put({"type": "log", "text": clean})
            if line.strip().startswith("Done.") and "(" in line:
                done_size = line.split("(")[-1].rstrip(")")

        proc.wait()

        if proc.returncode == 0 and job["output"].exists():
            jobs[job_id]["status"] = "done"
            q.put({"type": "done", "size": done_size})
        else:
            jobs[job_id]["status"] = "error"
            q.put({"type": "error", "message": "Repair failed — see log above for details."})

    except Exception as exc:
        jobs[job_id]["status"] = "error"
        q.put({"type": "error", "message": str(exc)})

    q.put(None)  # sentinel


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port       = int(os.environ.get("PORT", 8742))
    is_local   = not os.environ.get("RAILWAY_ENVIRONMENT") and port == 8742

    if is_local:
        url = f"http://localhost:{port}"
        print(f"\n  dreamina-frame-repair  →  {url}\n")
        def _open():
            time.sleep(0.9)
            webbrowser.open(url)
        threading.Thread(target=_open, daemon=True).start()

    app.run(host="0.0.0.0", port=port, threaded=True, debug=False)
