import subprocess
import time
import os

processes = []

processes.append(subprocess.Popen(["python", "synthetic_streaming_engine.py"]))
time.sleep(3)

processes.append(subprocess.Popen(["python", "live_pipeline_processor.py"]))
time.sleep(3)

port = os.environ.get("PORT", "8501")
processes.append(subprocess.Popen([
    "streamlit", "run", "live_pv_dashboard_recovery_improved.py",
    "--server.port", port,
    "--server.address", "0.0.0.0"
]))

for p in processes:
    p.wait()