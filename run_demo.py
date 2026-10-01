"""One command to run everything:   python run_demo.py

1. trains the pipeline if ./artifacts is empty (about 1-4 minutes, < 1 GB RAM, max 2 CPU threads)
2. starts the API + dashboard on http://127.0.0.1:8000 and opens your browser
"""
import config  # noqa: F401  (caps CPU threads before numpy/xgboost are imported)
import sys
import threading
import time
import webbrowser

from config import ARTIFACTS


def main(port: int = 8000):
    if not (ARTIFACTS / "model.json").exists():
        from src.train import main as train
        train()
    import uvicorn
    threading.Thread(target=lambda: (time.sleep(2.5), webbrowser.open(f"http://127.0.0.1:{port}")), daemon=True).start()
    uvicorn.run("app.main:app", host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8000)
