"""Desktop entry point: starts the Streamlit dashboard as a subprocess and
opens it in the default browser, so a packaged executable behaves like a
normal desktop app rather than requiring the user to run a CLI command.
"""
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[3]
APP_PATH = BASE_DIR / "src" / "dashboard" / "app.py"
PORT = 8501


def main() -> None:
    process = subprocess.Popen([
        sys.executable, "-m", "streamlit", "run", str(APP_PATH),
        "--server.port", str(PORT), "--server.headless", "true",
    ])
    time.sleep(3)
    webbrowser.open(f"http://localhost:{PORT}")
    process.wait()


if __name__ == "__main__":
    main()
