"""Builds a standalone Windows executable of the CCA-BDP dashboard via
PyInstaller, bundling launcher.py plus the results/, .streamlit/, and src/
directories the app reads at runtime.

NOT executed in this session: this platform's dependency footprint
(scikit-learn, shap, streamlit, gseapy, xgboost, lightgbm, reportlab,
python-docx, kaleido, matplotlib, lifelines) is large, and PyInstaller
bundling of packages like this commonly needs iterative --hidden-import
fixes the first time it's run. Given the build/debug cycle for a bundle
this size can take a long time with no guarantee of success on the first
attempt, this was written but not run - run it yourself and expect to add
--hidden-import flags for whichever package PyInstaller's static analysis
misses first (streamlit and shap are the most common offenders for this
kind of bundling problem).

Usage: python src/deployment/desktop/build_executable.py
"""
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[3]
LAUNCHER = BASE_DIR / "src" / "deployment" / "desktop" / "launcher.py"

PYINSTALLER_ARGS = [
    sys.executable, "-m", "PyInstaller",
    "--name", "CCA-BDP",
    "--onefile",
    "--add-data", f"{BASE_DIR / 'results'};results",
    "--add-data", f"{BASE_DIR / '.streamlit'};.streamlit",
    "--add-data", f"{BASE_DIR / 'src'};src",
    "--collect-all", "streamlit",
    "--collect-all", "shap",
    str(LAUNCHER),
]


def main() -> None:
    print("Running PyInstaller (this can take several minutes)...")
    print(" ".join(str(a) for a in PYINSTALLER_ARGS))
    subprocess.run(PYINSTALLER_ARGS, check=True, cwd=BASE_DIR)
    print("Build complete. Executable at dist/CCA-BDP.exe")


if __name__ == "__main__":
    main()
