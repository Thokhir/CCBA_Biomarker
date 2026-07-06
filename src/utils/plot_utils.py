from pathlib import Path
import matplotlib.pyplot as plt
from typing import Optional, List


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_fig(name: str,
             fig: Optional[plt.Figure] = None,
             out_dir: Optional[Path] = None,
             fmt: str = "png",
             dpi: int = 300,
             transparent: bool = False,
             close: bool = True) -> Path:
    """Save a single matplotlib figure.

    - `name`: display name for the file (only this needs to change per plot)
    - `fig`: optional Figure object; if None, uses the current figure
    - `out_dir`: where to save (defaults to ./plots)
    - `fmt`, `dpi`, `transparent`, `close`: standard savefig options

    Returns the saved file Path.
    """
    if out_dir is None:
        # default to project-level data/intermediate/plots
        out_dir = Path(__file__).resolve().parents[2] / "data" / "intermediate" / "plots"
    out_dir = Path(out_dir)
    _ensure_dir(out_dir)

    if fig is None:
        fig = plt.gcf()

    # sanitize filename
    safe = "".join(c if (c.isalnum() or c in ("-", "_", " ")) else "_" for c in name).strip().replace(" ", "_")
    file_path = out_dir / f"{safe}.{fmt}"

    fig.savefig(file_path, dpi=dpi, transparent=transparent, bbox_inches="tight")
    if close:
        try:
            plt.close(fig)
        except Exception:
            pass
    return file_path


def save_all_figs(prefix: str = "plot",
                  out_dir: Optional[Path] = None,
                  fmt: str = "png",
                  dpi: int = 300,
                  transparent: bool = False,
                  close: bool = True) -> List[Path]:
    """Save all open matplotlib figures.

    - `prefix` will be prepended to each filename followed by the figure number.
    - Returns a list of saved Paths.
    """
    if out_dir is None:
        out_dir = Path(__file__).resolve().parents[2] / "data" / "intermediate" / "plots"
    out_dir = Path(out_dir)
    _ensure_dir(out_dir)

    saved: List[Path] = []
    for num in plt.get_fignums():
        fig = plt.figure(num)
        name = f"{prefix}_{num}"
        path = save_fig(name, fig=fig, out_dir=out_dir, fmt=fmt, dpi=dpi, transparent=transparent, close=close)
        saved.append(path)
    return saved


def save_current(name: str, **kwargs) -> Path:
    """Convenience wrapper to save the current figure with a name."""
    return save_fig(name, **kwargs)
