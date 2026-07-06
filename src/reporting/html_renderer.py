"""Renders a ReportContent object to a single self-contained HTML file
(images embedded as base64 so the file is portable without a figures/
subfolder alongside it).
"""
import base64
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = BASE_DIR / "results"

CSS = """
body { font-family: -apple-system, Helvetica, Arial, sans-serif; max-width: 900px;
       margin: 2rem auto; padding: 0 1.5rem; color: #1A2B2A; line-height: 1.5; }
h1 { color: #0F6E68; margin-bottom: 0.25rem; }
.subtitle { color: #6b7574; margin-bottom: 2rem; }
h2 { color: #0F6E68; border-bottom: 2px solid #0F6E68; padding-bottom: 0.25rem; margin-top: 2.5rem; }
table { border-collapse: collapse; width: 100%; margin: 1rem 0; }
th, td { border: 1px solid #E0E6E5; padding: 0.5rem 0.75rem; text-align: left; font-size: 0.9rem; }
th { background: #0F6E68; color: white; }
caption, .caption { text-align: left; font-style: italic; color: #6b7574; font-size: 0.85rem; margin-top: 0.25rem; }
img { max-width: 100%; margin-top: 0.75rem; border: 1px solid #E0E6E5; border-radius: 8px; }
"""


def _embed_image(relative_path: str) -> str:
    fig_file = RESULTS_DIR / relative_path
    if not fig_file.exists():
        return ""
    encoded = base64.b64encode(fig_file.read_bytes()).decode("ascii")
    return f'<img src="data:image/png;base64,{encoded}" alt="{relative_path}">'


def _dataframe_to_html(df) -> str:
    return df.round(4).to_html(index=False, border=0)


def render_html(content, out_path: Path) -> None:
    parts = [f"<html><head><meta charset='utf-8'><title>{content.title}</title><style>{CSS}</style></head><body>"]
    parts.append(f"<h1>{content.title}</h1>")
    parts.append(f"<div class='subtitle'>{content.subtitle} &middot; Generated {content.generated_at}</div>")

    for section in content.sections:
        parts.append(f"<h2>{section.title}</h2>")
        for paragraph in section.paragraphs:
            parts.append(f"<p>{paragraph}</p>")

        if section.table is not None:
            parts.append(_dataframe_to_html(section.table))
            if section.table_caption:
                parts.append(f"<div class='caption'>{section.table_caption}</div>")

        if section.figure_path:
            img_html = _embed_image(section.figure_path)
            if img_html:
                parts.append(img_html)
                if section.figure_caption:
                    parts.append(f"<div class='caption'>{section.figure_caption}</div>")

    parts.append("</body></html>")
    out_path.write_text("\n".join(parts), encoding="utf-8")
