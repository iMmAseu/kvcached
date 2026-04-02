from __future__ import annotations

import html
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


CSS = """
:root {
  --bg: #f5f7fb;
  --surface: #ffffff;
  --surface-muted: #f9fbff;
  --text: #1f2937;
  --text-soft: #4b5563;
  --border: #d7deea;
  --accent: #0f4c81;
  --accent-soft: #e9f2fb;
  --code-bg: #f3f6fa;
  --shadow: 0 14px 40px rgba(15, 23, 42, 0.08);
  --font-sans: "Segoe UI", "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei",
    "Noto Sans", Arial, sans-serif;
  --font-mono: "Cascadia Code", Consolas, "SFMono-Regular", Menlo, Monaco, monospace;
  --radius: 18px;
}

*,
*::before,
*::after {
  box-sizing: border-box;
}

html {
  scroll-behavior: smooth;
}

body {
  margin: 0;
  background: linear-gradient(180deg, #f7f9fc 0%, #f3f6fb 100%);
  color: var(--text);
  font-family: var(--font-sans);
  line-height: 1.78;
}

a {
  color: var(--accent);
  text-decoration: none;
}

a:hover {
  text-decoration: underline;
}

.page {
  max-width: 1040px;
  margin: 0 auto;
  padding: 32px 20px 64px;
}

.hero,
.toc,
.content {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
}

.hero {
  padding: 30px 34px 26px;
  margin-bottom: 22px;
}

.eyebrow {
  margin: 0 0 10px;
  color: var(--accent);
  font-size: 13px;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.hero h1 {
  margin: 0;
  font-size: clamp(30px, 4vw, 42px);
  line-height: 1.2;
  letter-spacing: -0.02em;
}

.subtitle {
  margin: 16px 0 0;
  color: var(--text-soft);
  font-size: 16px;
}

.hero-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 12px 16px;
  margin-top: 20px;
  color: var(--text-soft);
  font-size: 14px;
}

.meta-pill {
  display: inline-flex;
  align-items: center;
  padding: 7px 12px;
  border-radius: 999px;
  background: var(--accent-soft);
  border: 1px solid #cfe1f2;
}

.toc {
  padding: 22px 28px;
  margin-bottom: 22px;
}

.toc h2 {
  margin: 0 0 12px;
  font-size: 20px;
}

.toc ul {
  margin: 0;
  padding-left: 20px;
}

.toc li {
  margin: 8px 0;
}

.content {
  padding: 28px 34px 38px;
}

.content h2,
.content h3,
.content h4,
.content h5,
.content h6 {
  scroll-margin-top: 24px;
  line-height: 1.35;
  letter-spacing: -0.01em;
}

.content h2 {
  margin: 34px 0 14px;
  padding-top: 8px;
  font-size: 28px;
}

.content h3 {
  margin: 28px 0 12px;
  font-size: 22px;
}

.content h4 {
  margin: 24px 0 10px;
  font-size: 18px;
}

.content p {
  margin: 14px 0;
}

.content ul,
.content ol {
  margin: 14px 0 18px;
  padding-left: 24px;
}

.content li {
  margin: 8px 0;
}

.content code {
  padding: 0.15em 0.4em;
  border-radius: 8px;
  background: var(--code-bg);
  font-family: var(--font-mono);
  font-size: 0.94em;
}

.content pre {
  margin: 18px 0;
  padding: 18px 18px;
  overflow-x: auto;
  border: 1px solid var(--border);
  border-radius: 14px;
  background: #f6f8fb;
}

.content pre code {
  display: block;
  padding: 0;
  background: transparent;
  white-space: pre-wrap;
  word-break: break-word;
  line-height: 1.65;
}

.table-wrap {
  margin: 18px 0 22px;
  overflow-x: auto;
  border: 1px solid var(--border);
  border-radius: 14px;
}

table {
  width: 100%;
  border-collapse: collapse;
  background: var(--surface);
}

thead {
  background: var(--surface-muted);
}

th,
td {
  padding: 12px 14px;
  border-bottom: 1px solid var(--border);
  text-align: left;
  vertical-align: top;
  word-break: break-word;
  font-variant-numeric: tabular-nums;
}

tbody tr:last-child td {
  border-bottom: none;
}

.footer {
  margin-top: 18px;
  color: var(--text-soft);
  font-size: 13px;
  text-align: center;
}

.landing {
  max-width: 860px;
  margin: 0 auto;
  padding: 40px 20px 64px;
}

.landing-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  padding: 34px;
}

.landing-card h1 {
  margin: 0 0 12px;
  font-size: clamp(28px, 4vw, 40px);
  line-height: 1.2;
}

.landing-card p {
  margin: 0 0 22px;
  color: var(--text-soft);
}

.landing-links {
  display: grid;
  gap: 12px;
}

.landing-link {
  display: block;
  padding: 16px 18px;
  border: 1px solid var(--border);
  border-radius: 14px;
  background: var(--surface-muted);
  color: var(--text);
  font-weight: 600;
}

.landing-link small {
  display: block;
  margin-top: 6px;
  color: var(--text-soft);
  font-weight: 400;
}

@media (max-width: 720px) {
  .page {
    padding: 18px 12px 36px;
  }

  .hero,
  .toc,
  .content {
    padding-left: 18px;
    padding-right: 18px;
  }

  .content {
    padding-top: 22px;
    padding-bottom: 28px;
  }

  .content h2 {
    font-size: 24px;
  }
}
"""


HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
UL_RE = re.compile(r"^\s*-\s+(.*)$")
OL_RE = re.compile(r"^\s*(\d+)\.\s+(.*)$")


def render_inline(text: str) -> str:
    parts = text.split("`")
    rendered: list[str] = []
    for i, part in enumerate(parts):
        if i % 2 == 1:
            rendered.append(f"<code>{html.escape(part)}</code>")
            continue
        segment = html.escape(part)
        segment = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", segment)
        segment = re.sub(r"\*(.+?)\*", r"<em>\1</em>", segment)
        segment = re.sub(
            r"\[(.+?)\]\((.+?)\)",
            lambda m: f'<a href="{html.escape(m.group(2), quote=True)}">{m.group(1)}</a>',
            segment,
        )
        rendered.append(segment)
    return "".join(rendered)


def parse_table(lines: list[str], start: int) -> tuple[str, int]:
    table_lines: list[str] = []
    i = start
    while i < len(lines) and lines[i].strip().startswith("|"):
        table_lines.append(lines[i].strip())
        i += 1

    header = [cell.strip() for cell in table_lines[0].strip("|").split("|")]
    aligns = [cell.strip() for cell in table_lines[1].strip("|").split("|")]
    rows = [
        [cell.strip() for cell in row.strip("|").split("|")]
        for row in table_lines[2:]
    ]

    def align_style(cell: str) -> str:
        left = cell.startswith(":")
        right = cell.endswith(":")
        if left and right:
            return ' style="text-align:center"'
        if right:
            return ' style="text-align:right"'
        return ""

    out = ['<div class="table-wrap"><table><thead><tr>']
    for idx, cell in enumerate(header):
        out.append(f"<th{align_style(aligns[idx])}>{render_inline(cell)}</th>")
    out.append("</tr></thead><tbody>")
    for row in rows:
        out.append("<tr>")
        for idx, cell in enumerate(row):
            align = aligns[idx] if idx < len(aligns) else ""
            out.append(f"<td{align_style(align)}>{render_inline(cell)}</td>")
        out.append("</tr>")
    out.append("</tbody></table></div>")
    return "".join(out), i


def render_markdown(md_path: Path, html_path: Path, lang: str, sibling_html: str | None) -> None:
    lines = md_path.read_text(encoding="utf-8").splitlines()
    title = md_path.stem
    first_paragraph = ""
    content_parts: list[str] = []
    toc_entries: list[tuple[int, str, str]] = []
    paragraph_buffer: list[str] = []
    heading_index = 0
    consumed_h1 = False

    def flush_paragraph() -> None:
        nonlocal first_paragraph
        if not paragraph_buffer:
            return
        text = " ".join(part.strip() for part in paragraph_buffer).strip()
        if text:
            rendered = render_inline(text)
            content_parts.append(f"<p>{rendered}</p>")
            if not first_paragraph:
                first_paragraph = text
        paragraph_buffer.clear()

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            flush_paragraph()
            i += 1
            continue

        if stripped.startswith("```"):
            flush_paragraph()
            lang_tag = stripped[3:].strip()
            code_lines: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            code = html.escape("\n".join(code_lines))
            class_attr = f' class="language-{lang_tag}"' if lang_tag else ""
            content_parts.append(f"<pre><code{class_attr}>{code}</code></pre>")
            i += 1
            continue

        heading = HEADING_RE.match(line)
        if heading:
            flush_paragraph()
            level = len(heading.group(1))
            text = heading.group(2).strip()
            if level == 1 and not consumed_h1:
                title = text
                consumed_h1 = True
            else:
                heading_index += 1
                anchor = f"section-{heading_index}"
                content_parts.append(
                    f'<h{level} id="{anchor}">{render_inline(text)}</h{level}>'
                )
                if level == 2:
                    toc_entries.append((level, text, anchor))
            i += 1
            continue

        if stripped.startswith("|") and i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            if next_line.startswith("|") and "-" in next_line:
                flush_paragraph()
                table_html, i = parse_table(lines, i)
                content_parts.append(table_html)
                continue

        ul_match = UL_RE.match(line)
        if ul_match:
            flush_paragraph()
            items: list[str] = []
            while i < len(lines):
                match = UL_RE.match(lines[i])
                if not match:
                    break
                items.append(f"<li>{render_inline(match.group(1).strip())}</li>")
                i += 1
            content_parts.append("<ul>" + "".join(items) + "</ul>")
            continue

        ol_match = OL_RE.match(line)
        if ol_match:
            flush_paragraph()
            items = []
            while i < len(lines):
                match = OL_RE.match(lines[i])
                if not match:
                    break
                items.append(f"<li>{render_inline(match.group(2).strip())}</li>")
                i += 1
            content_parts.append("<ol>" + "".join(items) + "</ol>")
            continue

        paragraph_buffer.append(line)
        i += 1

    flush_paragraph()

    toc_title = "目录" if lang == "zh-CN" else "Contents"
    source_label = "源文件" if lang == "zh-CN" else "Source"
    export_label = "静态 HTML，适合直接部署到 GitHub Pages" if lang == "zh-CN" else "Static HTML suitable for direct GitHub Pages hosting"
    sibling_label = "English Version" if lang == "zh-CN" else "中文版"

    toc_html = ['<nav class="toc"><h2>', toc_title, "</h2><ul>"]
    for _level, text, anchor in toc_entries:
        toc_html.append(f'<li><a href="#{anchor}">{html.escape(text)}</a></li>')
    toc_html.append("</ul></nav>")

    hero_meta = [
        f'<span class="meta-pill">{source_label}: {html.escape(md_path.name)}</span>',
        f'<span class="meta-pill">{export_label}</span>',
    ]
    if sibling_html:
        hero_meta.append(
            f'<a class="meta-pill" href="{html.escape(sibling_html, quote=True)}">{sibling_label}</a>'
        )

    description = html.escape(first_paragraph[:220]).strip()

    doc = f"""<!DOCTYPE html>
<html lang="{lang}">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="{description}">
  <title>{html.escape(title)}</title>
  <style>{CSS}</style>
</head>
<body>
  <div class="page">
    <header class="hero">
      <p class="eyebrow">kvcached Docs</p>
      <h1>{html.escape(title)}</h1>
      <p class="subtitle">{html.escape(first_paragraph)}</p>
      <div class="hero-meta">{''.join(hero_meta)}</div>
    </header>
    {''.join(toc_html)}
    <main class="content">
      {''.join(content_parts)}
    </main>
    <div class="footer">Generated from Markdown for static hosting.</div>
  </div>
</body>
</html>
"""
    html_path.write_text(doc, encoding="utf-8")


def main() -> None:
    render_markdown(
        DOCS / "vllm_async_mp_investigation.md",
        DOCS / "vllm_async_mp_investigation.html",
        "zh-CN",
        "vllm_async_mp_investigation_en.html",
    )
    render_markdown(
        DOCS / "vllm_async_mp_investigation_en.md",
        DOCS / "vllm_async_mp_investigation_en.html",
        "en",
        "vllm_async_mp_investigation.html",
    )
    index = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>kvcached Docs</title>
  <style>{CSS}</style>
</head>
<body>
  <div class="landing">
    <div class="landing-card">
      <p class="eyebrow">kvcached Docs</p>
      <h1>vLLM Async Scheduling + MP</h1>
      <p>Static HTML documents for direct GitHub Pages hosting.</p>
      <div class="landing-links">
        <a class="landing-link" href="vllm_async_mp_investigation.html">
          中文文档
          <small>Investigation, fix rationale, and benchmark notes.</small>
        </a>
        <a class="landing-link" href="vllm_async_mp_investigation_en.html">
          English Version
          <small>Technical blog style English write-up.</small>
        </a>
      </div>
    </div>
  </div>
</body>
</html>
"""
    (DOCS / "index.html").write_text(index, encoding="utf-8")


if __name__ == "__main__":
    main()
