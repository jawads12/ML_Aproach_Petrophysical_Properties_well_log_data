#!/usr/bin/env python3
"""Generate a PDF defense-preparation report from the Markdown source."""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_MD = PROJECT_ROOT / "defense_prep/ML_Defense_Preparation_Report.md"
OUTPUT_PDF = PROJECT_ROOT / "defense_prep/ML_Defense_Preparation_Report.pdf"


def clean_inline(text: str) -> str:
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    while "**" in text:
        text = text.replace("**", "", 1)
    text = text.replace("`", "")
    return text


def is_table_start(lines: list[str], index: int) -> bool:
    if index + 1 >= len(lines):
        return False
    return lines[index].strip().startswith("|") and set(lines[index + 1].strip()) <= {"|", "-", " ", ":"}


def consume_table(lines: list[str], index: int, styles) -> tuple[Table, int]:
    rows: list[list[Paragraph]] = []
    current = index
    while current < len(lines) and lines[current].strip().startswith("|"):
        raw = lines[current].strip()
        if set(raw) <= {"|", "-", " ", ":"}:
            current += 1
            continue
        cells = [cell.strip() for cell in raw.strip("|").split("|")]
        rows.append([Paragraph(clean_inline(cell), styles["TableCell"]) for cell in cells])
        current += 1

    page_width = A4[0] - 4 * cm
    column_count = max(len(row) for row in rows)
    col_width = page_width / column_count
    table = Table(rows, colWidths=[col_width] * column_count, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2933")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#c7c7c7")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f7fa")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table, current


def make_styles():
    base = getSampleStyleSheet()
    base["Title"].alignment = TA_CENTER
    base["Title"].fontSize = 20
    base["Title"].leading = 24
    base["Heading1"].fontSize = 16
    base["Heading1"].leading = 20
    base["Heading1"].spaceBefore = 14
    base["Heading1"].spaceAfter = 8
    base["Heading2"].fontSize = 13
    base["Heading2"].leading = 16
    base["Heading2"].spaceBefore = 10
    base["Heading2"].spaceAfter = 6
    base["BodyText"].fontSize = 10
    base["BodyText"].leading = 14
    base["BodyText"].spaceAfter = 7
    base.add(
        ParagraphStyle(
            name="BulletCustom",
            parent=base["BodyText"],
            leftIndent=14,
            firstLineIndent=-8,
            bulletIndent=4,
        )
    )
    base.add(
        ParagraphStyle(
            name="TableCell",
            parent=base["BodyText"],
            fontSize=8,
            leading=10,
            spaceAfter=0,
        )
    )
    return base


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawString(2 * cm, 1.2 * cm, "ML Defense Preparation Report")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
    canvas.restoreState()


def build_story(markdown: str, styles):
    story = []
    lines = markdown.splitlines()
    i = 0
    first_title = True
    while i < len(lines):
        line = lines[i].rstrip()
        stripped = line.strip()
        if not stripped:
            story.append(Spacer(1, 4))
            i += 1
            continue
        if is_table_start(lines, i):
            table, i = consume_table(lines, i, styles)
            story.append(table)
            story.append(Spacer(1, 8))
            continue
        if stripped.startswith("# "):
            if not first_title:
                story.append(PageBreak())
            first_title = False
            story.append(Paragraph(clean_inline(stripped[2:]), styles["Title"]))
            story.append(Spacer(1, 10))
        elif stripped.startswith("## "):
            story.append(Paragraph(clean_inline(stripped[3:]), styles["Heading1"]))
        elif stripped.startswith("### "):
            story.append(Paragraph(clean_inline(stripped[4:]), styles["Heading2"]))
        elif stripped.startswith("- "):
            story.append(Paragraph(clean_inline(stripped[2:]), styles["BulletCustom"], bulletText="-"))
        elif stripped[0:2].isdigit() and ". " in stripped[:5]:
            story.append(Paragraph(clean_inline(stripped), styles["BodyText"]))
        else:
            story.append(Paragraph(clean_inline(stripped), styles["BodyText"]))
        i += 1
    return story


def main() -> None:
    markdown = SOURCE_MD.read_text(encoding="utf-8")
    styles = make_styles()
    doc = SimpleDocTemplate(
        str(OUTPUT_PDF),
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        title="ML Defense Preparation Report",
    )
    story = build_story(markdown, styles)
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(OUTPUT_PDF)


if __name__ == "__main__":
    main()
