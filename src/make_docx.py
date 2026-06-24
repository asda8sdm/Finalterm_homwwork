from __future__ import annotations

import html
import re
import struct
import zipfile
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_DIR / "report"
FIG_DIR = PROJECT_DIR / "outputs" / "figures"
DOCX_PATH = REPORT_DIR / "final_report.docx"

NS_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS_R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
NS_WP = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
NS_A = "http://schemas.openxmlformats.org/drawingml/2006/main"
NS_PIC = "http://schemas.openxmlformats.org/drawingml/2006/picture"


def esc(text: str) -> str:
    return html.escape(text, quote=False)


def png_size(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"Not a PNG file: {path}")
    return struct.unpack(">II", data[16:24])


def run_text(text: str, code: bool = False, bold: bool = False, color: str | None = None) -> str:
    props = []
    if bold:
        props.append("<w:b/>")
    if color:
        props.append(f'<w:color w:val="{color}"/>')
    if code:
        props.append('<w:rFonts w:ascii="Consolas" w:hAnsi="Consolas" w:eastAsia="Consolas"/>')
    rpr = f"<w:rPr>{''.join(props)}</w:rPr>" if props else ""
    space = ' xml:space="preserve"' if text.startswith(" ") or text.endswith(" ") else ""
    return f"<w:r>{rpr}<w:t{space}>{esc(text)}</w:t></w:r>"


def inline_markdown(text: str) -> str:
    parts: list[str] = []
    pos = 0
    for match in re.finditer(r"`([^`]+)`", text):
        if match.start() > pos:
            parts.append(run_text(text[pos : match.start()]))
        parts.append(run_text(match.group(1), code=True))
        pos = match.end()
    if pos < len(text):
        parts.append(run_text(text[pos:]))
    return "".join(parts) if parts else run_text("")


def paragraph(text: str = "", style: str | None = None, center: bool = False, caption: bool = False) -> str:
    ppr = []
    if style:
        ppr.append(f'<w:pStyle w:val="{style}"/>')
    if center:
        ppr.append('<w:jc w:val="center"/>')
    if caption:
        ppr.append('<w:spacing w:before="80" w:after="160"/>')
    ppr_xml = f"<w:pPr>{''.join(ppr)}</w:pPr>" if ppr else ""
    if caption:
        return f"<w:p>{ppr_xml}{run_text(text, bold=True, color='4F6F8F')}</w:p>"
    return f"<w:p>{ppr_xml}{inline_markdown(text)}</w:p>"


def bullet(text: str, level: int = 0) -> str:
    indent = 420 + level * 360
    return (
        f'<w:p><w:pPr><w:ind w:left="{indent}" w:hanging="240"/><w:spacing w:after="80"/></w:pPr>'
        f'{run_text("• ")}{inline_markdown(text)}</w:p>'
    )


def code_paragraph(text: str) -> str:
    return (
        '<w:p><w:pPr><w:spacing w:before="0" w:after="0"/>'
        '<w:ind w:left="240"/><w:shd w:fill="F6F8FA"/></w:pPr>'
        f"{run_text(text, code=True)}</w:p>"
    )


def table(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    col_count = len(rows[0])
    width = 9360 // col_count
    grid = "".join(f'<w:gridCol w:w="{width}"/>' for _ in range(col_count))
    trs = []
    for i, row in enumerate(rows):
        cells = []
        for cell in row:
            shade = '<w:shd w:fill="D9EAF7"/>' if i == 0 else ('<w:shd w:fill="F7FBFD"/>' if i % 2 == 0 else "")
            tc_mar = '<w:tcMar><w:top w:w="80" w:type="dxa"/><w:left w:w="80" w:type="dxa"/><w:bottom w:w="80" w:type="dxa"/><w:right w:w="80" w:type="dxa"/></w:tcMar>'
            cells.append(
                '<w:tc>'
                f'<w:tcPr><w:tcW w:w="{width}" w:type="dxa"/>{tc_mar}{shade}</w:tcPr>'
                f"<w:p>{inline_markdown(cell.strip())}</w:p>"
                "</w:tc>"
            )
        trs.append(f"<w:tr>{''.join(cells)}</w:tr>")
    return (
        '<w:tbl><w:tblPr><w:tblStyle w:val="TableGrid"/>'
        '<w:tblW w:w="0" w:type="auto"/><w:tblBorders>'
        '<w:top w:val="single" w:sz="4" w:space="0" w:color="BFBFBF"/>'
        '<w:left w:val="single" w:sz="4" w:space="0" w:color="BFBFBF"/>'
        '<w:bottom w:val="single" w:sz="4" w:space="0" w:color="BFBFBF"/>'
        '<w:right w:val="single" w:sz="4" w:space="0" w:color="BFBFBF"/>'
        '<w:insideH w:val="single" w:sz="4" w:space="0" w:color="BFBFBF"/>'
        '<w:insideV w:val="single" w:sz="4" w:space="0" w:color="BFBFBF"/>'
        f"</w:tblBorders></w:tblPr><w:tblGrid>{grid}</w:tblGrid>{''.join(trs)}</w:tbl>"
    )


def image_paragraph(path: Path, rel_id: str, doc_pr_id: int, max_width_inches: float = 6.7) -> str:
    width_px, height_px = png_size(path)
    width_emu = int(max_width_inches * 914400)
    height_emu = int(width_emu * height_px / width_px)
    name = path.name
    return f"""
<w:p>
  <w:pPr><w:jc w:val="center"/></w:pPr>
  <w:r>
    <w:drawing>
      <wp:inline distT="0" distB="0" distL="0" distR="0">
        <wp:extent cx="{width_emu}" cy="{height_emu}"/>
        <wp:effectExtent l="0" t="0" r="0" b="0"/>
        <wp:docPr id="{doc_pr_id}" name="{esc(name)}"/>
        <wp:cNvGraphicFramePr><a:graphicFrameLocks noChangeAspect="1"/></wp:cNvGraphicFramePr>
        <a:graphic>
          <a:graphicData uri="{NS_PIC}">
            <pic:pic>
              <pic:nvPicPr><pic:cNvPr id="{doc_pr_id}" name="{esc(name)}"/><pic:cNvPicPr/></pic:nvPicPr>
              <pic:blipFill><a:blip r:embed="{rel_id}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>
              <pic:spPr>
                <a:xfrm><a:off x="0" y="0"/><a:ext cx="{width_emu}" cy="{height_emu}"/></a:xfrm>
                <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
              </pic:spPr>
            </pic:pic>
          </a:graphicData>
        </a:graphic>
      </wp:inline>
    </w:drawing>
  </w:r>
</w:p>
"""


def parse_markdown(md: str) -> tuple[str, list[tuple[str, Path]]]:
    body: list[str] = []
    image_rels: list[tuple[str, Path]] = []
    lines = md.splitlines()
    i = 0
    in_code = False
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("```"):
            in_code = not in_code
            i += 1
            continue
        if in_code:
            body.append(code_paragraph(line))
            i += 1
            continue

        if stripped.startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i].strip())
                i += 1
            rows = []
            for tline in table_lines:
                if re.fullmatch(r"\|[\s:\-\|]+\|", tline):
                    continue
                rows.append([c.strip() for c in tline.strip("|").split("|")])
            body.append(table(rows))
            continue

        img_match = re.match(r"!\[(.*?)\]\((.*?)\)", stripped)
        if img_match:
            rel_id = f"rIdImg{len(image_rels) + 1}"
            img_path = (REPORT_DIR / img_match.group(2)).resolve()
            image_rels.append((rel_id, img_path))
            body.append(paragraph(img_match.group(1), center=True, caption=True))
            body.append(image_paragraph(img_path, rel_id, len(image_rels)))
            i += 1
            continue

        if stripped.startswith("# "):
            body.append(paragraph(stripped[2:], style="Title", center=True))
        elif stripped.startswith("## "):
            body.append(paragraph(stripped[3:], style="Heading1"))
        elif stripped.startswith("### "):
            body.append(paragraph(stripped[4:], style="Heading2"))
        elif re.match(r"^\d+\. ", stripped):
            body.append(bullet(stripped, level=0))
        elif stripped.startswith("- "):
            body.append(bullet(stripped[2:], level=0))
        elif not stripped:
            body.append(paragraph(""))
        else:
            body.append(paragraph(stripped))
        i += 1

    return "\n".join(body), image_rels


def document_xml(body_xml: str) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="{NS_W}" xmlns:r="{NS_R}" xmlns:wp="{NS_WP}" xmlns:a="{NS_A}" xmlns:pic="{NS_PIC}">
  <w:body>
    {body_xml}
    <w:sectPr>
      <w:pgSz w:w="11906" w:h="16838"/>
      <w:pgMar w:top="1150" w:right="950" w:bottom="1150" w:left="950" w:header="720" w:footer="720" w:gutter="0"/>
    </w:sectPr>
  </w:body>
</w:document>
"""


def styles_xml() -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="{NS_W}">
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal">
    <w:name w:val="Normal"/>
    <w:pPr><w:spacing w:after="120" w:line="360" w:lineRule="auto"/></w:pPr>
    <w:rPr><w:rFonts w:ascii="Arial" w:hAnsi="Arial" w:eastAsia="SimSun"/><w:sz w:val="22"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Title">
    <w:name w:val="Title"/>
    <w:pPr><w:spacing w:before="120" w:after="260"/></w:pPr>
    <w:rPr><w:b/><w:color w:val="1F4E79"/><w:rFonts w:ascii="Arial" w:hAnsi="Arial" w:eastAsia="SimHei"/><w:sz w:val="36"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading1">
    <w:name w:val="heading 1"/>
    <w:basedOn w:val="Normal"/>
    <w:pPr><w:spacing w:before="260" w:after="120"/><w:keepNext/></w:pPr>
    <w:rPr><w:b/><w:color w:val="1F4E79"/><w:rFonts w:ascii="Arial" w:hAnsi="Arial" w:eastAsia="SimHei"/><w:sz w:val="28"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading2">
    <w:name w:val="heading 2"/>
    <w:basedOn w:val="Normal"/>
    <w:pPr><w:spacing w:before="180" w:after="100"/><w:keepNext/></w:pPr>
    <w:rPr><w:b/><w:color w:val="4F6F8F"/><w:rFonts w:ascii="Arial" w:hAnsi="Arial" w:eastAsia="SimHei"/><w:sz w:val="24"/></w:rPr>
  </w:style>
  <w:style w:type="table" w:styleId="TableGrid">
    <w:name w:val="Table Grid"/>
    <w:tblPr><w:tblBorders>
      <w:top w:val="single" w:sz="4" w:space="0" w:color="BFBFBF"/>
      <w:left w:val="single" w:sz="4" w:space="0" w:color="BFBFBF"/>
      <w:bottom w:val="single" w:sz="4" w:space="0" w:color="BFBFBF"/>
      <w:right w:val="single" w:sz="4" w:space="0" w:color="BFBFBF"/>
      <w:insideH w:val="single" w:sz="4" w:space="0" w:color="BFBFBF"/>
      <w:insideV w:val="single" w:sz="4" w:space="0" w:color="BFBFBF"/>
    </w:tblBorders></w:tblPr>
  </w:style>
</w:styles>
"""


def rels_xml(image_rels: list[tuple[str, Path]]) -> str:
    rels = [
        '<Relationship Id="rIdStyles" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
    ]
    for rel_id, path in image_rels:
        rels.append(
            f'<Relationship Id="{rel_id}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/{path.name}"/>'
        )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        + "".join(rels)
        + "</Relationships>"
    )


def content_types_xml(image_rels: list[tuple[str, Path]]) -> str:
    image_defaults = '<Default Extension="png" ContentType="image/png"/>'
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  {image_defaults}
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
</Types>
"""


def root_rels_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>
"""


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    md_path = REPORT_DIR / "final_report.md"
    markdown_text = md_path.read_text(encoding="utf-8")
    body_xml, image_rels = parse_markdown(markdown_text)

    with zipfile.ZipFile(DOCX_PATH, "w", compression=zipfile.ZIP_DEFLATED) as docx:
        docx.writestr("[Content_Types].xml", content_types_xml(image_rels))
        docx.writestr("_rels/.rels", root_rels_xml())
        docx.writestr("word/document.xml", document_xml(body_xml))
        docx.writestr("word/styles.xml", styles_xml())
        docx.writestr("word/_rels/document.xml.rels", rels_xml(image_rels))
        for _, path in image_rels:
            docx.write(path, f"word/media/{path.name}")

    print(f"Wrote {DOCX_PATH}")


if __name__ == "__main__":
    main()
