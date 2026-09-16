import sys
import zipfile
import os

sys.stdout.reconfigure(encoding="utf-8")

ROOT = r"D:\Projects\-FURP-2026-Frank-VRPTW_project-main"
TEMPLATE = os.path.join(ROOT, "Poster", "Poster Template_2026 FoSE UG PGT Showcase.pptx")
OUTPUT = os.path.join(ROOT, "Poster", "Poster_Filled.pptx")
FIGURE = os.path.join(ROOT, "results", "EVRPTW_ESOGU_C60", "MIN_BATTERY_CAPACITY921", "Figure_1.png")

DARK = "10263B"
BODY = "333333"
GRAY = "777777"
WHITE = "FFFFFF"

# Portrait template page size stays unchanged
W = 21396325
H = 30267275


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def run(text, size, bold=False, color=BODY, italic=False):
    style = f'<a:rPr lang="en-US" sz="{size}" b="1" i="0">' if bold else (
        f'<a:rPr lang="en-US" sz="{size}" b="0" i="1">' if italic else
        f'<a:rPr lang="en-US" sz="{size}" b="0" i="0">'
    )
    style += (
        f'<a:solidFill><a:srgbClr val="{color}"/></a:solidFill>'
        '<a:latin typeface="Calibri" pitchFamily="34" charset="0"/>'
        "</a:rPr>"
    )
    return f"<a:r>{style}<a:t>{esc(text)}</a:t></a:r>"


def para(text, size=2000, bold=False, color=BODY, bullet=False, align="l", italic=False):
    bu = '<a:buFont typeface="Arial"/><a:buChar char="•"/>' if bullet else ""
    al = {"l": "l", "c": "ctr", "r": "r"}[align]
    ppr = (
        f'<a:pPr marL="274320" indent="-274320" algn="{al}">'
        f"{bu}</a:pPr>"
    )
    return f"<a:p>{ppr}{run(text, size, bold, color, italic)}</a:p>"


def textbox(x, y, w, h, paras_xml, shape_id, name="TextBox"):
    return (
        f'<p:sp><p:nvSpPr><p:cNvPr id="{shape_id}" name="{name}"/>'
        '<p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>'
        f'<p:spPr><a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{w}" cy="{h}"/></a:xfrm>'
        '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/><a:ln><a:noFill/></a:ln></p:spPr>'
        '<p:txBody><a:bodyPr wrap="square" lIns="45720" tIns="27432" rIns="45720" bIns="27432"/>'
        f"<a:lstStyle/>{paras_xml}</p:txBody></p:sp>"
    )


def table(x, y, w, col_widths, rows, shape_id, name="Table", row_h=400000, font_size=1600):
    grid = "".join(f'<a:gridCol w="{cw}"/>' for cw in col_widths)
    trs = ""
    for ri, row in enumerate(rows):
        tcs = ""
        for cell in row:
            bold = ri == 0
            fill = f'<a:solidFill><a:srgbClr val="{DARK if ri == 0 else "F2F2F2" if ri % 2 else "FFFFFF"}"/></a:solidFill>'
            tcs += (
                f'<a:tc><a:txBody><a:bodyPr/><a:p>'
                f'<a:pPr algn="ctr"/>'
                f"{run(cell, font_size, bold, WHITE if ri == 0 else BODY)}"
                "</a:p></a:txBody>"
                f'<a:tcPr marL="45720" marR="45720" marT="18288" marB="18288">{fill}'
                '<a:lnL><a:noFill/></a:lnL><a:lnR><a:noFill/></a:lnR><a:lnT><a:noFill/></a:lnT><a:lnB><a:noFill/></a:lnB></a:tcPr></a:tc>'
            )
        trs += f'<a:tr h="{row_h}">{tcs}</a:tr>'
    tbl = (
        '<a:tbl><a:tblPr firstRow="1" bandRow="1"><a:tableStyleId>{5C22544A-7EE6-4342-B048-85BDC9FD1C3A}</a:tableStyleId></a:tblPr>'
        f"<a:tblGrid>{grid}</a:tblGrid>{trs}</a:tbl>"
    )
    return (
        f'<p:graphicFrame><p:nvGraphicFramePr><p:cNvPr id="{shape_id}" name="{name}"/>'
        '<p:cNvGraphicFramePr/><p:nvPr/></p:nvGraphicFramePr>'
        f'<p:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{w}" cy="{len(rows) * row_h}"/></p:xfrm>'
        '<a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/table">'
        f"{tbl}</a:graphicData></a:graphic></p:graphicFrame>"
    )


def picture(x, y, cx, cy, rid, shape_id, name="Figure"):
    return (
        f'<p:pic><p:nvPicPr><p:cNvPr id="{shape_id}" name="{name}"/>'
        '<p:cNvPicPr><a:picLocks noChangeAspect="1"/></p:cNvPicPr><p:nvPr/></p:nvPicPr>'
        f'<p:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></p:blipFill>'
        f'<p:spPr><a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
        '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/><a:ln><a:noFill/></a:ln></p:spPr></p:pic>'
    )


def heading(text, shape_id, x, y, w, h=800000, size=3000):
    return textbox(
        x, y, w, h,
        para(text, size=size, bold=True, color=DARK),
        shape_id, f"Heading {shape_id}",
    )


def build():
    with zipfile.ZipFile(TEMPLATE, "r") as zin:
        names = zin.namelist()
        data = {n: zin.read(n) for n in names}

    slide = data["ppt/slides/slide1.xml"].decode("utf-8")
    new_shapes = []
    sid = 2000

    # ---- Title band (inside template top banner) ----
    title_xml = (
        para("Vehicle Routing Optimization", size=4400, bold=True, color=WHITE, align="c")
        + para("From Heuristics to Deep Reinforcement Learning", size=2400, bold=False, color=WHITE, align="c")
        + para("A Comparative Study on CVRP, VRPTW and EVRP-TW", size=1700, bold=False, color=WHITE, align="c")
    )
    new_shapes.append(textbox(3900000, 50000, 13000000, 1650000, title_xml, sid, "Title Band"))
    sid += 1

    col_w = 9800000
    gap = 400000
    col1_x = 500000
    col2_x = col1_x + col_w + gap

    # ================= Row 1: Background | Methods =================
    r1 = 2400000
    new_shapes.append(heading("Background", sid, col1_x, r1, col_w))
    sid += 1
    bg = (
        para("Vehicle routing: serve customers from a depot", bullet=True)
        + para("CVRP: limited vehicle load", bullet=True)
        + para("VRPTW: customer time windows", bullet=True)
        + para("EVRP-TW: battery and charging stations", bullet=True)
        + para("Goal: minimum total travel distance", bullet=True)
    )
    new_shapes.append(textbox(col1_x, r1 + 950000, col_w, 4000000, bg, sid, "Background Bullets"))
    sid += 1

    new_shapes.append(heading("Methods", sid, col2_x, r1, col_w))
    sid += 1
    mt = (
        para("OR-Tools: route model with dimensions", bullet=True)
        + para("load / time / energy limits", bullet=True, size=1600, color=GRAY)
        + para("PyVRP: population-based search", bullet=True)
        + para("POMO: multiple starts, shared baseline", bullet=True)
        + para("RL4CO: PyTorch + Lightning framework", bullet=True)
    )
    new_shapes.append(textbox(col2_x, r1 + 950000, col_w, 4200000, mt, sid, "Method Bullets"))
    sid += 1

    # ================= Row 2: Experiments tables | Figure =================
    r2 = 10800000

    # Left block: experiment tables
    new_shapes.append(heading("Experiment 1: CVRP Comparison", sid, col1_x, r2, col_w, size=2600))
    sid += 1
    rows1 = [
        ["Instance", "Solver", "Objective", "Gap"],
        ["A-n32-k5", "PyVRP", "784", "0%"],
        ["A-n32-k5", "OR-Tools", "784", "0%"],
        ["A-n55-k9", "PyVRP", "1073", "0%"],
        ["A-n55-k9", "OR-Tools", "1110", "3.45%"],
    ]
    new_shapes.append(table(col1_x, r2 + 900000, col_w, [2600000, 2800000, 2400000, 2000000], rows1, sid, "CVRP Table", row_h=360000))
    sid += 1
    new_shapes.append(textbox(
        col1_x, r2 + 2800000, col_w, 600000,
        para("Known optima: 784 / 1073. PyVRP reaches both; OR-Tools gap on the medium instance.", size=1500, color=GRAY, italic=True),
        sid, "CVRP Caption",
    ))
    sid += 1

    new_shapes.append(heading("Experiment 2: Battery Sensitivity", sid, col1_x, r2 + 3600000, col_w, size=2600))
    sid += 1
    rows2 = [
        ["Instance", "Battery", "Feasible", "Objective", "Stations"],
        ["C20", "no limit", "Yes", "12939", "0"],
        ["C20", "697 (critical)", "Yes", "12939", "0"],
        ["C20", "696", "No", "-", "-"],
        ["C60", "no limit", "Yes", "30142", "2"],
        ["C60", "921 (critical)", "Yes", "30520", "2"],
        ["C60", "920", "No", "-", "-"],
    ]
    new_shapes.append(table(
        col1_x, r2 + 4500000, col_w,
        [1500000, 2200000, 1800000, 2000000, 1700000],
        rows2, sid, "Battery Table", row_h=320000, font_size=1500,
    ))
    sid += 1
    new_shapes.append(textbox(
        col1_x, r2 + 6900000, col_w, 800000,
        para("Critical capacity: 697 (C20), 921 (C60). Objective +1.25% at C60. Stations also serve as time adjustment points.", size=1500, color=GRAY, italic=True),
        sid, "Battery Caption",
    ))
    sid += 1

    # Right block: figure + research line
    new_shapes.append(heading("Route at Critical Battery", sid, col2_x, r2, col_w, size=2600))
    sid += 1
    fig_w = 9200000
    fig_h = int(fig_w * 962 / 1706)
    new_shapes.append(picture(col2_x + 300000, r2 + 900000, fig_w, fig_h, "rId3", sid, "Figure 1"))
    sid += 1
    new_shapes.append(textbox(
        col2_x, r2 + 900000 + fig_h + 100000, col_w, 600000,
        para("Figure: C60 route at critical battery (921 kWh), including two charging station visits.", size=1500, color=GRAY, italic=True),
        sid, "Figure Caption",
    ))
    sid += 1

    new_shapes.append(heading("Research Line", sid, col2_x, r2 + 7000000, col_w, size=2600))
    sid += 1
    rl = (
        para("Traditional solvers: OR-Tools, PyVRP", bullet=True)
        + para("Learning-based method: POMO + RL4CO", bullet=True)
        + para("Datasets: Augerat CVRP, ESOGU EVRP-TW", bullet=True)
    )
    new_shapes.append(textbox(col2_x, r2 + 7900000, col_w, 2200000, rl, sid, "Research Line Bullets"))
    sid += 1

    # ================= Row 3: POMO | Status + Future =================
    r3 = 20400000

    new_shapes.append(heading("POMO + RL4CO", sid, col1_x, r3, col_w, size=2600))
    sid += 1
    pm = (
        para("Custom environment: src/POMO_Test_2.py", bullet=True)
        + para("generator: random instances", bullet=True, size=1600, color=GRAY)
        + para("env: reset / step / reward", bullet=True, size=1600, color=GRAY)
        + para("mask: hard constraints", bullet=True, size=1600, color=GRAY)
        + para("multi-start node selection", bullet=True, size=1600, color=GRAY)
        + para("Solution check included", bullet=True)
    )
    new_shapes.append(textbox(col1_x, r3 + 950000, col_w, 3400000, pm, sid, "POMO Bullets"))
    sid += 1

    new_shapes.append(heading("Baseline Test: TSP20", sid, col1_x, r3 + 4600000, col_w, size=2600))
    sid += 1
    ts = (
        para("GPU available: True", bullet=True)
        + para("Training reward: -4.017", bullet=True)
        + para("Validation reward: -3.952", bullet=True)
        + para("Model parameters: 1.3 M", bullet=True)
    )
    new_shapes.append(textbox(col1_x, r3 + 5550000, col_w, 2600000, ts, sid, "TSP20 Bullets"))
    sid += 1

    new_shapes.append(heading("Current Status", sid, col2_x, r3, col_w, size=2600))
    sid += 1
    st = (
        para("Environment ready, basic functions tested", bullet=True)
        + para("POMO training on EVRP-TW: next step", bullet=True)
    )
    new_shapes.append(textbox(col2_x, r3 + 950000, col_w, 1800000, st, sid, "Status Bullets"))
    sid += 1

    new_shapes.append(heading("Future Work", sid, col2_x, r3 + 3000000, col_w, size=2600))
    sid += 1
    ft = (
        para("Train POMO on EVRP-TW instances", bullet=True)
        + para("Compare with OR-Tools baseline", bullet=True)
        + para("Real charging behavior at stations", bullet=True)
        + para("Use ESOGU real data", bullet=True)
    )
    new_shapes.append(textbox(col2_x, r3 + 3950000, col_w, 2800000, ft, sid, "Future Bullets"))
    sid += 1

    # ================= Bottom conclusion =================
    new_shapes.append(textbox(
        500000, 28400000, W - 1000000, 700000,
        para(
            "Conclusion: PyVRP strong on classical CVRP; OR-Tools flexible for EVRP-TW; learning-based environment ready for the next stage.",
            size=1900, bold=True, color=DARK, align="c",
        ),
        sid, "Conclusion",
    ))
    sid += 1

    insert_xml = "".join(new_shapes)
    slide = slide.replace("</p:spTree>", insert_xml + "</p:spTree>")
    data["ppt/slides/slide1.xml"] = slide.encode("utf-8")

    rels = data["ppt/slides/_rels/slide1.xml.rels"].decode("utf-8")
    rels = rels.replace(
        "</Relationships>",
        '<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="../media/figure1.png"/></Relationships>',
    )
    data["ppt/slides/_rels/slide1.xml.rels"] = rels.encode("utf-8")

    with open(FIGURE, "rb") as f:
        data["ppt/media/figure1.png"] = f.read()

    with zipfile.ZipFile(OUTPUT, "w", zipfile.ZIP_DEFLATED) as zout:
        for n in data:
            zout.writestr(n, data[n])

    print("saved:", OUTPUT)


if __name__ == "__main__":
    build()
