from openpyxl import Workbook
from openpyxl.styles import Border, Side

THIN = Side(style="thin")

def make_border(p):
    return Border(
        top=THIN if p["top"] else None,
        bottom=THIN if p["bottom"] else None,
        left=THIN if p["left"] else None,
    )

def wave_policy(curr, prev, is_bus=False):
    if is_bus:
        return {
            "top": True,
            "bottom": True,
            "left": curr != prev,
            "value": str(curr),
        }
    else:
        return {
            "top": curr == 1,
            "bottom": curr == 0,
            "left": curr != prev,
            "value": None,
        }


def draw_wave(ws, row, col_start, values, is_bus=False):
    prev = values[0]

    for i, curr in enumerate(values):
        col = col_start + i
        cell = ws.cell(row=row, column=col)

        p = wave_policy(curr, prev, is_bus)
        cell.border = make_border(p)

        if p["value"] is not None:
            cell.value = p["value"]

        prev = curr

