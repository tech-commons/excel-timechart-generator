import load_timing_excel as ld
import simulate as sim
import draw_timing as dr
from openpyxl import Workbook
from openpyxl.utils import get_column_letter
from collections import defaultdict


EMPTY_ROW_HEIGHT = 5

def split_hier(sig):
    if "__" in sig:
        inst, local = sig.split("__", 1)
        return inst, local
    return None, sig

def set_wave_column_width(ws, col_start, num_cycles, width=2.8):
    for c in range(col_start, col_start + num_cycles):
        ws.column_dimensions[get_column_letter(c)].width = width

wb = Workbook()
ws = wb.active
ws.title = "Timing"

waves, logic, draw_order = ld.load_timing_excel("input.xlsx")
waves_all = sim.simulate(waves, logic)

start_row = 2

groups = defaultdict(list)

for sig, data in waves_all.items():
    inst, local = split_hier(sig)
    groups[inst].append((sig, data))

row = start_row
first = True

current_inst = None
inst_start_row = None

for idx, (item_type, name) in enumerate(draw_order):

    # ----------------------------
    # instance header
    # ----------------------------
    if item_type == "instance_header":

        # もし前のインスタンスがあれば閉じる
        if current_inst is not None:
            inst_end = row - 1
            ws.row_dimensions.group(
                inst_start_row + 1,
                inst_end,
                outline_level=1,
                hidden=True
            )

        ws.cell(row=row, column=1, value=f"[{name}]")

        current_inst = name
        inst_start_row = row

        row += 1
        continue

    # ----------------------------
    # signal
    # ----------------------------
    bw, values = waves_all[name]

    ws.cell(row=row, column=1, value=name)

    if first:
        set_wave_column_width(ws, 6, len(values), width=2.8)
        first = False

    dr.draw_wave(ws, row, 6, values, bw != 1)

    empty_row = row + 1
    ws.row_dimensions[empty_row].height = EMPTY_ROW_HEIGHT

    row += 2

    # ----------------------------
    # 次がinstance_headerなら今のを閉じる
    # ----------------------------
    next_is_header = (
        idx + 1 < len(draw_order)
        and draw_order[idx + 1][0] == "instance_header"
    )

    if next_is_header and current_inst is not None:
        inst_end = row - 1
        ws.row_dimensions.group(
            inst_start_row + 1,
            inst_end,
            outline_level=1,
            hidden=True
        )
        current_inst = None


# ----------------------------
# 最後のインスタンスを閉じる
# ----------------------------
if current_inst is not None:
    inst_end = row - 1
    ws.row_dimensions.group(
        inst_start_row + 1,
        inst_end,
        outline_level=1,
        hidden=True
    )



wb.save("timing_out.xlsx")

