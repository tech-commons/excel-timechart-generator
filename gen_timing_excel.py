import load_timing_excel as ld
import simulate as sim
import draw_timing as dr
from openpyxl import Workbook

wb = Workbook()
ws = wb.active
ws.title = "Timing"

waves, logic = ld.load_timing_excel("input.xlsx")
waves_all = sim.simulate(waves, logic)

#print(waves)
#print(logic)
#print(waves_all)

row = 2
for sig, (bit_width, values) in waves_all.items():
    ws.cell(row=row, column=1, value=sig)
    print("bit_width=" + str(bit_width))
    dr.draw_wave(ws, row, 3, values, (bit_width!=1))
    row += 2   # ★ 1行空ける

wb.save("timing_out.xlsx")

