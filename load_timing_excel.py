import pandas as pd
import re

HIER_SEP = "__"

def load_timing_excel(
    filename,
    top_sheet="TOP",
    signal_col="signal",
    expr_col="expr",
    bit_width_col="bit_width",
    fill_init=0
):

    xls = pd.ExcelFile(filename)

    # ----------------------------
    # サブモジュール定義
    # ----------------------------
    submodules = {}

    for sheet in xls.sheet_names:
        if sheet == top_sheet:
            continue

        df = pd.read_excel(xls, sheet)
        mod = {}

        for _, row in df.iterrows():
            sig = row[signal_col]
            bw_val = row.get(bit_width_col)

            if pd.isna(bw_val):
                bw = None
            else:
                bw = int(bw_val)

            expr = row[expr_col]
            mod[sig] = (bw, expr)

        submodules[sheet] = mod

    # ----------------------------
    # TOP読み込み
    # ----------------------------
    df = pd.read_excel(xls, top_sheet)
    cycle_cols = df.columns[3:]

    waves = {}
    logic = {}
    draw_order = []   # ← ★ 追加（順序保持）

    for _, row in df.iterrows():

        sig  = row[signal_col]
        bw_val = row.get(bit_width_col)

        if pd.isna(bw_val):
            bw = None
        else:
            bw = int(bw_val)

        expr = row[expr_col]

        if pd.isna(sig):
            continue  # 完全空行はスキップ
        # ----------------------
        # 入力信号
        # ----------------------
        if pd.isna(expr):
            if bw is None:
                raise ValueError(f"bit_width missing for signal {sig}")

            values = (
                row[cycle_cols]
                .ffill()
                .fillna(fill_init)
            )

            waves[sig] = (
                bw,
                [str(v) for v in values]
            )

            draw_order.append(("signal", sig))
            continue

        # ----------------------
        # インスタンス判定
        # ----------------------
        m = re.match(r"(\w+)\((.*)\)", str(expr))

        if m and m.group(1) in submodules:

            mod_name, arg_str = m.groups()
            mod = submodules[mod_name]

            draw_order.append(("instance_header", sig))

            # 引数解析
            ports = {}
            if arg_str.strip():
                for a in arg_str.split(","):
                    k, v = a.split("=")
                    ports[k.strip()] = v.strip()

            for local_sig, (bw_sub, sub_expr) in mod.items():

                full_sig = f"{sig}{HIER_SEP}{local_sig}"

                e = sub_expr

                # ローカル信号置換
                for s in mod.keys():
                    e = re.sub(rf"\b{s}\b", f"{sig}{HIER_SEP}{s}", e)

                # ポート置換
                for p, net in ports.items():
                    e = re.sub(rf"\b{p}\b", net, e)

                logic[full_sig] = (bw_sub, e)

                draw_order.append(("signal", full_sig))

        else:
            # 通常生成信号
            logic[sig] = (bw, expr)
            draw_order.append(("signal", sig))

    return waves, logic, draw_order

