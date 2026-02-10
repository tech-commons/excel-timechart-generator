import pandas as pd
import re

def load_timing_excel(
    filename,
    top_sheet="TOP",
    signal_col="signal",
    expr_col="expr",
    bit_width_col="bit_width",
    fill_init=0
):
    """
    Excelタイミング定義を読み込み、
    waves（入力信号）と logic（生成信号の論理式）を返す

    - TOPシート：入力信号 + インスタンス宣言
    - 他シート ：サブモジュール定義
    """

    xls = pd.ExcelFile(filename)

    # ----------------------------
    # サブモジュール定義の読み込み
    # ----------------------------
    submodules = {}

    for sheet in xls.sheet_names:
        if sheet == top_sheet:
            continue

        df = pd.read_excel(xls, sheet)
        mod = {}

        for _, row in df.iterrows():
            sig = row[signal_col]
            bw  = int(row[bit_width_col])
            expr = row[expr_col]
            mod[sig] = (bw, expr)

        submodules[sheet] = mod

    # ----------------------------
    # TOPシート読み込み
    # ----------------------------
    df = pd.read_excel(xls, top_sheet)

    cycle_cols = df.columns[3:]

    # --- 入力信号 ---
    input_df = df[df[expr_col].isna()].copy()
    input_df[cycle_cols] = (
        input_df[cycle_cols]
        .ffill(axis=1)
        .fillna(fill_init)
    )

    waves = {
        row[signal_col]: (
            int(row[bit_width_col]),
            [str(v) for v in row[cycle_cols]]
        )
        for _, row in input_df.iterrows()
    }

    # --- logic（生成信号 or インスタンス） ---
    logic = {}

    inst_df = df[df[expr_col].notna()]

    for _, row in inst_df.iterrows():
        sig  = row[signal_col]
        expr = row[expr_col]

        # インスタンス判定：MOD(...)
        m = re.match(r"(\w+)\((.*)\)", str(expr))
        if m and m.group(1) in submodules:
            mod_name, arg_str = m.groups()
            mod = submodules[mod_name]

            # 引数解析
            ports = {}
            if arg_str.strip():
                for a in arg_str.split(","):
                    k, v = a.split("=")
                    ports[k.strip()] = v.strip()

            # サブモジュール展開
            for local_sig, (bw, sub_expr) in mod.items():
                inst = sig
                HIER_SEP = "__"
                full_sig = f"{inst}{HIER_SEP}{local_sig}"

                e = sub_expr

                # ローカル信号置換
                for s in mod.keys():
                    e = re.sub(rf"\b{s}\b", f"{inst}{HIER_SEP}{s}", e)

                # ポート置換
                for p, net in ports.items():
                    e = re.sub(rf"\b{p}\b", net, e)

                logic[full_sig] = (bw, e)

        else:
            # 通常の生成信号
            logic[sig] = (int(row[bit_width_col]), expr)

    return waves, logic

