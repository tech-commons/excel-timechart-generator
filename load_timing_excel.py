import pandas as pd

def load_timing_excel(
    filename,
    signal_col="signal",
    expr_col="expr",
    bit_width_col="bit_width",
    fill_init=0
):
    """
    Excelタイミング定義を読み込み、
    waves（入力信号）と logic（生成信号の論理式）を返す

    Excelフォーマット：
      1列目: signal
      2列目: bit_width
      3列目: expr
      4列目以降: C0..CN

    Returns:
        waves: dict[str, tuple[int, list[str]]]
        logic: dict[str, str]
    """

    df = pd.read_excel(filename)

    # サイクル列（C0..CN）
    cycle_cols = df.columns[3:]

    # --- 入力信号（exprが空） ---
    input_df = df[df[expr_col].isna()].copy()

    # 空白セルは前値コピー（横方向）
    input_df[cycle_cols] = (
        input_df[cycle_cols]
        .ffill(axis=1)
        .fillna(fill_init)
    )

    waves = {
        row[signal_col]: (row[bit_width_col], [str(v) for v in row[cycle_cols]])
        for _, row in input_df.iterrows()
    }

    # --- 生成信号（exprあり） ---
    logic_df = df[df[expr_col].notna()]

    logic = {
        row[signal_col]: row[expr_col]
        for _, row in logic_df.iterrows()
    }

    return waves, logic

