import ast
import operator

class SafeEvaluator(ast.NodeVisitor):
    BIN_OPS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.BitAnd: operator.and_,
        ast.BitOr: operator.or_,
        ast.LShift: operator.lshift,
        ast.RShift: operator.rshift,
    }

    UNARY_OPS = {
        ast.USub: operator.neg,
        ast.Invert: operator.invert,
    }

    def __init__(self, env):
        self.env = env

    def visit_Expression(self, node):
        return self.visit(node.body)

    def visit_Constant(self, node):
        return node.value

    def visit_Name(self, node):
        if node.id not in self.env:
            raise ValueError(f"Undefined signal: {node.id}")
        return self.env[node.id]

    def visit_BinOp(self, node):
        op_type = type(node.op)
        if op_type not in self.BIN_OPS:
            raise ValueError(f"Unsupported operator: {op_type}")
        left = self.visit(node.left)
        right = self.visit(node.right)
        return self.BIN_OPS[op_type](left, right)

    def visit_UnaryOp(self, node):
        op_type = type(node.op)
        if op_type not in self.UNARY_OPS:
            raise ValueError(f"Unsupported unary operator: {op_type}")
        operand = self.visit(node.operand)
        return self.UNARY_OPS[op_type](operand)

    def generic_visit(self, node):
        raise ValueError(f"Unsupported syntax: {type(node).__name__}")

def eval_expr_safe(expr, env):
    tree = ast.parse(expr, mode="eval")
    evaluator = SafeEvaluator(env)
    return evaluator.visit(tree)

import re

def is_sequential(expr: str) -> bool:
    return bool(re.search(r'@\d+', expr))


def preprocess_delay(expr: str) -> str:
    return re.sub(r'([A-Za-z_]\w*)@(\d+)',
                  r'\1__d\2',
                  expr)



def build_env(waves_all, t):
    """
    waves_all:
      { signal: (bit_width, [v0, v1, ...]) }
    """
    env = {}

    for sig, (bit_width, values) in waves_all.items():
        if t < len(values):
            env[sig] = values[t]

        for d in range(1, t + 1):
            env[f"{sig}__d{d}"] = values[t - d]

    return env


def simulate(waves, logic):
    """
    waves: 入力信号
      { "A": (1, ["0","1",...]) }

    logic: 生成信号の論理式
      { "Q": "Q@1 + D" }

    """

    # --- 初期化 ---
    waves_all = {
        sig: (bit_width, [int(float(v)) for v in values])
        for sig, (bit_width, values) in waves.items()
    }

    num_cycles = len(next(iter(waves.values()))[1])

    # 生成信号の初期値（0）
    for sig in logic:
        waves_all[sig] = (1, [0] * num_cycles)

    # combinational / sequential 分離
    comb = {}
    seq = {}

    for sig, expr in logic.items():
        if is_sequential(expr):
            seq[sig] = expr
        else:
            comb[sig] = expr

    # --- サイクルループ ---
    for t in range(num_cycles):

        env = build_env(waves_all, t)

        # --- combinational ---
        for sig, expr in comb.items():
            e = preprocess_delay(expr)
            val = eval_expr_safe(e, env)
            bit_width, values = waves_all[sig]
            values[t] = val
            #waves_all[sig][t] = apply_width(
            #    val, widths.get(sig)
            #) if widths else val

        # --- sequential ---
        if t + 1 < num_cycles:
            for sig, expr in seq.items():
                e = preprocess_delay(expr)
                val = eval_expr_safe(e, env)
                bit_width, values = waves_all[sig]
                values[t + 1] = val
                #waves_all[sig][t + 1] = apply_width(
                #    val, widths.get(sig)
                #) if widths else val

    return waves_all

def apply_width(value, width):
    if width is None:
        return value
    return value & ((1 << width) - 1)

