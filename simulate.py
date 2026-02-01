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

    def visit_Compare(self, node):
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise ValueError("Only single comparison supported")

        left = self.visit(node.left)
        right = self.visit(node.comparators[0])
        op = node.ops[0]

        if isinstance(op, ast.Eq):
            return int(left == right)
        elif isinstance(op, ast.NotEq):
            return int(left != right)
        elif isinstance(op, ast.Lt):
            return int(left < right)
        elif isinstance(op, ast.LtE):
            return int(left <= right)
        elif isinstance(op, ast.Gt):
            return int(left > right)
        elif isinstance(op, ast.GtE):
            return int(left >= right)
        else:
            raise ValueError("Unsupported comparison operator")

    def visit_IfExp(self, node):
        cond = self.visit(node.test)
        if cond:
            return self.visit(node.body)
        else:
            return self.visit(node.orelse)


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

def preprocess_ternary(expr: str) -> str:
    """
    A ? B : C  →  (B if A else C)
    """
    # 単純版（ネストなし想定）
    m = re.search(r'(.+?)\?(.+?):(.+)', expr)
    if not m:
        return expr

    cond = m.group(1).strip()
    true_expr = m.group(2).strip()
    false_expr = m.group(3).strip()

    return f"({true_expr} if {cond} else {false_expr})"

def preprocess_expr(expr: str) -> str:
    expr = preprocess_ternary(expr)
    expr = preprocess_delay(expr)
    return expr

def is_reg_expr(expr: str) -> bool:
    return expr.strip().startswith("REG(")

def parse_reg(expr: str):
    """
    REG(set=S, clr=C, en=E, d=D, prio="clr>set>load>hold")
    """
    m = re.match(r"REG\((.*)\)", expr.strip())
    if not m:
        raise ValueError("Invalid REG syntax")

    args = {}
    for item in m.group(1).split(","):
        k, v = item.split("=", 1)
        args[k.strip()] = v.strip().strip('"')

    return args


def eval_reg(args, env, prev_val):
    prio = args.get("prio", "clr>set>load>hold").split(">")

    for p in prio:
        p = p.strip()

        if p == "clr":
            cond = args.get("clr")
            if cond and eval_expr_safe(cond, env):
                return 0

        elif p == "set":
            cond = args.get("set")
            if cond and eval_expr_safe(cond, env):
                return 1

        elif p == "load":
            en = args.get("en", "1")   # ★ en省略時は常に有効
            if eval_expr_safe(en, env):
                return eval_expr_safe(args.get("d", str(prev_val)), env)

        elif p == "hold":
            return prev_val

    return prev_val




def build_env(waves_all, t):
    """
    waves_all:
      { signal: (bit_width, [v0, v1, ...]) }
    """
    env = {}

    for sig, (bit_width, values) in waves_all.items():
        # 現在値
        env[sig] = values[t] if t < len(values) else 0

        # delay値
        for d in range(1, t + 2):
            if t - d >= 0:
                env[f"{sig}__d{d}"] = values[t - d]
            else:
                env[f"{sig}__d{d}"] = 0   # 初期値

    return env


def simulate(waves, logic):
    """
    waves: 入力信号
      { "A": (1, ["0","1",...]) }

    logic: 生成信号の論理式
      { "Q": (4, "Q@1 + D") }

    """

    # --- 初期化 ---
    waves_all = {
        sig: (bit_width, [int(float(v)) for v in values])
        for sig, (bit_width, values) in waves.items()
    }

    num_cycles = len(next(iter(waves.values()))[1])

    # 生成信号の初期値（0）
    for sig, (bit_width, _) in logic.items():
        waves_all[sig] = (bit_width, [0] * num_cycles)

    # combinational / sequential 分離
    comb = {}
    seq = {}

    for sig, (_, expr) in logic.items():
        if is_reg_expr(expr):
            seq[sig] = expr
        elif is_sequential(expr):
            seq[sig] = expr
        else:
            comb[sig] = expr

    # --- サイクルループ ---
    for t in range(num_cycles):

        env = build_env(waves_all, t)
        #print("env =", env)

        # --- combinational ---
        for sig, expr in comb.items():
            e = preprocess_expr(expr)
            val = eval_expr_safe(e, env)
            bit_width, values = waves_all[sig]
            val = apply_width(val, bit_width)
            values[t] = val

        # --- sequential ---
        if t + 1 < num_cycles:
            for sig, expr in seq.items():
                bit_width, values = waves_all[sig]

                if is_reg_expr(expr):
                    args = parse_reg(expr)
                    prev = values[t]
                    val = eval_reg(args, env, prev)
                else:
                    e = preprocess_expr(expr)
                    val = eval_expr_safe(e, env)

                val = apply_width(val, bit_width)
                values[t + 1] = val


    #print("cnt =", waves_all["cnt"])
    return waves_all

def apply_width(value, width):
    if width is None:
        return value
    return value & ((1 << width) - 1)

