"""Synthetic Qiskit programs, grouped into difficulty tiers.

The tiers are about how much the *code* hides the circuit, not how big the
circuit is:

  1 flat       - one gate per line; reading == transcribing
  2 loops      - for-loops over qubits/layers must be unrolled mentally
  3 control    - helper functions, Python if/else, nested loops, lists, dead code
  4 algorithms - textbook circuits (GHZ, QFT, Bernstein-Vazirani, Grover, ...)

Every program ends with a QuantumCircuit in the variable `qc`.
"""

import random

HEADER = "from math import pi\nfrom qiskit import QuantumCircuit\n"

ONE_Q = ["h", "x", "y", "z", "s", "t", "sdg", "tdg"]
ONE_Q_ROT = ["rx", "ry", "rz"]
TWO_Q = ["cx", "cz", "swap"]
TWO_Q_ROT = ["cp", "crz"]
ANGLES = ["pi", "pi / 2", "pi / 4", "pi / 8", "-pi / 2", "-pi / 4", "pi / 3", "2 * pi / 3"]


def angle(rng: random.Random) -> str:
    return rng.choice(ANGLES) if rng.random() < 0.7 else f"{rng.uniform(-3, 3):.2f}"


def random_gate(rng: random.Random, n: int) -> str:
    """One gate call on random literal qubits."""
    kinds = ["1q"] * 4 + ["rot"] * 2 + (["2q"] * 3 + ["2rot"] if n >= 2 else []) + (["ccx"] if n >= 3 else [])
    kind = rng.choice(kinds)
    if kind == "1q":
        return f"qc.{rng.choice(ONE_Q)}({rng.randrange(n)})"
    if kind == "rot":
        return f"qc.{rng.choice(ONE_Q_ROT)}({angle(rng)}, {rng.randrange(n)})"
    if kind == "ccx":
        a, b, c = rng.sample(range(n), 3)
        return f"qc.ccx({a}, {b}, {c})"
    a, b = rng.sample(range(n), 2)
    if kind == "2q":
        return f"qc.{rng.choice(TWO_Q)}({a}, {b})"
    return f"qc.{rng.choice(TWO_Q_ROT)}({angle(rng)}, {a}, {b})"


def indent(lines: list[str], level: int = 1) -> list[str]:
    return ["    " * level + line for line in lines]


# --------------------------------------------------------------------------- tier 1

def tier1_flat(rng: random.Random) -> str:
    n = rng.randint(2, 5)
    measure = rng.random() < 0.3
    lines = [f"qc = QuantumCircuit({n}, {n})" if measure else f"qc = QuantumCircuit({n})"]
    lines += [random_gate(rng, n) for _ in range(rng.randint(3, 14))]
    if measure:
        lines += [f"qc.measure({q}, {q})" for q in sorted(rng.sample(range(n), rng.randint(1, n)))]
    return HEADER + "\n" + "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- tier 2

def _loop_block(rng: random.Random, nv: str) -> list[str]:
    g1, rot, g2 = rng.choice(ONE_Q), rng.choice(ONE_Q_ROT), rng.choice(["cx", "cz"])
    options = [
        [f"for i in range({nv}):", f"    qc.{g1}(i)"],
        [f"for i in range({nv}):", f"    qc.{rot}({angle(rng)}, i)"],
        [f"for i in range({nv} - 1):", f"    qc.{g2}(i, i + 1)"],
        [f"for i in reversed(range({nv} - 1)):", f"    qc.{g2}(i, i + 1)"],
        [f"for i in range(0, {nv} - 1, 2):", f"    qc.{g2}(i, i + 1)"],
        [f"for i in range(1, {nv} - 1, 2):", f"    qc.{g2}(i, i + 1)"],
        [f"for i in range({nv}):", f"    qc.cx(i, (i + 1) % {nv})"],
        [f"for i in range(1, {nv}):", f"    qc.cx(0, i)"],
        [f"for i in range({nv}):", f"    qc.rz(pi / (i + 1), i)"],
        [f"qc.{g1}({nv} - 1)"],
    ]
    return rng.choice(options)


def tier2_loops(rng: random.Random) -> str:
    nv = rng.choice(["n", "num_qubits", "width"])
    n = rng.randint(3, 5)
    lines = [f"{nv} = {n}", f"qc = QuantumCircuit({nv})"]
    if rng.random() < 0.4:  # repeated layers
        reps = rng.randint(2, 3)
        body = _loop_block(rng, nv) + _loop_block(rng, nv)
        lines += [f"for layer in range({reps}):"] + indent(body)
    for _ in range(rng.randint(1, 3)):
        lines += _loop_block(rng, nv)
    return HEADER + "\n" + "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- tier 3

def tier3_control(rng: random.Random) -> str:
    n = rng.randint(3, 5)
    lines = [f"n = {n}"]
    pieces = rng.sample(["helper", "parity", "flag", "nested", "angles", "while"], k=rng.randint(2, 3))

    if "helper" in pieces:
        lines = [
            "def entangle(qc, a, b):",
            rng.choice(["    qc.h(a)", "    qc.ry(pi / 4, a)"]),
            f"    qc.{rng.choice(['cx', 'cz'])}(a, b)",
            "",
        ] + lines
    lines.append("qc = QuantumCircuit(n)")

    for piece in pieces:
        if piece == "helper":
            lines += rng.choice([
                ["for i in range(0, n - 1, 2):", "    entangle(qc, i, i + 1)"],
                ["entangle(qc, 0, n - 1)"],
                ["for i in range(n - 1):", "    entangle(qc, i + 1, i)"],
            ])
        elif piece == "parity":
            a, b = rng.sample(ONE_Q_ROT, 2)
            lines += ["for i in range(n):", "    if i % 2 == 0:", f"        qc.{a}({angle(rng)}, i)",
                      "    else:", f"        qc.{b}({angle(rng)}, i)"]
        elif piece == "flag":
            flag = rng.choice([True, False])
            lines += [f"add_swaps = {flag}", "if add_swaps:", "    qc.swap(0, n - 1)",
                      "else:", f"    qc.{rng.choice(ONE_Q)}(n - 1)"]
        elif piece == "nested":
            lines += ["for i in range(n):", "    for j in range(i + 1, n):",
                      f"        qc.{rng.choice(['cz', 'cx'])}(i, j)"]
        elif piece == "angles":
            vals = ", ".join(f"{rng.uniform(0, 3):.2f}" for _ in range(rng.randint(2, n)))
            lines += [f"angles = [{vals}]", "for q, theta in enumerate(angles):",
                      f"    qc.{rng.choice(ONE_Q_ROT)}(theta, q)"]
        elif piece == "while":
            lines += ["q = n - 1", "while q > 0:", f"    qc.cx(q, q - 1)", f"    q -= {rng.choice([1, 2])}"]
    return HEADER + "\n" + "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- tier 4

def _bits(rng: random.Random, k: int) -> str:
    return "".join(rng.choice("01") for _ in range(k))


def tier4_algorithms(rng: random.Random) -> str:
    algo = rng.choice(["ghz", "qft", "bv", "grover", "ansatz", "dj"])
    if algo == "ghz":
        body = f"""n = {rng.randint(3, 6)}
qc = QuantumCircuit(n, n)
qc.h(0)
for i in range(n - 1):
    qc.cx(i, i + 1)
qc.measure(range(n), range(n))
"""
    elif algo == "qft":
        body = f"""n = {rng.randint(3, 5)}
qc = QuantumCircuit(n)
for j in range(n):
    qc.h(j)
    for k in range(j + 1, n):
        qc.cp(pi / 2 ** (k - j), k, j)
for j in range(n // 2):
    qc.swap(j, n - j - 1)
"""
    elif algo == "bv":
        body = f"""secret = "{_bits(rng, rng.randint(3, 5))}"
n = len(secret)
qc = QuantumCircuit(n + 1, n)
qc.x(n)
qc.h(range(n + 1))
for i, bit in enumerate(reversed(secret)):
    if bit == "1":
        qc.cx(i, n)
qc.h(range(n))
qc.measure(range(n), range(n))
"""
    elif algo == "grover":
        body = f"""marked = "{_bits(rng, 2)}"
qc = QuantumCircuit(2)
qc.h([0, 1])
# oracle: flip the phase of |marked>
for i, bit in enumerate(reversed(marked)):
    if bit == "0":
        qc.x(i)
qc.cz(0, 1)
for i, bit in enumerate(reversed(marked)):
    if bit == "0":
        qc.x(i)
# diffuser
qc.h([0, 1])
qc.x([0, 1])
qc.cz(0, 1)
qc.x([0, 1])
qc.h([0, 1])
"""
    elif algo == "ansatz":
        body = f"""n = {rng.randint(3, 4)}
reps = {rng.randint(1, 2)}
thetas = [0.1 * (k + 1) for k in range(n * (reps + 1))]
qc = QuantumCircuit(n)
k = 0
for r in range(reps + 1):
    for q in range(n):
        qc.ry(thetas[k], q)
        k += 1
    if r < reps:
        for q in range(n - 1):
            qc.cx(q, q + 1)
"""
    else:  # Deutsch-Jozsa with a balanced oracle given by a bit mask
        body = f"""mask = "{_bits(rng, rng.randint(3, 4))}"
n = len(mask)
qc = QuantumCircuit(n + 1, n)
qc.x(n)
for q in range(n + 1):
    qc.h(q)
for i, bit in enumerate(mask):
    if bit == "1":
        qc.x(i)
for i in range(n):
    qc.cx(i, n)
for i, bit in enumerate(mask):
    if bit == "1":
        qc.x(i)
for q in range(n):
    qc.h(q)
    qc.measure(q, q)
"""
    return HEADER + "\n" + body


TIERS = {1: tier1_flat, 2: tier2_loops, 3: tier3_control, 4: tier4_algorithms}


def generate_program(rng: random.Random, tier: int) -> str:
    return TIERS[tier](rng)
