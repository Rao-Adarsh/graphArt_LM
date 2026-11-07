"""The two-step answer format: text only, no torch (the tracer is graphart/cos_trace.py).

The model sees the code with numbered lines and answers in two calls:

  step 1, nodes   the usual INPUTS/NODES section, each node line followed by the
                  source lines of the call stack that make it, outermost first:
                      4 call_module branch2.0.conv Conv2d(...) [1,96,28,28]  @64>56>16
  step 2, edges   given the code and step 1's nodes (id, op, target, stack only), one
                  line per node that has inputs, each input with the variable it was
                  read from when the code names one:
                      20 <- outputs[0]=3 arg0; outputs[1]=9 arg1
                      13 <- 12 arg0

`assemble` turns the two replies into the ordinary INPUTS/NODES/EDGES answer, so the
two-step model is scored by the unchanged scorer and classifier. That is a change of
notation only: every node and edge is the model's own.
"""

import re

from graphart.schema import ParseError, from_text, to_text


def numbered(code: str) -> str:
    """The code as the model sees it: every line with its number.

    Numbered from the code exactly as stored, so line n is line n of the source the
    ground truth was traced from.
    """
    lines = code.splitlines()
    width = len(str(len(lines)))
    return "\n".join(f"{i:>{width}}| {line}" for i, line in enumerate(lines, 1))


# --------------------------------------------------------------------------
# writing
# --------------------------------------------------------------------------
def render_step1(graph: dict, stack: dict[int, list[int]]) -> str:
    """INPUTS/NODES, each node with its source stack."""
    text = to_text(graph).split("\nEDGES")[0]
    out = []
    for line in text.splitlines():
        head = line.split(" ", 1)[0]
        if head.isdigit() and stack.get(int(head)):
            line += "  @" + ">".join(map(str, stack[int(head)]))
        out.append(line)
    return "\n".join(out)


def render_step2(graph: dict, names: dict[tuple[int, int], str]) -> str:
    """EDGES, one line per node with inputs, sources tagged with their variables."""
    by_dst: dict[int, list] = {}
    for e in graph["edges"]:
        by_dst.setdefault(e["dst"], []).append(e)
    out = ["EDGES"]
    for dst in sorted(by_dst):
        parts = []
        for e in sorted(by_dst[dst], key=lambda e: min(e["args"])):
            name = names.get((e["src"], dst))
            slots = " ".join(f"arg{a}" for a in e["args"])
            parts.append(f"{name + '=' if name else ''}{e['src']} {slots}")
        out.append(f"{dst} <- " + "; ".join(parts))
    return "\n".join(out)


# --------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------
_STACK_RE = re.compile(r"\s+@\d+(?:>\d+)*\s*$")
_PART_RE = re.compile(r"^(?:([A-Za-z_][\w.]*(?:\[\d+\])?)=)?(\d+)((?:\s+arg\d+)+)$")
_BRIEF_RE = re.compile(r"^(\d+)\s+(\w+)\s+(\S+)")


def _section(text: str, start: str, stop: str | None = None) -> str:
    """The reply from its last `start` header line to a closing code fence, or `stop`.

    The same slack the one-step scorer gives (pipeline/score.extract_graph_block): a
    preamble and a fenced block are fine; prose after an unfenced answer is an error.
    """
    lines = text.splitlines()
    starts = [i for i, line in enumerate(lines) if line.strip().startswith(start)]
    block = lines[starts[-1]:] if starts else lines
    out = []
    for line in block:
        if line.strip().startswith("```") or (stop and line.strip().startswith(stop)):
            break
        out.append(line)
    return "\n".join(out).strip()


def parse_step1(text: str) -> tuple[dict, dict[int, list[int]]]:
    """Graph with nodes only, and the stack read off each node line."""
    lines, stack = [], {}
    for line in _section(text, "INPUTS", "EDGES").splitlines():
        m = _STACK_RE.search(line)
        if m:
            stack[int(line.split()[0])] = [int(x) for x in m.group(0).strip()[1:].split(">")]
            line = line[:m.start()]
        lines.append(line)
    return from_text("\n".join(lines)), stack


def parse_step2(text: str) -> tuple[list[dict], dict[tuple[int, int], str]]:
    """Edges, and the variable named for each (src, dst)."""
    edges, names = [], {}
    for raw in _section(text, "EDGES").splitlines():
        line = raw.strip()
        if not line or line == "EDGES":
            continue
        dst, sep, rest = line.partition("<-")
        if not sep or not dst.strip().isdigit():
            raise ParseError(f"cannot parse {raw!r}")
        for part in rest.split(";"):
            m = _PART_RE.match(part.strip())
            if not m:
                raise ParseError(f"cannot parse {part!r} in {raw!r}")
            src = int(m.group(2))
            edges.append({"src": src, "dst": int(dst), "args": [int(a) for a in re.findall(r"arg(\d+)", m.group(3))]})
            if m.group(1):
                names[(src, int(dst))] = m.group(1)
    edges.sort(key=lambda e: (e["dst"], e["src"]))
    return edges, names


def step2_input(step1: str) -> str:
    """Step 1's nodes as step 2 is given them: id, op, target and stack only.

    Read loosely, line by line, because at inference this is the model's own step-1
    reply: a line that is not a node is skipped here rather than failing step 2.
    """
    out = []
    for line in _section(step1, "NODES", "EDGES").splitlines():
        m = _BRIEF_RE.match(line.strip())
        if m:
            s = _STACK_RE.search(line)
            out.append(" ".join(m.groups()) + (f"  {s.group(0).strip()}" if s else ""))
    return "\n".join(out)


def assemble(step1: str, step2: str) -> str:
    """The two replies as one ordinary answer; raises ParseError if either is unreadable."""
    graph, _ = parse_step1(step1)
    graph["edges"], _ = parse_step2(step2)
    return to_text(graph)
