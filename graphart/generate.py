"""Synthetic nn.Module programs, grouped into difficulty tiers.

The tiers are about how much the *code* hides the forward pass, not how big
the network is:

  1 flat      - one layer per line; reading it is transcription
  2 loops     - Sequential / ModuleList / built in a loop, so it must be unrolled
  3 branches  - residuals, concat splits, dead branches, a reused layer
  4 blocks    - recognisable architectures: residual, inception, transformer, U-net

Tier 3 exists because real architectures are topologically thin -- measured
across torchvision, the edge-to-node ratio is about 1.1 almost everywhere.
Found code will not supply branching, so the generator has to manufacture it.

THE HOUSE-STYLE PROBLEM
-----------------------
A generator that always writes `self.conv1`, always imports the same way and
always names its class `Net` teaches a model its habits rather than the task.
The model then scores well on our test split, which shares those habits, and
collapses on real code -- and we would not find out.

So *what the network is* and *how the code is written* are separated. The tier
functions choose layers and connections; `Style` independently chooses the
import form, class name, attribute naming scheme, activation idiom, forward
variable convention, whether there is a docstring, and more. The same network
can come out looking like several different people wrote it.
"""

import random

# --------------------------------------------------------------------------
# style pools -- every one of these is a dimension a model could otherwise
# learn instead of the task
# --------------------------------------------------------------------------
HEADERS = [
    ("import torch\nimport torch.nn as nn\nimport torch.nn.functional as F", "nn", "F"),
    ("import torch\nfrom torch import nn\nfrom torch.nn import functional as F", "nn", "F"),
    ("import torch\nimport torch.nn as nn\nimport torch.nn.functional as fn", "nn", "fn"),
    ("import torch\nfrom torch import nn\nimport torch.nn.functional as func", "nn", "func"),
]

CLASS_NAMES = ["Net", "Model", "MyNet", "Classifier", "FeatureNet", "SmallNet", "Backbone",
               "TinyModel", "Encoder", "Head", "Branchy", "BlockNet", "DemoNet", "Module1"]

MODEL_VARS = ["model", "net", "m", "module", "mdl"]

# (functional spelling, module class). The functional spellings normalise to
# the same label in extract.ALIASES, so varying them is free diversity.
ACTIVATIONS = [("relu", "ReLU"), ("gelu", "GELU"), ("tanh", "Tanh"),
               ("sigmoid", "Sigmoid"), ("silu", "SiLU")]
# Only these have a top-level torch.* spelling; gelu and silu live in
# functional only, so the `torch.` variant is not available for them.
TORCH_SPELLED = {"relu", "sigmoid", "tanh"}

SEMANTIC_NAMES = ["stem", "body", "head", "proj", "mix", "gate", "neck", "tail",
                  "encode", "decode", "shrink", "expand", "merge", "pre", "post"]

NAME_SCHEMES = ["typed", "typed_underscore", "short", "generic", "semantic"]
FLOW_STYLES = ["reuse", "chain", "numbered"]
BRANCH_NAMES = ["a", "b", "c", "left", "right", "p", "q", "u", "v"]


class Style:
    """Every naming and idiom decision for one sample."""

    def __init__(self, rng: random.Random):
        self.rng = rng
        self.imports, self.nn, self.F = rng.choice(HEADERS)
        self.cls = rng.choice(CLASS_NAMES)
        self.var = rng.choice(MODEL_VARS)
        self.long_super = rng.random() < 0.25
        self.act_fn, self.act_cls = rng.choice(ACTIVATIONS)
        self.act_as_module = rng.random() < 0.45
        self.shared_act = self.act_as_module and rng.random() < 0.6
        self.torch_spelling = rng.random() < 0.4      # torch.relu vs F.relu
        self.scheme = rng.choice(NAME_SCHEMES)
        self.flow = rng.choice(FLOW_STYLES)
        self.docstring = rng.random() < 0.3
        self.comments = rng.random() < 0.25
        self.no_bias = rng.random() < 0.3
        self._kind_counts: dict[str, int] = {}
        self._used: set[str] = set()
        self._total = 0
        self._semantic = rng.sample(SEMANTIC_NAMES, k=len(SEMANTIC_NAMES))

    def attr(self, kind: str) -> str:
        """A fresh attribute name for a layer of this kind.

        Uniqueness is enforced here rather than trusted to the scheme. The
        short scheme abbreviates by first letter, so `flat` and `fc` both want
        `f1`; a collision would silently overwrite the first layer in __init__
        and leave the generated program quietly wrong.
        """
        self._total += 1
        n = self._kind_counts.get(kind, 0) + 1
        self._kind_counts[kind] = n
        if self.scheme == "semantic":
            base = self._semantic[(self._total - 1) % len(self._semantic)]
        elif self.scheme == "typed":
            base = f"{kind}{n}"
        elif self.scheme == "typed_underscore":
            base = f"{kind}_{n}"
        elif self.scheme == "short":
            base = f"{kind[0]}{n}"
        else:
            base = f"layer{self._total}"
        name, suffix = base, 1
        while name in self._used:
            suffix += 1
            name = f"{base}_{suffix}"
        self._used.add(name)
        return name

    def super_line(self) -> str:
        return (f"        super({self.cls}, self).__init__()" if self.long_super
                else "        super().__init__()")

    def bias_arg(self) -> str:
        return ", bias=False" if self.no_bias else ""


class Builder:
    """Accumulates __init__ and forward lines while tracking the live variable."""

    def __init__(self, rng: random.Random, style: Style):
        self.rng = rng
        self.style = style
        self.inits: list[str] = []
        self.body: list[str] = []
        self.var = "x"
        self._n = 0
        self._act_attr: str | None = None

    # -- naming ------------------------------------------------------------
    def _next_var(self) -> str:
        if self.style.flow == "reuse":
            return "x"
        if self.style.flow == "chain":
            return "h"
        self._n += 1
        return f"h{self._n}"

    # -- emitting ----------------------------------------------------------
    def declare(self, kind: str, ctor: str) -> str:
        name = self.style.attr(kind)
        self.inits.append(f"        self.{name} = {ctor}")
        return name

    def assign(self, rhs: str) -> str:
        out = self._next_var()
        self.body.append(f"        {out} = {rhs}")
        self.var = out
        return out

    def use(self, attr: str, src: str | None = None) -> str:
        return self.assign(f"self.{attr}({src or self.var})")

    def act(self, src: str | None = None) -> str:
        s, src = self.style, src or self.var
        if s.act_as_module:
            if s.shared_act:
                if self._act_attr is None:
                    self._act_attr = self.declare("act", f"{s.nn}.{s.act_cls}()")
                attr = self._act_attr
            else:
                attr = self.declare("act", f"{s.nn}.{s.act_cls}()")
            return self.use(attr, src)
        use_torch = s.torch_spelling and s.act_fn in TORCH_SPELLED
        call = f"torch.{s.act_fn}({src})" if use_torch else f"{s.F}.{s.act_fn}({src})"
        return self.assign(call)

    def comment(self, text: str) -> None:
        if self.style.comments:
            self.body.append(f"        # {text}")

    # -- rendering ---------------------------------------------------------
    def render(self, input_expr: str) -> str:
        s = self.style
        doc = f'        """{s.cls} built for the graph task."""\n' if s.docstring else ""
        init_block = "\n".join(self.inits) or "        pass"
        body_block = "\n".join(self.body)
        return (f"{s.imports}\n\n\n"
                f"class {s.cls}({s.nn}.Module):\n"
                f"    def __init__(self):\n"
                f"{doc}{s.super_line()}\n"
                f"{init_block}\n\n"
                f"    def forward(self, x):\n"
                f"{body_block}\n"
                f"        return {self.var}\n\n\n"
                f"{s.var} = {s.cls}()\n"
                f"example_input = {input_expr}\n")


# --------------------------------------------------------------------------
# shared pieces
# --------------------------------------------------------------------------
def _widths(rng: random.Random, n: int, lo: int = 8, hi: int = 64) -> list[int]:
    choices = [8, 12, 16, 24, 32, 48, 64]
    return [rng.choice([c for c in choices if lo <= c <= hi]) for _ in range(n)]


def _linear(b: Builder, fan_in: int, fan_out: int) -> str:
    s = b.style
    return b.declare("fc", f"{s.nn}.Linear({fan_in}, {fan_out}{s.bias_arg()})")


def _conv(b: Builder, cin: int, cout: int, k: int = 3) -> str:
    s = b.style
    pad = k // 2
    return b.declare("conv", f"{s.nn}.Conv2d({cin}, {cout}, {k}, padding={pad}{s.bias_arg()})")


def _norm(b: Builder, channels: int, two_d: bool) -> str | None:
    if b.rng.random() > 0.55:
        return None
    s = b.style
    if two_d:
        return b.declare("bn", f"{s.nn}.BatchNorm2d({channels})")
    return b.declare("norm", f"{s.nn}.LayerNorm({channels})")


def _cnn_tail(b: Builder, channels: int, classes: int) -> None:
    """Pool to 1x1, flatten, classify -- keeps spatial arithmetic out of the way."""
    s = b.style
    pool = b.declare("pool", f"{s.nn}.AdaptiveAvgPool2d(1)")
    b.use(pool)
    flat = b.declare("flat", f"{s.nn}.Flatten()")
    b.use(flat)
    b.use(_linear(b, channels, classes))


# --------------------------------------------------------------------------
# tier 1 -- flat
# --------------------------------------------------------------------------
def tier1_flat(rng: random.Random, style: Style) -> str:
    b = Builder(rng, style)
    if rng.random() < 0.5:
        dims = _widths(rng, rng.randint(3, 5))
        classes = rng.choice([2, 4, 10])
        b.comment("plain stack of fully connected layers")
        prev = dims[0]
        for i, width in enumerate(dims[1:], 1):
            b.use(_linear(b, prev, width))
            b.act()
            prev = width
        b.use(_linear(b, prev, classes))
        return b.render(f"torch.randn({rng.choice([1, 2, 4])}, {dims[0]})")

    chans = [3] + _widths(rng, rng.randint(2, 3), 8, 48)
    classes = rng.choice([2, 10])
    size = rng.choice([16, 32])
    for cin, cout in zip(chans, chans[1:]):
        b.use(_conv(b, cin, cout, rng.choice([3, 3, 5])))
        n = _norm(b, cout, True)
        if n:
            b.use(n)
        b.act()
    _cnn_tail(b, chans[-1], classes)
    return b.render(f"torch.randn(1, 3, {size}, {size})")


# --------------------------------------------------------------------------
# tier 2 -- loops
# --------------------------------------------------------------------------
def tier2_loops(rng: random.Random, style: Style) -> str:
    b, s = Builder(rng, style), style
    width = rng.choice([16, 24, 32])
    depth = rng.randint(2, 4)
    classes = rng.choice([2, 4, 10])
    form = rng.choice(["sequential", "modulelist", "loop_build"])

    b.use(_linear(b, width, width))
    b.act()

    if form == "sequential":
        inner = f"{s.nn}.Linear({width}, {width}{s.bias_arg()}), {s.nn}.{s.act_cls}()"
        name = b.declare("block", f"{s.nn}.Sequential(\n            "
                                  f"*[{s.nn}.Sequential({inner}) for _ in range({depth})]\n        )")
        b.comment(f"{depth} identical blocks")
        b.use(name)
    elif form == "modulelist":
        name = b.declare("blocks", f"{s.nn}.ModuleList(\n            "
                                   f"[{s.nn}.Linear({width}, {width}{s.bias_arg()}) "
                                   f"for _ in range({depth})]\n        )")
        var = rng.choice(["layer", "blk", "mod"])
        b.body.append(f"        for {var} in self.{name}:")
        b.body.append(f"            {b.var} = {var}({b.var})")
        b.act()
    else:
        name = b.declare("stack", f"{s.nn}.Sequential()")
        b.inits.append(f"        for _ in range({depth}):")
        b.inits.append(f"            self.{name}.append({s.nn}.Linear("
                       f"{width}, {width}{s.bias_arg()}))")
        b.inits.append(f"            self.{name}.append({s.nn}.{s.act_cls}())")
        b.use(name)

    b.use(_linear(b, width, classes))
    return b.render(f"torch.randn({rng.choice([1, 2])}, {width})")


# --------------------------------------------------------------------------
# tier 3 -- branches, where the topology actually lives
# --------------------------------------------------------------------------
def tier3_branches(rng: random.Random, style: Style) -> str:
    b, s = Builder(rng, style), style
    width = rng.choice([16, 24, 32])
    classes = rng.choice([2, 4, 10])
    patterns = rng.sample(["residual", "concat", "dead", "reuse", "fanout"],
                          k=rng.randint(2, 3))

    b.use(_linear(b, width, width))
    b.act()

    for pattern in patterns:
        if pattern == "residual":
            b.comment("skip connection")
            keep = b.var
            b.use(_linear(b, width, width))
            b.act()
            b.assign(f"{b.var} + {keep}")

        elif pattern == "concat":
            src = b.var
            half = max(4, width // 2)
            left, right = rng.sample(BRANCH_NAMES, 2)
            la, ra = _linear(b, width, half), _linear(b, width, half)
            b.body.append(f"        {left} = self.{la}({src})")
            b.body.append(f"        {right} = self.{ra}({src})")
            b.assign(f"torch.cat([{left}, {right}], dim=1)")
            if half * 2 != width:
                b.use(_linear(b, half * 2, width))

        elif pattern == "dead":
            flag = rng.choice(["use_extra", "deep", "wide", "extra_head"])
            b.inits.append(f"        self.{flag} = False")
            extra = _linear(b, width, width)
            b.body.append(f"        if self.{flag}:")
            b.body.append(f"            {b.var} = self.{extra}({b.var})")

        elif pattern == "reuse":
            b.comment("the same layer applied twice")
            shared = _linear(b, width, width)
            b.use(shared)
            b.act()
            b.use(shared)

        elif pattern == "fanout":
            src = b.var
            names = rng.sample(BRANCH_NAMES, 3)
            attrs = [_linear(b, width, width) for _ in range(3)]
            for n, a in zip(names, attrs):
                b.body.append(f"        {n} = self.{a}({src})")
            b.assign(f"{names[0]} + {names[1]}")
            b.assign(f"{b.var} + {names[2]}")

    b.use(_linear(b, width, classes))
    return b.render(f"torch.randn({rng.choice([1, 2])}, {width})")


# --------------------------------------------------------------------------
# tier 4 -- recognisable architectures
# --------------------------------------------------------------------------
def tier4_blocks(rng: random.Random, style: Style) -> str:
    kind = rng.choice(["residual_cnn", "inception", "transformer", "unet"])
    b, s = Builder(rng, style), style

    if kind == "residual_cnn":
        ch = rng.choice([8, 16, 32])
        size = rng.choice([16, 32])
        blocks = rng.randint(2, 3)
        b.use(_conv(b, 3, ch))
        b.act()
        for _ in range(blocks):
            b.comment("residual block")
            keep = b.var
            b.use(_conv(b, ch, ch))
            n1 = _norm(b, ch, True)
            if n1:
                b.use(n1)
            b.act()
            b.use(_conv(b, ch, ch))
            b.assign(f"{b.var} + {keep}")
            b.act()
        _cnn_tail(b, ch, rng.choice([2, 10]))
        return b.render(f"torch.randn(1, 3, {size}, {size})")

    if kind == "inception":
        ch = rng.choice([8, 16])
        size = rng.choice([16, 32])
        b.use(_conv(b, 3, ch))
        b.act()
        src = b.var
        names = rng.sample(BRANCH_NAMES, 3)
        b.comment("three parallel branches")
        for name, k in zip(names, [1, 3, 5]):
            a = _conv(b, ch, ch, k)
            b.body.append(f"        {name} = self.{a}({src})")
        b.assign(f"torch.cat([{names[0]}, {names[1]}, {names[2]}], dim=1)")
        b.act()
        _cnn_tail(b, ch * 3, rng.choice([2, 10]))
        return b.render(f"torch.randn(1, 3, {size}, {size})")

    if kind == "transformer":
        dim = rng.choice([16, 32])
        tokens = rng.choice([4, 8])
        hidden = dim * rng.choice([2, 4])
        b.comment("attention sublayer")
        keep = b.var
        n1 = b.declare("norm", f"{s.nn}.LayerNorm({dim})")
        b.use(n1)
        attn = b.declare("attn", f"{s.nn}.MultiheadAttention({dim}, "
                                 f"{rng.choice([1, 2, 4])}, batch_first=True)")
        out, side = rng.sample(BRANCH_NAMES, 2)
        b.body.append(f"        {out}, {side} = self.{attn}({b.var}, {b.var}, {b.var})")
        b.var = out
        b.assign(f"{b.var} + {keep}")
        b.comment("feed-forward sublayer")
        keep2 = b.var
        n2 = b.declare("norm", f"{s.nn}.LayerNorm({dim})")
        b.use(n2)
        b.use(_linear(b, dim, hidden))
        b.act()
        b.use(_linear(b, hidden, dim))
        b.assign(f"{b.var} + {keep2}")
        return b.render(f"torch.randn(1, {tokens}, {dim})")

    ch = rng.choice([8, 16])
    size = rng.choice([16, 32])
    b.comment("down path")
    b.use(_conv(b, 3, ch))
    b.act()
    skip = b.var
    down = b.declare("down", f"{s.nn}.MaxPool2d(2)")
    b.use(down)
    b.use(_conv(b, ch, ch * 2))
    b.act()
    b.comment("up path, with the skip concatenated back in")
    up = b.declare("up", f"{s.nn}.Upsample(scale_factor=2)")
    b.use(up)
    b.use(_conv(b, ch * 2, ch))
    b.assign(f"torch.cat([{b.var}, {skip}], dim=1)")
    b.use(_conv(b, ch * 2, ch))
    b.act()
    _cnn_tail(b, ch, rng.choice([2, 10]))
    return b.render(f"torch.randn(1, 3, {size}, {size})")


TIERS = {1: tier1_flat, 2: tier2_loops, 3: tier3_branches, 4: tier4_blocks}


def generate_program(rng: random.Random, tier: int) -> str:
    """One random nn.Module program for the given tier."""
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier}; choose from {sorted(TIERS)}")
    return TIERS[tier](rng, Style(rng))
