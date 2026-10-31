"""Library-style nn.Module programs: larger graphs, written the way torchvision is (tiers 5-6).

    from graphart.generate_large import generate_large_program
    code = generate_large_program(random.Random(0), tier=6)

  5 block    - one building block, as a library defines it: ResNet basic/bottleneck,
               Inception, inverted residual (MobileNet/MNASNet), Fire, dense block,
               ShuffleNet unit, transformer encoder layer. ~10-45 nodes.
  6 network  - a whole small network assembled from those blocks: stem, stages built
               by a _make_layer method or a config table, head. ~40-150 nodes.

WHY A SECOND GENERATOR
----------------------
The closing experiments showed the model fine-tuned on generate.py's programs learned
that generator's shortcuts, not the rule: every program was one class, the input was
always `x`, every `if self.flag:` was False, and helper classes, helper functions,
grouped convolutions, non-square kernels and un-shortened `F.` functions never appeared.
On real torchvision code it drew most graphs as a straight chain.

This generator writes what torchvision writes: several classes and helper functions per
program, constructor arguments, nested nn.Sequential stages (paths like
`stage2.1.conv.0`), `Optional` sub-modules checked with `is not None`, flags that are
True as often as False, depthwise and grouped convolutions, kernels like (1, 7),
`F.avg_pool2d`, `torch.flatten`, tensor methods, tuple unpacking.

KEEPING TORCHVISION A REAL TEST
-------------------------------
The 21 torchvision samples are the out-of-distribution test (RQ4). Programs here are
*like* them but are not copies: class names, attribute names, variable names, channel
widths, depths and branch layouts are drawn at random, and the dataset builder rejects
any program whose graph is isomorphic to a torchvision sample.

As in generate.py, what the network is and how its code is written are separate
decisions: `LStyle` picks the idioms, the family functions pick the structure.

SECOND ROUND (after Session 1)
------------------------------
The adapter trained on this set got 7/21 torchvision graphs right. Its misses on those
21 pointed at idioms still missing here, now added: the conv unit as torchvision's
BasicConv2d (`**kwargs` passed to nn.Conv2d, ending in `F.relu(x, inplace=True)`) or as
an nn.Sequential subclass; building blocks passed in (`conv_block=None`,
`norm_layer=None`, then a default); helper methods that return a layer
(`@staticmethod depthwise_conv`); Inception branches that fork into a (1,3) and a (3,1)
conv and re-join; `assert` and `torch._assert`. The held-out torchvision set
(data/generated/torchvision_holdout.jsonl) was frozen before this round and played no
part in it.
"""

import random

# --------------------------------------------------------------------------
# style
# --------------------------------------------------------------------------
HEADERS_PLAIN = [
    "import torch\nimport torch.nn as nn\nimport torch.nn.functional as F",
    "import torch\nfrom torch import nn\nfrom torch.nn import functional as F",
]
HEADERS_TYPED = [
    "from typing import Callable, List, Optional\n\nimport torch\nimport torch.nn as nn\n"
    "import torch.nn.functional as F\nfrom torch import Tensor",
    "from typing import Any, Callable, List, Optional\n\nimport torch\nfrom torch import Tensor, nn\n"
    "import torch.nn.functional as F",
]
INPUT_NAMES = ["x", "x", "x", "input", "inp", "feats", "tensor", "z"]
INSTANCE_NAMES = ["model", "net", "block", "m", "network", "module"]
# (module class, functional name, has a torch.* spelling)
CNN_ACTS = [("ReLU", "relu", True), ("ReLU", "relu", True), ("ReLU6", "relu6", False),
            ("SiLU", "silu", False), ("Hardswish", "hardswish", False)]


class LStyle:
    """Every idiom decision for one program."""

    def __init__(self, rng: random.Random):
        self.rng = rng
        self.typed = rng.random() < 0.5
        self.imports = rng.choice(HEADERS_TYPED if self.typed else HEADERS_PLAIN)
        self.arg = rng.choice(INPUT_NAMES)
        self.instance = rng.choice(INSTANCE_NAMES)
        self.act_cls, self.act_fn, self.act_torch = rng.choice(CNN_ACTS)
        self.inplace = rng.random() < 0.6
        self.act_idiom = rng.choice(["shared", "separate", "functional"])
        self.long_super = rng.random() < 0.25
        # The conv-norm-act helper:
        #   class       explicit arguments, forward() ends in an activation module or F.<act>
        #   kwargs      torchvision's BasicConv2d: (in, out, **kwargs) passed on to nn.Conv2d,
        #               so callers give padding themselves
        #   sequential  a class that subclasses nn.Sequential (paths like conv.1.0)
        #   function    a function returning nn.Sequential
        self.helper_kind = rng.choice(["class", "kwargs", "sequential", "function"])
        self.helper = rng.choice({
            "class": ["BasicConv2d", "ConvBNReLU", "ConvNormAct", "ConvBlock", "ConvUnit"],
            "kwargs": ["BasicConv2d", "ConvBN", "ConvLayer", "BasicConv"],
            "sequential": ["ConvBNActivation", "ConvNormActivation", "ConvBNAct", "Conv2dBNAct"],
            # never `conv_block`: that is the injected argument's name
            "function": ["conv_bn", "conv_bn_act", "conv_bn_relu", "make_conv"]}[self.helper_kind])
        # how a helper class applies its activation: a module (always, or behind a flag),
        # or F.<act>(x, inplace=True) as torchvision's BasicConv2d does
        self.helper_act_flag = rng.choice(["identity", "flag", "functional"])
        self.inplace_fn = rng.random() < 0.5          # F.relu(x, inplace=True)
        # torchvision passes building blocks in: conv_block=None / norm_layer=None, then
        # `if conv_block is None: conv_block = BasicConv2d`
        self.inject = rng.random() < 0.4

    def act_ctor(self) -> str:
        return f"nn.{self.act_cls}(inplace=True)" if self.inplace and self.act_cls != "GELU" \
            else f"nn.{self.act_cls}()"

    def act_fn_call(self, src: str) -> str:
        """F.<act>(src), with inplace=True when the style says so."""
        return f"F.{self.act_fn}({src}, inplace=True)" if self.inplace_fn else f"F.{self.act_fn}({src})"

    def act_call(self, attr: str, src: str) -> str:
        """Apply the activation: a module attribute, or a function call."""
        if self.act_idiom == "functional":
            if self.act_torch and self.rng.random() < 0.4:
                return f"torch.{self.act_fn}({src})"
            return self.act_fn_call(src)
        return f"self.{attr}({src})"

    def callable_arg(self, name: str) -> str:
        """`name: Optional[Callable[..., nn.Module]] = None`, or `name=None` untyped."""
        return f"{name}: Optional[Callable[..., nn.Module]] = None" if self.typed else f"{name}=None"

    def sig(self, args: str, ret: bool = True) -> str:
        """`def forward(self, x: Tensor) -> Tensor` or the untyped form."""
        if not self.typed:
            return f"({args})"
        typed = ", ".join(a if a == "self" or ":" in a or "=" in a else f"{a}: Tensor"
                          for a in (s.strip() for s in args.split(",")))
        return f"({typed}) -> Tensor" if ret else f"({typed})"

    def super_line(self, cls: str) -> str:
        return f"super({cls}, self).__init__()" if self.long_super else "super().__init__()"


class Program:
    """Collects top-level definitions; renders imports + defs + instance + input."""

    def __init__(self, style: LStyle):
        self.style = style
        self.defs: list[str] = []
        self.have: set[str] = set()

    def add(self, name: str, text: str) -> None:
        if name not in self.have:
            self.have.add(name)
            self.defs.append(text.strip("\n"))

    def render(self, ctor: str, input_expr: str) -> str:
        s = self.style
        return (f"{s.imports}\n\n\n" + "\n\n\n".join(self.defs) +
                f"\n\n\n{s.instance} = {ctor}\nexample_input = {input_expr}\n")


def _ind(lines: list[str], n: int) -> str:
    return "\n".join((" " * n + l) if l else "" for l in lines)


# --------------------------------------------------------------------------
# shared helpers: the conv-norm-act unit, conv3x3/conv1x1
# --------------------------------------------------------------------------
def hcall(s: LStyle, H: str, cin, cout, k, stride=None, groups=None) -> str:
    """A call to the conv-norm-act helper (or whatever stands in for it, e.g. an injected
    `conv_block`). The kwargs helper takes no padding of its own, so it is passed here."""
    args = [str(cin), str(cout), f"kernel_size={k!r}"]
    if stride is not None:
        args.append(f"stride={stride}")
    if s.helper_kind == "kwargs":
        pad = k // 2 if isinstance(k, int) else tuple(x // 2 for x in k)
        if pad:
            args.append(f"padding={pad!r}")
    if groups is not None:
        args.append(f"groups={groups}")
    return f"{H}({', '.join(args)})"


def helper_unit(p: Program) -> str:
    """Define the conv + BatchNorm + activation helper once; return its name.

    Call it through hcall(): H(in, out, kernel_size=..., stride=..., groups=...)
    """
    s = p.style
    name = s.helper
    a = s.arg
    pad = ("(kernel_size - 1) // 2 if isinstance(kernel_size, int) "
           "else tuple((k - 1) // 2 for k in kernel_size)")
    if s.helper_kind == "kwargs":
        eps = s.rng.choice(["", ", eps=0.001"])
        if s.helper_act_flag == "functional" or s.rng.random() < 0.5:
            act_init, tail = "", f"return {s.act_fn_call(a)}"
        else:
            act_init, tail = f"\n        self.act = {s.act_ctor()}", f"return self.act({a})"
        kw = "**kwargs: Any" if s.typed and "Any" in s.imports else "**kwargs"
        p.add(name, f'''
class {name}(nn.Module):
    def __init__(self, in_channels, out_channels, {kw}):
        {s.super_line(name)}
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels{eps}){act_init}

    def forward{s.sig(f"self, {a}")}:
        {a} = self.conv({a})
        {a} = self.bn({a})
        {tail}
''')
        return name
    if s.helper_kind == "sequential":
        sup = f"super({name}, self).__init__(*layers)" if s.long_super else "super().__init__(*layers)"
        p.add(name, f'''
class {name}(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1, act=True):
        padding = {pad}
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
        ]
        if act:
            layers.append({s.act_ctor()})
        {sup}
''')
        return name
    if s.helper_kind == "function":
        p.add(name, f'''
def {name}(in_channels, out_channels, kernel_size=3, stride=1, groups=1, act=True):
    padding = {pad}
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                  groups=groups, bias=False),
        nn.BatchNorm2d(out_channels),
    ]
    if act:
        layers.append({s.act_ctor()})
    return nn.Sequential(*layers)
''')
        return name
    if s.helper_act_flag == "identity":
        act_init = f"        self.act = {s.act_ctor()} if act else nn.Identity()"
        body = [f"{a} = self.conv({a})", f"{a} = self.bn({a})", f"return self.act({a})"]
    elif s.helper_act_flag == "functional":
        act_init = "        self.use_act = act"
        body = [f"{a} = self.conv({a})", f"{a} = self.bn({a})", "if not self.use_act:", f"    return {a}",
                f"return {s.act_fn_call(a)}"]
    else:
        act_init = f"        self.use_act = act\n        self.act = {s.act_ctor()}"
        body = [f"{a} = self.conv({a})", f"{a} = self.bn({a})", "if self.use_act:",
                f"    {a} = self.act({a})", f"return {a}"]
    p.add(name, f'''
class {name}(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1, act=True):
        {s.super_line(name)}
        padding = {pad}
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              padding=padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
{act_init}

    def forward{s.sig(f"self, {a}")}:
{_ind(body, 8)}
''')
    return name


def conv_helpers(p: Program, rng: random.Random) -> tuple[str, str]:
    """conv3x3 / conv1x1 helper functions (names varied); return their names."""
    k3, k1 = rng.choice([("conv3x3", "conv1x1"), ("conv3", "conv1"), ("make_conv3", "make_conv1")])
    p.add(k3, f'''
def {k3}(in_planes, out_planes, stride=1, groups=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1,
                     groups=groups, bias=False)
''')
    p.add(k1, f'''
def {k1}(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
''')
    return k3, k1


def _act_attrs(s: LStyle, n: int) -> list[str]:
    """Attribute names for n activations: one shared module, or one each."""
    return ["relu"] * n if s.act_idiom == "shared" else [f"relu{i + 1}" for i in range(n)]


def _act_inits(s: LStyle, names: list[str]) -> list[str]:
    if s.act_idiom == "functional":
        return []
    return [f"self.{n} = {s.act_ctor()}" for n in dict.fromkeys(names)]


# --------------------------------------------------------------------------
# family: ResNet basic / bottleneck block
# --------------------------------------------------------------------------
def resnet_block_def(p: Program, rng: random.Random, bottleneck: bool) -> tuple[str, int]:
    """Define a residual block class; return (class name, expansion)."""
    s = p.style
    k3, k1 = conv_helpers(p, rng)
    cls = rng.choice(["Bottleneck", "BottleneckBlock", "ResBottleneck"] if bottleneck
                     else ["BasicBlock", "ResBlock", "ResidualBlock", "SkipBlock", "ResUnit"])
    conv = rng.choice([("conv1", "conv2", "conv3"), ("conv_a", "conv_b", "conv_c"), ("c1", "c2", "c3")])
    bn = rng.choice([("bn1", "bn2", "bn3"), ("norm1", "norm2", "norm3"), ("bn_a", "bn_b", "bn_c")])
    down = rng.choice(["downsample", "shortcut", "proj", "skip"])
    ident = rng.choice(["identity", "residual", "shortcut_out", "skip_val"])
    out = rng.choice(["out", "y", "h"])
    a = s.arg
    n_conv = 3 if bottleneck else 2
    acts = _act_attrs(s, n_conv)
    exp = 4 if bottleneck else 1
    width = "planes"
    inits = []
    norm = "nn.BatchNorm2d"
    if s.inject:          # torchvision: `norm_layer=None`, then default to BatchNorm2d
        norm = "norm_layer"
        inits += ["if norm_layer is None:", "    norm_layer = nn.BatchNorm2d"]
    if bottleneck:
        inits += [f"self.{conv[0]} = {k1}(inplanes, {width})", f"self.{bn[0]} = {norm}({width})",
                  f"self.{conv[1]} = {k3}({width}, {width}, stride)", f"self.{bn[1]} = {norm}({width})",
                  f"self.{conv[2]} = {k1}({width}, planes * self.expansion)",
                  f"self.{bn[2]} = {norm}(planes * self.expansion)"]
    else:
        inits += [f"self.{conv[0]} = {k3}(inplanes, planes, stride)", f"self.{bn[0]} = {norm}(planes)",
                  f"self.{conv[1]} = {k3}(planes, planes)", f"self.{bn[1]} = {norm}(planes)"]
    inits += _act_inits(s, acts) + [f"self.{down} = {down}", "self.stride = stride"]
    body = [f"{ident} = {a}", ""]
    src = a
    for i in range(n_conv):
        body += [f"{out} = self.{conv[i]}({src})", f"{out} = self.{bn[i]}({out})"]
        if i < n_conv - 1:
            body.append(f"{out} = {s.act_call(acts[i], out)}")
        src = out
    body += ["", f"if self.{down} is not None:", f"    {ident} = self.{down}({a})", ""]
    body.append(rng.choice([f"{out} += {ident}", f"{out} = {out} + {ident}"]))
    body += [f"{out} = {s.act_call(acts[-1], out)}", f"return {out}"]
    down_arg = f"{down}: Optional[nn.Module] = None" if s.typed else f"{down}=None"
    if s.inject:
        down_arg += ", " + s.callable_arg("norm_layer")
    p.add(cls, f'''
class {cls}(nn.Module):
    expansion = {exp}

    def __init__(self, inplanes, planes, stride=1, {down_arg}):
        {s.super_line(cls)}
{_ind(inits, 8)}

    def forward{s.sig(f"self, {a}")}:
{_ind(body, 8)}
''')
    return cls, exp


def resnet_block(rng, s, p):
    bottleneck = rng.random() < 0.4
    cls, exp = resnet_block_def(p, rng, bottleneck)
    k1 = [n for n in p.have if n in ("conv1x1", "conv1", "make_conv1")][0]
    planes = rng.choice([8, 16, 32])
    stride = rng.choice([1, 1, 2])
    inplanes = rng.choice([planes * exp, planes * exp, planes * 2]) if bottleneck else rng.choice([planes, planes, planes // 2 or 8])
    size = rng.choice([16, 28, 32])
    needs_down = stride != 1 or inplanes != planes * exp
    if needs_down or rng.random() < 0.2:
        down = (f"nn.Sequential({k1}({inplanes}, {planes * exp}, {stride}), "
                f"nn.BatchNorm2d({planes * exp}))")
        # positional: the block's parameter is not always called `downsample`
        ctor = f"{cls}({inplanes}, {planes}, {stride}, {down})"
    else:
        ctor = f"{cls}({inplanes}, {planes})"
    return ctor, f"torch.randn(1, {inplanes}, {size}, {size})"


def resnet_network(rng, s, p):
    bottleneck = rng.random() < 0.35
    blk, exp = resnet_block_def(p, rng, bottleneck)
    k1 = [n for n in p.have if n in ("conv1x1", "conv1", "make_conv1")][0]
    cls = rng.choice(["ResNetLike", "TinyResNet", "ResidualNet", "Backbone", "ResNetSmall"])
    base = rng.choice([8, 16, 24])
    n_stages = rng.randint(2, 4)
    layers = [rng.randint(1, 3 if not bottleneck else 2) for _ in range(n_stages)]
    widths = [base * (2 ** i) for i in range(n_stages)]
    stage = rng.choice(["layer", "stage", "block", "res"])
    a = s.arg
    stem_k, stem_s = rng.choice([(7, 2), (3, 1), (5, 2)])
    pool = rng.random() < 0.7
    acts = _act_attrs(s, 1)
    inits = [f"self.inplanes = {base}",
             f"self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size={stem_k}, stride={stem_s}, "
             f"padding={stem_k // 2}, bias=False)",
             "self.bn1 = nn.BatchNorm2d(self.inplanes)"] + _act_inits(s, acts)
    if pool:
        inits.append("self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)")
    for i, (w, n) in enumerate(zip(widths, layers)):
        stride = "" if i == 0 else ", stride=2"
        inits.append(f"self.{stage}{i + 1} = self._make_layer(block, {w}, {n}{stride})")
    inits += ["self.avgpool = nn.AdaptiveAvgPool2d((1, 1))",
              f"self.fc = nn.Linear({widths[-1]} * block.expansion, num_classes)"]
    fwd = [f"{a} = self.conv1({a})", f"{a} = self.bn1({a})", f"{a} = {s.act_call(acts[0], a)}"]
    if pool:
        fwd.append(f"{a} = self.maxpool({a})")
    fwd += [f"{a} = self.{stage}{i + 1}({a})" for i in range(n_stages)]
    flat = rng.choice([f"torch.flatten({a}, 1)", f"{a}.flatten(1)", f"{a}.view({a}.size(0), -1)"])
    fwd += [f"{a} = self.avgpool({a})", f"{a} = {flat}", f"{a} = self.fc({a})", f"return {a}"]
    p.add(cls, f'''
class {cls}(nn.Module):
    def __init__(self, block, num_classes={rng.choice([10, 100, 1000])}):
        {s.super_line(cls)}
{_ind(inits, 8)}

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                {k1}(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward{s.sig(f"self, {a}")}:
{_ind(fwd, 8)}
''')
    size = rng.choice([32, 64])
    return f"{cls}({blk})", f"torch.randn(1, 3, {size}, {size})"


# --------------------------------------------------------------------------
# family: Inception
# --------------------------------------------------------------------------
KERNELS = [1, 3, 5, (1, 3), (3, 1), (1, 7), (7, 1)]


def inception_block_def(p: Program, rng: random.Random) -> tuple[str, callable]:
    """Define an Inception-style block class; return (name, out_channels(in, args))."""
    s = p.style
    H = helper_unit(p)
    cls = rng.choice(["InceptionA", "InceptionBlock", "MixedBlock", "Inception", "MultiBranch"])
    a = s.arg
    n_br = rng.randint(3, 4)
    pool_branch = rng.random() < 0.75
    branches = []                       # list of lists of (kernel, width-multiplier)
    for i in range(n_br - (1 if pool_branch else 0)):
        depth = 1 if i == 0 else rng.randint(2, 3)
        ks = [1] + [rng.choice(KERNELS[1:]) for _ in range(depth - 1)]
        if depth >= 3 and rng.random() < 0.6:
            pair = rng.choice([((1, 7), (7, 1)), ((1, 3), (3, 1)), ((7, 1), (1, 7))])
            ks = [1, *pair]
        branches.append(ks)
    names = [f"branch{i + 1}" for i in range(len(branches))]
    if rng.random() < 0.5:
        names = [f"branch_{'abcd'[i]}" for i in range(len(branches))]
    # torchvision passes the conv unit in (`conv_block=None`, defaulting to BasicConv2d)
    CB = "conv_block" if s.inject else H
    inits, fwd, outs = [], [], []
    if s.inject:
        inits += ["if conv_block is None:", f"    conv_block = {H}"]
    for bi, (bname, ks) in enumerate(zip(names, branches)):
        prev = "in_channels"
        var = rng.choice(["b", "y", "br"]) + str(bi + 1)
        src = a
        # a branch can fork at its end into a (1,3) and a (3,1) conv, concatenated
        fork = len(ks) >= 2 and rng.random() < 0.3
        for li, k in enumerate(ks):
            attr = bname if len(ks) == 1 else f"{bname}_{li + 1}"
            mult = rng.choice([1, 2]) if li else 1
            w = f"ch{bi + 1}" + (f" * {mult}" if mult > 1 else "")
            if fork and li == len(ks) - 1:
                inits += [f"self.{attr}a = {hcall(s, CB, prev, w, (1, 3))}",
                          f"self.{attr}b = {hcall(s, CB, prev, w, (3, 1))}"]
                fwd += [f"{var} = [", f"    self.{attr}a({src}),", f"    self.{attr}b({src}),", "]",
                        f"{var} = torch.cat({var}, 1)"]
                src, prev = var, f"({w}) * 2"
                continue
            inits.append(f"self.{attr} = {hcall(s, CB, prev, w, k)}")
            fwd.append(f"{var} = self.{attr}({src})")
            src, prev = var, w
        outs.append((var, prev))
    pool_widths = []
    if pool_branch:
        var = "bp"
        how = rng.choice(["F.avg_pool2d", "F.max_pool2d", "module"])
        if how == "module":
            inits.append("self.pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)")
            fwd.append(f"{var} = self.pool({a})")
        else:
            fwd.append(f"{var} = {how}({a}, kernel_size=3, stride=1, padding=1)")
        inits.append(f"self.branch_pool = {hcall(s, CB, 'in_channels', 'pool_features', 1)}")
        fwd.append(f"{var} = self.branch_pool({var})")
        outs.append((var, "pool_features"))
    params = ", ".join([f"ch{i + 1}" for i in range(len(branches))] + (["pool_features"] if pool_branch else [])
                       + ([s.callable_arg("conv_block")] if s.inject else []))
    if rng.random() < 0.5:
        defs = f'''
    def _forward{s.sig(f"self, {a}", ret=False)}:
{_ind(fwd, 8)}
        return [{", ".join(v for v, _ in outs)}]

    def forward{s.sig(f"self, {a}")}:
        outputs = self._forward({a})
        return torch.cat(outputs, 1)
'''
    else:
        defs = f'''
    def forward{s.sig(f"self, {a}")}:
{_ind(fwd, 8)}
        return torch.cat([{", ".join(v for v, _ in outs)}], dim=1)
'''
    p.add(cls, f'''
class {cls}(nn.Module):
    def __init__(self, in_channels, {params}):
        {s.super_line(cls)}
{_ind(inits, 8)}
{defs}''')

    def out_channels(widths: dict) -> int:
        total = 0
        for _, w in outs:
            expr = w
            for k, v in widths.items():
                expr = expr.replace(k, str(v))
            total += eval(expr)          # widths like "ch2 * 2", built above
        return total
    return cls, out_channels, len(branches), pool_branch


def _inception_args(rng, n_br, pool_branch):
    widths = {f"ch{i + 1}": rng.choice([8, 12, 16, 24]) for i in range(n_br)}
    if pool_branch:
        widths["pool_features"] = rng.choice([8, 16])
    return widths


def inception_block(rng, s, p):
    cls, out_ch, n_br, pool = inception_block_def(p, rng)
    cin = rng.choice([16, 24, 32, 48])
    w = _inception_args(rng, n_br, pool)
    size = rng.choice([8, 12, 17])
    return f"{cls}({cin}, {', '.join(map(str, w.values()))})", f"torch.randn(1, {cin}, {size}, {size})"


def inception_network(rng, s, p):
    cls_blk, out_ch, n_br, pool = inception_block_def(p, rng)
    H = s.helper
    cls = rng.choice(["InceptionNet", "MiniInception", "MultiBranchNet", "MixedNet"])
    a = s.arg
    stem = rng.choice([16, 24, 32])
    n_blocks = rng.randint(2, 4)
    inits = [f"self.stem1 = {hcall(s, H, 3, stem, 3, stride=2)}",
             f"self.stem2 = {hcall(s, H, stem, stem, 3)}"]
    fwd = [f"{a} = self.stem1({a})", f"{a} = self.stem2({a})"]
    if rng.random() < 0.6:
        fwd.append(f"{a} = F.max_pool2d({a}, kernel_size=3, stride=2)")
    cin = stem
    for i in range(n_blocks):
        w = _inception_args(rng, n_br, pool)
        inits.append(f"self.mixed_{i + 1} = {cls_blk}({cin}, {', '.join(map(str, w.values()))})")
        fwd.append(f"{a} = self.mixed_{i + 1}({a})")
        cin = out_ch(w)
        if i == n_blocks // 2 and rng.random() < 0.5:
            fwd.append(f"{a} = F.max_pool2d({a}, kernel_size=3, stride=2, padding=1)")
    inits += ["self.avgpool = nn.AdaptiveAvgPool2d((1, 1))", f"self.dropout = nn.Dropout(p={rng.choice([0.2, 0.5])})",
              f"self.fc = nn.Linear({cin}, num_classes)"]
    fwd += [f"{a} = self.avgpool({a})", f"{a} = torch.flatten({a}, 1)", f"{a} = self.dropout({a})",
            f"{a} = self.fc({a})", f"return {a}"]
    p.add(cls, f'''
class {cls}(nn.Module):
    def __init__(self, num_classes={rng.choice([10, 100])}):
        {s.super_line(cls)}
{_ind(inits, 8)}

    def forward{s.sig(f"self, {a}")}:
{_ind(fwd, 8)}
''')
    size = rng.choice([32, 48])
    return f"{cls}()", f"torch.randn(1, 3, {size}, {size})"


# --------------------------------------------------------------------------
# family: inverted residual (MobileNetV2 / MNASNet), optional squeeze-excitation
# --------------------------------------------------------------------------
def se_def(p: Program, rng: random.Random) -> str:
    s = p.style
    name = rng.choice(["SqueezeExcitation", "SEBlock", "SELayer"])
    a = s.arg
    gate = rng.choice(["nn.Sigmoid()", "nn.Hardsigmoid()"])
    p.add(name, f'''
class {name}(nn.Module):
    def __init__(self, channels, squeeze_channels):
        {s.super_line(name)}
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, squeeze_channels, 1)
        self.fc2 = nn.Conv2d(squeeze_channels, channels, 1)
        self.activation = nn.ReLU()
        self.scale_activation = {gate}

    def forward{s.sig(f"self, {a}")}:
        scale = self.avgpool({a})
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        scale = self.scale_activation(scale)
        return scale * {a}
''')
    return name


def inverted_residual_def(p: Program, rng: random.Random) -> str:
    s = p.style
    H = helper_unit(p)
    se = se_def(p, rng) if rng.random() < 0.5 else None
    cls = rng.choice(["InvertedResidual", "MBConv", "InvertedBottleneck", "MobileBlock"])
    a = s.arg
    flag = rng.choice(["use_res_connect", "use_residual", "has_skip", "residual"])
    se_line = []
    se_param = ""
    if se:
        se_param = ", use_se=False"
        se_line = ["if use_se:", f"    layers.append({se}(hidden_dim, max(1, hidden_dim // 4)))"]
    style_ret = rng.choice(["ifelse", "assign"])
    if style_ret == "ifelse":
        body = [f"if self.{flag}:", f"    return {a} + self.conv({a})", "return self.conv(" + a + ")"]
    else:
        body = [f"out = self.conv({a})", f"if self.{flag}:", f"    out = out + {a}", "return out"]
    # torchvision passes the norm layer in (`norm_layer=None`, defaulting to BatchNorm2d)
    norm, norm_param, norm_default = "nn.BatchNorm2d", "", []
    if s.inject:
        norm, norm_param = "norm_layer", ", " + s.callable_arg("norm_layer")
        norm_default = ["if norm_layer is None:", "    norm_layer = nn.BatchNorm2d", ""]
    k = rng.choice([3, 3, 5])
    p.add(cls, f'''
class {cls}(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio{se_param}{norm_param}):
        {s.super_line(cls)}
        self.stride = stride
{_ind(norm_default, 8)}
        hidden_dim = int(round(inp * expand_ratio))
        self.{flag} = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            layers.append({hcall(s, H, "inp", "hidden_dim", 1)})
        layers.append({hcall(s, H, "hidden_dim", "hidden_dim", k, stride="stride", groups="hidden_dim")})
{_ind(se_line, 8)}
        layers.append(nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False))
        layers.append({norm}(oup))
        self.conv = nn.Sequential(*layers)

    def forward{s.sig(f"self, {a}")}:
{_ind(body, 8)}
''')
    return cls, bool(se)


def inverted_residual_block(rng, s, p):
    cls, has_se = inverted_residual_def(p, rng)
    inp = rng.choice([16, 24, 32])
    oup = inp if rng.random() < 0.6 else rng.choice([16, 24, 32, 40])
    stride = 1 if oup == inp and rng.random() < 0.7 else rng.choice([1, 2])
    t = rng.choice([1, 3, 4, 6])
    se = f", use_se={rng.choice([True, False])}" if has_se else ""
    size = rng.choice([14, 16, 28])
    return f"{cls}({inp}, {oup}, {stride}, {t}{se})", f"torch.randn(1, {inp}, {size}, {size})"


def inverted_residual_network(rng, s, p):
    blk, has_se = inverted_residual_def(p, rng)
    H = s.helper
    cls = rng.choice(["MobileNetLike", "TinyMobileNet", "MBNet", "EfficientLike"])
    a = s.arg
    first = rng.choice([16, 24, 32])
    cfg = [(1, first // 2 or 8, 1, 1)]
    c = first // 2 or 8
    for _ in range(rng.randint(2, 4)):
        c = rng.choice([c, c + 8, c * 2]) if c < 64 else c
        cfg.append((rng.choice([3, 4, 6]), c, rng.randint(1, 3), rng.choice([1, 2])))
    last = rng.choice([64, 128])
    cfg_txt = ",\n".join(f"            [{t}, {cc}, {n}, {st}" + (f", {rng.choice([True, False])}]" if has_se else "]")
                         for t, cc, n, st in cfg)
    unpack = "t, c, n, s, se" if has_se else "t, c, n, s"
    call = f"{blk}(input_channel, c, stride, expand_ratio=t" + (", use_se=se)" if has_se else ")")
    fwd = [f"{a} = self.features({a})"]
    pool = rng.choice(["adaptive", "functional", "mean"])
    if pool == "adaptive":
        fwd += [f"{a} = nn.functional.adaptive_avg_pool2d({a}, (1, 1))", f"{a} = torch.flatten({a}, 1)"]
    elif pool == "functional":
        fwd += [f"{a} = F.adaptive_avg_pool2d({a}, 1)", f"{a} = torch.flatten({a}, 1)"]
    else:
        fwd += [f"{a} = {a}.mean([2, 3])"]
    fwd += [f"{a} = self.classifier({a})", f"return {a}"]
    p.add(cls, f'''
class {cls}(nn.Module):
    def __init__(self, num_classes={rng.choice([10, 100, 1000])}):
        {s.super_line(cls)}
        input_channel = {first}
        settings = [
{cfg_txt},
        ]
        features = [{hcall(s, H, 3, "input_channel", 3, stride=2)}]
        for {unpack} in settings:
            for i in range(n):
                stride = s if i == 0 else 1
                features.append({call})
                input_channel = c
        features.append({hcall(s, H, "input_channel", last, 1)})
        self.features = nn.Sequential(*features)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear({last}, num_classes),
        )

    def forward{s.sig(f"self, {a}")}:
{_ind(fwd, 8)}
''')
    size = rng.choice([32, 64])
    return f"{cls}()", f"torch.randn(1, 3, {size}, {size})"


# --------------------------------------------------------------------------
# family: SqueezeNet Fire
# --------------------------------------------------------------------------
def fire_def(p: Program, rng: random.Random) -> str:
    s = p.style
    cls = rng.choice(["Fire", "FireModule", "SqueezeExpand"])
    a = s.arg
    sq, e1, e3 = rng.choice([("squeeze", "expand1x1", "expand3x3"), ("sq", "ex1", "ex3"),
                             ("reduce", "expand_a", "expand_b")])
    names = ["squeeze_activation", "expand1x1_activation", "expand3x3_activation"] \
        if s.act_idiom != "shared" else ["act", "act", "act"]
    inits = [f"self.{sq} = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)",
             f"self.{e1} = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)",
             f"self.{e3} = nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)"]
    inits += _act_inits(s, names)
    fwd = [f"{a} = {s.act_call(names[0], f'self.{sq}({a})')}",
           "return torch.cat(",
           f"    [{s.act_call(names[1], f'self.{e1}({a})')}, {s.act_call(names[2], f'self.{e3}({a})')}], 1",
           ")"]
    p.add(cls, f'''
class {cls}(nn.Module):
    def __init__(self, inplanes, squeeze_planes, expand1x1_planes, expand3x3_planes):
        {s.super_line(cls)}
        self.inplanes = inplanes
{_ind(inits, 8)}

    def forward{s.sig(f"self, {a}")}:
{_ind(fwd, 8)}
''')
    return cls


def fire_block(rng, s, p):
    cls = fire_def(p, rng)
    cin = rng.choice([32, 64, 96, 128])
    sq = rng.choice([8, 16, 32])
    e = rng.choice([32, 64])
    size = rng.choice([13, 27, 55])
    return f"{cls}({cin}, {sq}, {e}, {e})", f"torch.randn(1, {cin}, {size}, {size})"


def fire_network(rng, s, p):
    fire = fire_def(p, rng)
    cls = rng.choice(["SqueezeNetLike", "TinySqueezeNet", "FireNet"])
    a = s.arg
    stem = rng.choice([32, 64])
    n_fire = rng.randint(3, 8)
    layers = [f"nn.Conv2d(3, {stem}, kernel_size=3, stride=2)", "nn.ReLU(inplace=True)",
              "nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)"]
    cin, sq = stem, rng.choice([8, 16])
    for i in range(n_fire):
        e = sq * 4
        layers.append(f"{fire}({cin}, {sq}, {e}, {e})")
        cin = 2 * e
        if i in (n_fire // 3, 2 * n_fire // 3) and i < n_fire - 1:
            layers.append("nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)")
        if i % 2 == 1:
            sq = min(sq * 2, 64)
    feat = ",\n".join(f"            {l}" for l in layers)
    p.add(cls, f'''
class {cls}(nn.Module):
    def __init__(self, num_classes={rng.choice([10, 100, 1000])}):
        {s.super_line(cls)}
        self.features = nn.Sequential(
{feat},
        )
        final_conv = nn.Conv2d({cin}, num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5), final_conv, nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward{s.sig(f"self, {a}")}:
        {a} = self.features({a})
        {a} = self.classifier({a})
        return torch.flatten({a}, 1)
''')
    size = rng.choice([64, 96, 128])
    return f"{cls}()", f"torch.randn(1, 3, {size}, {size})"


# --------------------------------------------------------------------------
# family: dense block (DenseNet)
# --------------------------------------------------------------------------
def dense_defs(p: Program, rng: random.Random) -> tuple[str, str]:
    s = p.style
    layer = rng.choice(["DenseLayer", "_DenseLayer", "DenseUnit"])
    block = rng.choice(["DenseBlock", "_DenseBlock", "DenseStage"])
    a = s.arg
    acts = _act_attrs(s, 2)
    inits = ["self.norm1 = nn.BatchNorm2d(num_input_features)"] + _act_inits(s, acts) + [
        "self.conv1 = nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, bias=False)",
        "self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)",
        "self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)"]
    fwd = [f"out = self.conv1({s.act_call(acts[0], f'self.norm1({a})')})",
           f"out = self.conv2({s.act_call(acts[1], 'self.norm2(out)')})", "return out"]
    p.add(layer, f'''
class {layer}(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size):
        {s.super_line(layer)}
{_ind(inits, 8)}

    def forward{s.sig(f"self, {a}")}:
{_ind(fwd, 8)}
''')
    feats = rng.choice(["features", "feats_list", "outputs"])
    p.add(block, f'''
class {block}(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate):
        {s.super_line(block)}
        self.layers = nn.ModuleList(
            [{layer}(num_input_features + i * growth_rate, growth_rate, bn_size) for i in range(num_layers)]
        )

    def forward{s.sig(f"self, {a}")}:
        {feats} = [{a}]
        for layer in self.layers:
            new = layer(torch.cat({feats}, 1))
            {feats}.append(new)
        return torch.cat({feats}, 1)
''')
    return layer, block


def dense_block(rng, s, p):
    _, block = dense_defs(p, rng)
    n, cin, g = rng.randint(2, 5), rng.choice([16, 24, 32]), rng.choice([8, 12, 16])
    size = rng.choice([8, 16])
    return f"{block}({n}, {cin}, {rng.choice([2, 4])}, {g})", f"torch.randn(1, {cin}, {size}, {size})"


def dense_network(rng, s, p):
    _, block = dense_defs(p, rng)
    cls = rng.choice(["DenseNetLike", "TinyDenseNet", "DenseBackbone"])
    a = s.arg
    g, bn = rng.choice([8, 12]), rng.choice([2, 4])
    stem = rng.choice([16, 24])
    blocks = [rng.randint(2, 4) for _ in range(rng.randint(2, 3))]
    inits = [f"self.conv0 = nn.Conv2d(3, {stem}, kernel_size=3, stride=2, padding=1, bias=False)",
             f"self.norm0 = nn.BatchNorm2d({stem})", "self.relu0 = nn.ReLU(inplace=True)"]
    fwd = [f"{a} = self.relu0(self.norm0(self.conv0({a})))"]
    c = stem
    for i, n in enumerate(blocks):
        inits.append(f"self.denseblock{i + 1} = {block}({n}, {c}, {bn}, {g})")
        fwd.append(f"{a} = self.denseblock{i + 1}({a})")
        c += n * g
        if i < len(blocks) - 1:
            inits.append(f"self.transition{i + 1} = nn.Sequential(nn.BatchNorm2d({c}), nn.ReLU(inplace=True), "
                         f"nn.Conv2d({c}, {c // 2}, kernel_size=1, bias=False), nn.AvgPool2d(kernel_size=2, stride=2))")
            fwd.append(f"{a} = self.transition{i + 1}({a})")
            c //= 2
    inits += [f"self.norm_final = nn.BatchNorm2d({c})", f"self.classifier = nn.Linear({c}, num_classes)"]
    fwd += [f"{a} = F.relu(self.norm_final({a}))", f"{a} = F.adaptive_avg_pool2d({a}, (1, 1))",
            f"{a} = torch.flatten({a}, 1)", f"return self.classifier({a})"]
    p.add(cls, f'''
class {cls}(nn.Module):
    def __init__(self, num_classes={rng.choice([10, 100])}):
        {s.super_line(cls)}
{_ind(inits, 8)}

    def forward{s.sig(f"self, {a}")}:
{_ind(fwd, 8)}
''')
    size = rng.choice([32, 64])
    return f"{cls}()", f"torch.randn(1, 3, {size}, {size})"


# --------------------------------------------------------------------------
# family: ShuffleNet unit
# --------------------------------------------------------------------------
def shuffle_defs(p: Program, rng: random.Random) -> str:
    s = p.style
    fn = rng.choice(["channel_shuffle", "shuffle_channels"])
    p.add(fn, f'''
def {fn}(x, groups):
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x
''')
    cls = rng.choice(["ShuffleUnit", "ShuffleBlock", "ShuffleResidual"])
    a = s.arg
    # the depthwise conv: written out, or built by a helper method that returns the layer
    # (torchvision's `@staticmethod def depthwise_conv(...) -> nn.Conv2d`)
    dw, dw_def = None, ""
    if rng.random() < 0.5:
        dw = rng.choice(["depthwise_conv", "dw_conv", "_depthwise"])
        ret = " -> nn.Conv2d" if s.typed else ""
        dw_def = f'''
    @staticmethod
    def {dw}(i, o, kernel_size, stride=1, padding=0, bias=False){ret}:
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)
'''
    dw1 = (f"self.{dw}(inp, inp, kernel_size=3, stride=self.stride, padding=1)" if dw else
           "nn.Conv2d(inp, inp, kernel_size=3, stride=self.stride, padding=1, groups=inp, bias=False)")
    dw2 = (f"self.{dw}(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1)" if dw else
           "nn.Conv2d(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1,\n"
           "                      groups=branch_features, bias=False)")
    p.add(cls, f'''
class {cls}(nn.Module):
    def __init__(self, inp, oup, stride):
        {s.super_line(cls)}
        self.stride = stride
        branch_features = oup // 2
        if self.stride > 1:
            self.branch1 = nn.Sequential(
                {dw1},
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch1 = nn.Sequential()
        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if self.stride > 1 else branch_features, branch_features, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            {dw2},
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )
{dw_def}
    def forward{s.sig(f"self, {a}")}:
        if self.stride == 1:
            x1, x2 = {a}.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1({a}), self.branch2({a})), dim=1)
        out = {fn}(out, 2)
        return out
''')
    return cls


def shuffle_block(rng, s, p):
    cls = shuffle_defs(p, rng)
    stride = rng.choice([1, 2])
    c = rng.choice([16, 24, 32, 48])
    oup = c if stride == 1 else rng.choice([c, c * 2])
    size = rng.choice([14, 28])
    return f"{cls}({c}, {oup}, {stride})", f"torch.randn(1, {c}, {size}, {size})"


def shuffle_network(rng, s, p):
    unit = shuffle_defs(p, rng)
    cls = rng.choice(["ShuffleNetLike", "TinyShuffleNet", "ShuffleBackbone"])
    a = s.arg
    stem = rng.choice([16, 24])
    stages = [rng.randint(1, 3) for _ in range(rng.randint(2, 3))]
    widths = [stem * 2 * (2 ** i) for i in range(len(stages))]
    lines = [f"self.conv1 = nn.Sequential(nn.Conv2d(3, {stem}, 3, 2, 1, bias=False), "
             f"nn.BatchNorm2d({stem}), nn.ReLU(inplace=True))",
             "self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)",
             f"input_channels = {stem}"]
    fwd = [f"{a} = self.conv1({a})", f"{a} = self.maxpool({a})"]
    for i, (n, w) in enumerate(zip(stages, widths)):
        lines.append(f"seq = [{unit}(input_channels, {w}, 2)]")
        lines.append(f"for _ in range({n - 1}):")
        lines.append(f"    seq.append({unit}({w}, {w}, 1))")
        lines.append(f"self.stage{i + 2} = nn.Sequential(*seq)")
        lines.append(f"input_channels = {w}")
        fwd.append(f"{a} = self.stage{i + 2}({a})")
    last = rng.choice([128, 256])
    lines += [f"self.conv5 = nn.Sequential(nn.Conv2d(input_channels, {last}, 1, 1, 0, bias=False), "
              f"nn.BatchNorm2d({last}), nn.ReLU(inplace=True))", f"self.fc = nn.Linear({last}, num_classes)"]
    fwd += [f"{a} = self.conv5({a})", f"{a} = {a}.mean([2, 3])", f"{a} = self.fc({a})", f"return {a}"]
    p.add(cls, f'''
class {cls}(nn.Module):
    def __init__(self, num_classes={rng.choice([10, 1000])}):
        {s.super_line(cls)}
{_ind(lines, 8)}

    def forward{s.sig(f"self, {a}")}:
{_ind(fwd, 8)}
''')
    size = rng.choice([64, 96])
    return f"{cls}()", f"torch.randn(1, 3, {size}, {size})"


# --------------------------------------------------------------------------
# family: transformer encoder (ViT)
# --------------------------------------------------------------------------
def encoder_defs(p: Program, rng: random.Random) -> str:
    s = p.style
    mlp = rng.choice(["MLPBlock", "FeedForward", "MLP"])
    enc = rng.choice(["EncoderBlock", "TransformerLayer", "EncoderLayer"])
    a = s.arg
    act = rng.choice(["nn.GELU()", "nn.GELU()", "nn.SiLU()"])
    p.add(mlp, f'''
class {mlp}(nn.Sequential):
    def __init__(self, in_dim, mlp_dim, dropout):
        super().__init__(
            nn.Linear(in_dim, mlp_dim),
            {act},
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, in_dim),
            nn.Dropout(dropout),
        )
''')
    # locals are h / y, never the input's own name, so the skip connection always
    # reads the untouched input
    if rng.random() < 0.75:          # pre-norm, as in ViT
        fwd = [f"h = self.ln_1({a})", "h, _ = self.self_attention(h, h, h, need_weights=False)",
               "h = self.dropout(h)", f"h = h + {a}", "", "y = self.ln_2(h)", "y = self.mlp(y)", "return h + y"]
    else:                            # post-norm, as in the original transformer
        fwd = [f"h, _ = self.self_attention({a}, {a}, {a}, need_weights=False)",
               f"h = self.ln_1(self.dropout(h) + {a})", "y = self.mlp(h)", "return self.ln_2(h + y)"]
    # A Python `assert` in __init__ adds nothing to the graph; torch._assert in forward
    # does (dim, eq, the shape it formats into its message, and the assert itself).
    init_assert = (["assert hidden_dim % num_heads == 0, \"hidden_dim must be divisible by num_heads\""]
                   if rng.random() < 0.4 else [])
    if rng.random() < 0.5:
        fwd = [f'torch._assert({a}.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {{{a}.shape}}")',
               *fwd]
    p.add(enc, f'''
class {enc}(nn.Module):
    def __init__(self, num_heads, hidden_dim, mlp_dim, dropout=0.0):
        {s.super_line(enc)}
{_ind(init_assert, 8)}
        self.num_heads = num_heads
        self.ln_1 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.ln_2 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.mlp = {mlp}(hidden_dim, mlp_dim, dropout)

    def forward{s.sig(f"self, {a}")}:
{_ind(fwd, 8)}
''')
    return enc


def encoder_block(rng, s, p):
    enc = encoder_defs(p, rng)
    dim = rng.choice([16, 32, 64])
    heads = rng.choice([1, 2, 4])
    tokens = rng.choice([4, 8, 16])
    return f"{enc}({heads}, {dim}, {dim * rng.choice([2, 4])})", f"torch.randn(1, {tokens}, {dim})"


def encoder_network(rng, s, p):
    enc = encoder_defs(p, rng)
    cls = rng.choice(["TinyViT", "VisionEncoder", "PatchTransformer", "ViTLike"])
    a = s.arg
    dim, heads = rng.choice([16, 32, 48]), rng.choice([2, 4])
    patch, size = rng.choice([(4, 16), (4, 32), (8, 32)])
    n_layers = rng.randint(2, 4)
    seq = (size // patch) ** 2
    cls_token = rng.random() < 0.5
    layers = ",\n".join(f"            {enc}({heads}, {dim}, {dim * 2})" for _ in range(n_layers))
    fwd = []
    if rng.random() < 0.4:           # ViT checks the image size before patchifying
        fwd += [f"n, c, h, w = {a}.shape",
                f'torch._assert(h == {size}, f"Wrong image height! Expected {size} but got {{h}}!")',
                f'torch._assert(w == {size}, f"Wrong image width! Expected {size} but got {{w}}!")']
    fwd += [f"x = self.conv_proj({a})", "x = x.flatten(2).transpose(1, 2)"]
    if cls_token:
        fwd += ["batch_class_token = self.class_token.expand(x.shape[0], -1, -1)",
                "x = torch.cat([batch_class_token, x], dim=1)"]
    fwd += ["x = x + self.pos_embedding", "x = self.encoder(x)", "x = self.ln(x)",
            "x = x[:, 0]" if cls_token else "x = x.mean(dim=1)", "return self.head(x)"]
    seq_len = seq + (1 if cls_token else 0)
    cls_init = f"\n        self.class_token = nn.Parameter(torch.zeros(1, 1, {dim}))" if cls_token else ""
    p.add(cls, f'''
class {cls}(nn.Module):
    def __init__(self, num_classes={rng.choice([10, 100])}):
        {s.super_line(cls)}
        self.conv_proj = nn.Conv2d(3, {dim}, kernel_size={patch}, stride={patch}){cls_init}
        self.pos_embedding = nn.Parameter(torch.zeros(1, {seq_len}, {dim}))
        self.encoder = nn.Sequential(
{layers},
        )
        self.ln = nn.LayerNorm({dim}, eps=1e-6)
        self.head = nn.Linear({dim}, num_classes)

    def forward{s.sig(f"self, {a}")}:
{_ind(fwd, 8)}
''')
    return f"{cls}()", f"torch.randn(1, 3, {size}, {size})"


# --------------------------------------------------------------------------
FAMILIES = {
    "resnet": (resnet_block, resnet_network),
    "inception": (inception_block, inception_network),
    "inverted_residual": (inverted_residual_block, inverted_residual_network),
    "fire": (fire_block, fire_network),
    "dense": (dense_block, dense_network),
    "shuffle": (shuffle_block, shuffle_network),
    "encoder": (encoder_block, encoder_network),
}
LARGE_TIERS = {5: "block", 6: "network"}


def generate_large_program(rng: random.Random, tier: int, family: str | None = None) -> str:
    """One library-style program: tier 5 = a block, tier 6 = a whole network."""
    if tier not in LARGE_TIERS:
        raise ValueError(f"unknown tier {tier}; choose from {sorted(LARGE_TIERS)}")
    family = family or rng.choice(sorted(FAMILIES))
    style = LStyle(rng)
    program = Program(style)
    make = FAMILIES[family][0 if tier == 5 else 1]
    ctor, input_expr = make(rng, style, program)
    return program.render(ctor, input_expr)
