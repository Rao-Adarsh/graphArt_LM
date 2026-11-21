"""Generator v3: whole networks written the way real model code is (hierarchical schema).

    from graphart.generate_v3 import generate_v3_program
    code = generate_v3_program(random.Random(0), kind="network")

WHY A THIRD GENERATOR
---------------------
Under the hierarchical schema (graphart/hier.py) a program's answer is its module tree plus
one wiring graph per block class, so a network's size no longer makes the answer long --
and the tests are now whole networks (docs/hierarchical_plan.md). Measured against real
torchvision and timm networks (docs/generator_rework_plan.md), generator v2's programs were
2-5x shorter, nested less, used fewer classes and block types, and lacked most real-code
idioms (`**kwargs`, `functools.partial`, injected norm layers, `OrderedDict` stages,
`forward_features`/`forward_head`, `nn.Identity` placeholders, shape asserts) and operations
(`permute`, `reshape`, `layer_norm`, attention written out, layer scale).

HOW
---
A network is a stem, 3-5 stages built from a config table, and a head. The stages are built
the way real code builds them -- loops in `__init__` over the config, one of several
containers (one flat `features` Sequential, a Sequential or ModuleList of stages, an
`OrderedDict` of named stages, `add_module`, or a Stage class) -- and every block family
supplies the loop body that builds one stage of its blocks (`Kind.stage_lines`). The block
classes themselves come from generator v2 (graphart/generate_large.py: ResNet, MBConv,
Fire, dense, ShuffleNet, Inception), plus two new families: a ConvNeXt-style block
(depthwise 7x7, channels-last LayerNorm through `permute`, layer scale, drop path) and a
ViT-style network (patch embedding, class token, position embedding, attention written out
or through `F.scaled_dot_product_attention`).

What a network may also carry: a builder function with named configurations and a
`_make_divisible` helper, `partial(nn.BatchNorm2d, eps=...)` as the norm layer, `**kwargs`,
`torch._assert` on the input, `forward_features` / `forward_head(pre_logits=False)`, a
`DropPath` that becomes `nn.Identity()` where its rate is 0 (so one block class appears in
two variants), and layers imported from torchvision.ops whose source is not shown
(`Conv2dNormActivation`, `StochasticDepth`, `SqueezeExcitation`) -- opaque boxes in the truth.

KEEPING THE TESTS UNSEEN
------------------------
Nothing here is modelled on the frozen tests' architectures (FROZEN_v2.json): no shifted
windows, no pooling attention, no segmentation decoders, no DLA / CSP / Ghost modules. Targets
come from aggregate statistics of the dev networks and timm training-type models only.
"""

import random
import re

from . import generate_large as V2
from .generate_large import LStyle, Program, _ind, conv_helpers, hcall, helper_unit

KINDS = ("resnet", "mbconv", "fire", "dense", "shuffle", "inception", "convnext", "regnet",
         "vit", "vgg", "mixer")
# How often each family is drawn: transformer code (attention written out: reshape, permute,
# unbind, matmul, softmax, shape reads) is a large share of real timm models.
WEIGHTS = {"vit": 4, "mixer": 2}
SCHEDULE = tuple(k for k in KINDS for _ in range(WEIGHTS.get(k, 1)))


# --------------------------------------------------------------------------
# style
# --------------------------------------------------------------------------
class V3Style(LStyle):
    """v2's idiom decisions plus the network-level ones v3 adds."""

    def __init__(self, rng: random.Random):
        super().__init__(rng)
        # real code is mostly typed (dev networks: 77% annotate forward, 73% Callable[...] args)
        if not self.typed and rng.random() < 0.5:
            self.typed = True
            self.imports = rng.choice(V2.HEADERS_TYPED)
        self.container = rng.choice(["features", "stages_seq", "modulelist", "ordereddict", "ordereddict",
                                     "add_module", "add_module", "stage_class", "method", "method"])
        self.builder = rng.random() < 0.65          # top-level builder function(s) + config
        self.partial_norm = rng.random() < 0.65     # norm_layer = partial(nn.BatchNorm2d, eps=...)
        self.kwargs_net = rng.random() < 0.6        # Net.__init__(..., **kwargs)
        self.head = rng.choice(["fc", "classifier", "split", "split"])
        self.head_class = rng.random() < 0.35       # a ClassifierHead class (timm)
        self.stem_class = rng.random() < 0.35       # a Stem class
        self.pool = rng.choice(["module", "mean", "functional"])
        self.assert_input = rng.random() < 0.4
        self.ops_imports = rng.random() < 0.4       # torchvision.ops layers, no source shown
        self.drop_path = rng.random() < 0.5         # DropPath(p) if p > 0 else nn.Identity()
        self.divisible = rng.random() < 0.4         # widths through _make_divisible(...)
        self.inject = self.inject or self.partial_norm or rng.random() < 0.4
        # acceptance round 2 (docs/generator_rework_plan.md)
        self.mix = rng.random() < 0.45            # two block families in one network (stages A, then B)
        self.feature_info = rng.random() < 0.5     # timm: self.feature_info.append(dict(num_chs=..., module=...))
        self.grad_ckpt = rng.random() < 0.4        # timm: if self.grad_checkpointing and not torch.jit.is_scripting()
        self.block_list = rng.random() < 0.45      # transformer blocks in an nn.ModuleList, looped in forward
        self.od_head = rng.random() < 0.4          # heads built in an OrderedDict (torchvision ViT)
        self.od_stem = rng.random() < 0.5          # stem = nn.Sequential(OrderedDict([("conv", ...), ...]))
        self.none_checks = rng.random() < 0.7      # `if x is None: x = default` rather than `x = x or default`


class V3Program(Program):
    """Program with extra import lines collected as definitions are added."""

    def __init__(self, style):
        super().__init__(style)
        self.imports: list[str] = []

    def need(self, line: str) -> None:
        if line not in self.imports:
            self.imports.append(line)

    def render(self, ctor: str, input_expr: str) -> str:
        s = self.style
        head = s.imports + ("\n" + "\n".join(self.imports) if self.imports else "")
        return (f"{head}\n\n\n" + "\n\n\n".join(self.defs) +
                f"\n\n\n{s.instance} = {ctor}\nexample_input = {input_expr}\n")


# --------------------------------------------------------------------------
# shared helpers written into programs
# --------------------------------------------------------------------------
def make_divisible_def(p: V3Program) -> str:
    name = p.style.rng.choice(["_make_divisible", "make_divisible", "_round_channels"])
    p.add(name, f'''
def {name}(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v
''')
    return name


def drop_path_def(p: V3Program) -> str:
    """A DropPath class (timm style) -- or torchvision's StochasticDepth, imported."""
    s = p.style
    if s.ops_imports:
        p.need("from torchvision.ops import StochasticDepth")
        return "StochasticDepth"
    fn = s.rng.choice(["drop_path", "_drop_path", "stochastic_depth"])
    cls = s.rng.choice(["DropPath", "StochasticDrop", "DropBlockPath"])
    p.add(fn, f'''
def {fn}(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    return x * random_tensor / keep_prob
''')
    p.add(cls, f'''
class {cls}(nn.Module):
    def __init__(self, drop_prob=None):
        {s.super_line(cls)}
        self.drop_prob = drop_prob

    def forward(self, x):
        return {fn}(x, self.drop_prob, self.training)
''')
    return cls


def dp_expr(dp_cls: str, rate: str) -> str:
    if dp_cls == "StochasticDepth":
        return f'StochasticDepth({rate}, "row") if {rate} > 0.0 else nn.Identity()'
    return f"{dp_cls}({rate}) if {rate} > 0.0 else nn.Identity()"


def layernorm2d_def(p: V3Program) -> str:
    s = p.style
    name = s.rng.choice(["LayerNorm2d", "ChannelsFirstLayerNorm", "LayerNormChannels"])
    if s.rng.random() < 0.5:
        p.add(name, f'''
class {name}(nn.LayerNorm):
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x
''')
    else:
        p.add(name, f'''
class {name}(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        {s.super_line(name)}
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
''')
    return name


# --------------------------------------------------------------------------
# block families: each supplies the code that builds one stage of its blocks
# --------------------------------------------------------------------------
class Kind:
    """A block family inside a network.

    `cols` names the config-table columns; `row(i, prev)` draws one stage's row;
    `stage_lines(v)` returns the code that appends one stage's modules to the list named
    v["blocks"], reading the row's columns by name, the input width from v["in"] and
    setting it to the stage's output width; `block_ctor()` returns a single block for the
    block tier. All widths are small so tracing stays fast.
    """
    cols: tuple = ()
    pools_between = False          # spatial downsampling by a pool before each stage > 0

    def __init__(self, p: V3Program, rng: random.Random):
        self.p, self.rng, self.s = p, rng, p.style


class ResNetKind(Kind):
    cols = ("planes", "n", "stride")

    def __init__(self, p, rng):
        super().__init__(p, rng)
        self.bottleneck = rng.random() < 0.4
        self.cls, self.exp = V2.resnet_block_def(p, rng, self.bottleneck)
        self.k1 = [n for n in p.have if n in ("conv1x1", "conv1", "make_conv1")][0]
        self.ref = self.cls                     # the name the stage code builds blocks with
        self.net_block = self.net_kw = None     # what the network takes for it (see _network_class)
        self.down = None
        if getattr(p, "network", False) and self.s.container != "stage_class":
            if rng.random() < 0.6:
                # torchvision: both block classes in the file, the network is given one
                V2.resnet_block_def(p, rng, not self.bottleneck)
                self.ref, self.net_block = "block", self.cls
            if rng.random() < 0.5:
                # timm: downsample built by a helper chosen by a config string
                self.down = self._down_defs()
                self.net_kw = f'down_type={rng.choice(["conv", "avg"])!r}'

    def _down_defs(self):
        s, p = self.s, self.p
        dflt = (["if norm_layer is None:", "    norm_layer = nn.BatchNorm2d"] if s.rng.random() < 0.6
                else ["norm_layer = norm_layer or nn.BatchNorm2d"])
        p.add("downsample_conv", f'''
def downsample_conv(in_chs, out_chs, stride=1, norm_layer=None):
{_ind(dflt, 4)}
    return nn.Sequential(nn.Conv2d(in_chs, out_chs, kernel_size=1, stride=stride, bias=False), norm_layer(out_chs))
''')
        p.add("downsample_avg", f'''
def downsample_avg(in_chs, out_chs, stride=1, norm_layer=None):
{_ind(dflt, 4)}
    pool = nn.AvgPool2d(2, stride, ceil_mode=True, count_include_pad=False) if stride > 1 else nn.Identity()
    return nn.Sequential(pool, nn.Conv2d(in_chs, out_chs, kernel_size=1, stride=1, bias=False), norm_layer(out_chs))
''')
        return True

    def row(self, i, prev):
        return (8 * 2 ** i, self.rng.randint(1, 3), 1 if i == 0 else 2)

    def stage_lines(self, v):
        norm = v.get("norm", "nn.BatchNorm2d")
        b = self.ref
        if self.down:
            make = [f'    down_fn = downsample_avg if down_type == "avg" else downsample_conv',
                    f"    downsample = down_fn({v['in']}, planes * {b}.expansion, stride, {norm})"]
        else:
            make = [f"    downsample = nn.Sequential(",
                    f"        {self.k1}({v['in']}, planes * {b}.expansion, stride),",
                    f"        {norm}(planes * {b}.expansion),",
                    f"    )"]
        return [f"downsample = None",
                f"if stride != 1 or {v['in']} != planes * {b}.expansion:"] + make + [
                f"{v['blocks']}.append({b}({v['in']}, planes, stride, downsample))",
                f"{v['in']} = planes * {b}.expansion",
                f"for _ in range(1, n):",
                f"    {v['blocks']}.append({b}({v['in']}, planes))"]

    def block_ctor(self):
        planes = self.rng.choice([8, 16])
        cin = planes * self.exp
        return f"{self.cls}({cin}, {planes})", cin


class MBConvKind(Kind):
    cols = ("t", "c", "n", "stride", "se")

    def __init__(self, p, rng):
        super().__init__(p, rng)
        self.cls, self.has_se = V2.inverted_residual_def(p, rng)

    def row(self, i, prev):
        return (1 if i == 0 else self.rng.choice([3, 4, 6]), 8 * (i + 2), self.rng.randint(1, 3),
                1 if i == 0 else 2, self.rng.random() < 0.5)

    def stage_lines(self, v):
        se = ", use_se=se" if self.has_se else ""
        width = f"{v['div']}(c * width_mult, 8)" if v.get("div") else "c"
        return [f"out_c = {width}",
                f"for i in range(n):",
                f"    {v['blocks']}.append({self.cls}({v['in']}, out_c, stride if i == 0 else 1, t{se}))",
                f"    {v['in']} = out_c"]

    def block_ctor(self):
        c = self.rng.choice([16, 24])
        se = ", use_se=True" if self.has_se and self.rng.random() < 0.6 else ""
        return f"{self.cls}({c}, {c}, 1, {self.rng.choice([3, 4, 6])}{se})", c


class FireKind(Kind):
    cols = ("sq", "e", "n")
    pools_between = True

    def __init__(self, p, rng):
        super().__init__(p, rng)
        self.cls = V2.fire_def(p, rng)

    def row(self, i, prev):
        sq = 8 * (i + 1)
        return (sq, sq * 4, self.rng.randint(1, 3))

    def stage_lines(self, v):
        return [f"for _ in range(n):",
                f"    {v['blocks']}.append({self.cls}({v['in']}, sq, e, e))",
                f"    {v['in']} = 2 * e"]

    def block_ctor(self):
        return f"{self.cls}(32, 8, 16, 16)", 32


class DenseKind(Kind):
    cols = ("n",)

    def __init__(self, p, rng):
        super().__init__(p, rng)
        self.layer, self.block = V2.dense_defs(p, rng)
        self.growth, self.bn_size = rng.choice([8, 12]), rng.choice([2, 4])
        s = p.style
        self.trans = rng.choice(["Transition", "_Transition", "TransitionLayer"])
        p.add(self.trans, f'''
class {self.trans}(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        {s.super_line(self.trans)}
        self.norm = nn.BatchNorm2d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
''')

    def row(self, i, prev):
        return (self.rng.randint(2, 4),)

    def stage_lines(self, v):
        g, b = self.growth, self.bn_size
        return [f"{v['blocks']}.append({self.block}(n, {v['in']}, {b}, {g}))",
                f"{v['in']} = {v['in']} + n * {g}",
                f"if not last:",
                f"    {v['blocks']}.append({self.trans}({v['in']}, {v['in']} // 2))",
                f"    {v['in']} = {v['in']} // 2"]

    def block_ctor(self):
        return f"{self.block}(3, 16, {self.bn_size}, {self.growth})", 16


class ShuffleKind(Kind):
    cols = ("c", "n")

    def __init__(self, p, rng):
        super().__init__(p, rng)
        self.cls = V2.shuffle_defs(p, rng)

    def row(self, i, prev):
        return (16 * 2 ** i, self.rng.randint(1, 3))

    def stage_lines(self, v):
        return [f"{v['blocks']}.append({self.cls}({v['in']}, c, 2))",
                f"for _ in range(n - 1):",
                f"    {v['blocks']}.append({self.cls}(c, c, 1))",
                f"{v['in']} = c"]

    def block_ctor(self):
        return f"{self.cls}(16, 16, 1)", 16


class InceptionKind(Kind):
    """Inception networks are written out block by block (as torchvision's are)."""
    cols = ()
    pools_between = True

    def __init__(self, p, rng):
        super().__init__(p, rng)
        self.cls, self.out_ch, self.n_br, self.pool = V2.inception_block_def(p, rng)

    def block_ctor(self):
        w = V2._inception_args(self.rng, self.n_br, self.pool)
        return f"{self.cls}(16, {', '.join(map(str, w.values()))})", 16


class ConvNeXtKind(Kind):
    cols = ("dim", "n")

    def __init__(self, p, rng):
        super().__init__(p, rng)
        s = p.style
        self.ln2d = layernorm2d_def(p)
        self.dp = drop_path_def(p) if s.drop_path else None
        self.cls = rng.choice(["CNBlock", "ConvNeXtBlock", "DepthwiseMLPBlock"])
        a = s.arg
        ls = rng.random() < 0.7
        perm_norm = rng.random() < 0.6          # channels-last LayerNorm via permute, or LayerNorm2d
        init = [f"self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)"]
        if perm_norm:
            init.append("self.norm = nn.LayerNorm(dim, eps=1e-6)")
        else:
            init.append(f"self.norm = {self.ln2d}(dim, eps=1e-6)")
        lin = "nn.Linear" if perm_norm else "nn.Conv2d"
        k = "" if perm_norm else ", kernel_size=1"
        init += [f"self.pwconv1 = {lin}(dim, {self.rng.choice([2, 4])} * dim{k})", "self.act = nn.GELU()"]
        init.append(f"self.pwconv2 = {lin}(self.pwconv1.out_{'features' if perm_norm else 'channels'}, dim{k})")
        if ls:
            init.append("self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim)) "
                        "if layer_scale_init_value > 0 else None")
        if self.dp:
            init.append(f"self.drop_path = {dp_expr(self.dp, 'drop_path')}")
        fwd = ["shortcut = " + a, f"{a} = self.dwconv({a})"]
        if perm_norm:
            fwd.append(f"{a} = {a}.permute(0, 2, 3, 1)")
        fwd += [f"{a} = self.norm({a})", f"{a} = self.pwconv1({a})", f"{a} = self.act({a})", f"{a} = self.pwconv2({a})"]
        if ls:
            g = "self.gamma" if perm_norm else "self.gamma.reshape(1, -1, 1, 1)"
            fwd += ["if self.gamma is not None:", f"    {a} = {g} * {a}"]
        if perm_norm:
            fwd.append(f"{a} = {a}.permute(0, 3, 1, 2)")
        fwd.append(f"{a} = shortcut + self.drop_path({a})" if self.dp else f"{a} = shortcut + {a}")
        fwd.append(f"return {a}")
        params = "dim, drop_path=0.0" + (", layer_scale_init_value=1e-6" if ls else "")
        p.add(self.cls, f'''
class {self.cls}(nn.Module):
    def __init__(self, {params}):
        {s.super_line(self.cls)}
{_ind(init, 8)}

    def forward{s.sig(f"self, {a}")}:
{_ind(fwd, 8)}
''')
        self.has_dp = bool(self.dp)

    def row(self, i, prev):
        return (8 * 2 ** (i + 1), self.rng.randint(1, 3))

    def stage_lines(self, v):
        dp = ", drop_path=dp_rates[cur + j]" if self.has_dp else ""
        lines = [f"if {v['in']} != dim:",
                 f"    {v['blocks']}.append(nn.Sequential({self.ln2d}({v['in']}, eps=1e-6), "
                 f"nn.Conv2d({v['in']}, dim, kernel_size=2, stride=2)))",
                 f"for j in range(n):",
                 f"    {v['blocks']}.append({self.cls}(dim{dp}))"]
        if self.has_dp:
            lines.append("cur += n")
        lines.append(f"{v['in']} = dim")
        return lines

    def block_ctor(self):
        return f"{self.cls}(16)", 16


class RegNetKind(Kind):
    """RegNet-style X/Y block: 1x1 -> grouped 3x3 -> (squeeze-excite) -> 1x1, projection shortcut."""
    cols = ("w", "d", "g")

    def __init__(self, p, rng):
        super().__init__(p, rng)
        s = p.style
        self.cls = rng.choice(["ResBottleneckBlock", "XBlock", "RegNetBlock", "YBlock"])
        self.se = rng.random() < 0.5
        se_cls = None
        if self.se:
            if s.ops_imports:
                p.need("from torchvision.ops import SqueezeExcitation")
                se_cls = "SqueezeExcitation"
            else:
                se_cls = V2.se_def(p, rng)
        a = s.arg
        names = rng.choice([("a", "b", "c"), ("conv_a", "conv_b", "conv_c"), ("f_a", "f_b", "f_c")])
        proj = rng.choice(["proj", "shortcut", "downsample"])
        act = s.act_ctor()
        se_init = (f"self.se = {se_cls}(w_b, max(1, int(round(se_ratio * in_ch)))) if se_ratio else None"
                   if self.se else "self.se = None")
        self.helper = rng.choice(["adjust_width", "_quantize", "round_width"])
        p.add(self.helper, f'''
def {self.helper}(width, group_width):
    group_width = min(group_width, width)
    return int(round(width / group_width) * group_width)
''')
        p.add(self.cls, f'''
class {self.cls}(nn.Module):
    def __init__(self, in_ch, out_ch, stride, group_width, bottleneck_ratio=1.0, se_ratio=None):
        {s.super_line(self.cls)}
        w_b = int(round(out_ch * bottleneck_ratio))
        groups = w_b // group_width
        self.{names[0]} = nn.Sequential(nn.Conv2d(in_ch, w_b, 1, bias=False), nn.BatchNorm2d(w_b), {act})
        self.{names[1]} = nn.Sequential(
            nn.Conv2d(w_b, w_b, 3, stride=stride, padding=1, groups=groups, bias=False),
            nn.BatchNorm2d(w_b),
            {act},
        )
        {se_init}
        self.{names[2]} = nn.Sequential(nn.Conv2d(w_b, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch))
        self.{proj} = None
        if in_ch != out_ch or stride != 1:
            self.{proj} = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False), nn.BatchNorm2d(out_ch))
        self.act = {act}

    def forward{s.sig(f"self, {a}")}:
        shortcut = {a} if self.{proj} is None else self.{proj}({a})
        {a} = self.{names[0]}({a})
        {a} = self.{names[1]}({a})
        if self.se is not None:
            {a} = self.se({a})
        {a} = self.{names[2]}({a})
        return self.act({a} + shortcut)
''')

    def row(self, i, prev):
        g = self.rng.choice([8, 16])
        return (16 * (i + 1) + self.rng.choice([0, 8]), self.rng.randint(1, 3), g)

    def stage_lines(self, v):
        se = f", se_ratio={self.rng.choice([0.25, 0.25, 0.5])}" if self.se else ""
        return [f"w = {self.helper}(w, g)",
                f"for i in range(d):",
                f"    {v['blocks']}.append({self.cls}({v['in']}, w, 2 if i == 0 else 1, g{se}))",
                f"    {v['in']} = w"]

    def block_ctor(self):
        se = ", se_ratio=0.25" if self.se else ""
        return f"{self.cls}(16, 32, 2, 8{se})", 16


CONV_KINDS = {"resnet": ResNetKind, "mbconv": MBConvKind, "fire": FireKind, "dense": DenseKind,
              "shuffle": ShuffleKind, "inception": InceptionKind, "convnext": ConvNeXtKind,
              "regnet": RegNetKind}


# --------------------------------------------------------------------------
# the CNN network template
# --------------------------------------------------------------------------
def _stem(p: V3Program, kind: Kind, rng) -> tuple[list[str], int]:
    """Lines that set self.stem (or conv/bn/act attributes); returns (init lines, width, forward lines)."""
    s = p.style
    a = s.arg
    w = rng.choice([8, 16])
    if isinstance(kind, ConvNeXtKind):
        return [f"self.stem = nn.Sequential(nn.Conv2d(in_chans, {w}, kernel_size=4, stride=4), "
                f"{kind.ln2d}({w}, eps=1e-6))"], w, [f"{a} = self.stem({a})"]
    style = rng.choice(["ops", "helper", "explicit", "explicit"]) if s.ops_imports else \
        rng.choice(["helper", "explicit", "explicit"])
    if style == "ops":
        p.need("from torchvision.ops import Conv2dNormActivation")
        return [f"self.stem = Conv2dNormActivation(in_chans, {w}, kernel_size=3, stride=2)"], w, \
               [f"{a} = self.stem({a})"]
    if style == "helper":
        H = helper_unit(p)
        return [f"self.stem = {hcall(s, H, 'in_chans', w, 3, stride=2)}"], w, [f"{a} = self.stem({a})"]
    norm = "norm_layer" if s.inject else "nn.BatchNorm2d"
    act = s.act_ctor()
    if s.od_stem and not s.stem_class:
        p.need("from collections import OrderedDict")
        return ["self.stem = nn.Sequential(OrderedDict([",
                f'    ("conv", nn.Conv2d(in_chans, {w}, kernel_size=3, stride=2, padding=1, bias=False)),',
                f'    ("bn", {norm}({w})),',
                f'    ("act", {act}),',
                "]))"], w, [f"{a} = self.stem({a})"]
    if s.stem_class:
        name = rng.choice(["Stem", "StemBlock", "ConvStem"])
        pool = rng.random() < 0.5
        p.add(name, f'''
class {name}(nn.Module):
    def __init__(self, in_chs, out_chs, norm_layer=nn.BatchNorm2d):
        {s.super_line(name)}
        self.conv = nn.Conv2d(in_chs, out_chs, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm = norm_layer(out_chs)
        self.act = {act}
{"        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)" if pool else ""}

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
{"        x = self.pool(x)" if pool else ""}
        return x
''')
        arg = ", norm_layer=norm_layer" if s.inject else ""
        return [f"self.stem = {name}(in_chans, {w}{arg})"], w, [f"{a} = self.stem({a})"]
    return [f"self.conv_stem = nn.Conv2d(in_chans, {w}, kernel_size=3, stride=2, padding=1, bias=False)",
            f"self.bn1 = {norm}({w})", f"self.act1 = {act}"], w, \
           [f"{a} = self.conv_stem({a})", f"{a} = self.bn1({a})", f"{a} = self.act1({a})"]


def head_class_def(p: V3Program) -> str:
    """timm's ClassifierHead: pool, flatten, dropout, classifier in a class of its own."""
    s = p.style
    name = s.rng.choice(["ClassifierHead", "Head", "NormMlpHead", "PoolHead"])
    pool = ("self.global_pool = nn.AdaptiveAvgPool2d(1)", "x = self.global_pool(x)") if s.rng.random() < 0.6 \
        else ("", "x = x.mean((2, 3), keepdim=True)")
    p.add(name, f'''
class {name}(nn.Module):
    def __init__(self, in_features, num_classes, drop_rate=0.0):
        {s.super_line(name)}
        {pool[0]}
        self.flatten = nn.Flatten(1)
        self.drop = nn.Dropout(drop_rate)
        self.fc = nn.Linear(in_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x, pre_logits: bool = False):
        {pool[1]}
        x = self.flatten(x)
        x = self.drop(x)
        return x if pre_logits else self.fc(x)
''')
    return name


def _head(s: V3Style, rng, width_var: str, p: V3Program) -> tuple[list[str], list[str], list[str]]:
    """(init lines, forward_features tail, forward_head lines) for the classifier head."""
    a = s.arg
    init, head = [], []
    if s.pool == "module":
        init.append(f"self.global_pool = nn.AdaptiveAvgPool2d({rng.choice(['1', '(1, 1)'])})")
        head.append(f"{a} = self.global_pool({a})")
    elif s.pool == "functional":
        head.append(f"{a} = F.adaptive_avg_pool2d({a}, 1)")
    else:
        head.append(f"{a} = {a}.mean(({rng.choice(['2, 3', '-2, -1'])}){', keepdim=True' if rng.random() < 0.3 else ''})")
    if s.head == "classifier" and s.od_head:
        p.need("from collections import OrderedDict")
        init += ["self.classifier = nn.Sequential(OrderedDict([",
                 '    ("flatten", nn.Flatten(1)),',
                 '    ("drop", nn.Dropout(p=drop_rate)),',
                 f'    ("fc", nn.Linear({width_var}, num_classes)),',
                 "]))"]
        head.append(f"return self.classifier({a})")
    elif s.head == "classifier":
        init.append(f"self.classifier = nn.Sequential(nn.Flatten(1), nn.Dropout(p=drop_rate), "
                    f"nn.Linear({width_var}, num_classes))")
        head.append(f"return self.classifier({a})")
    else:
        flat = rng.choice([f"torch.flatten({a}, 1)", f"{a}.flatten(1)", f"self.flatten({a})"])
        if "self.flatten" in flat:
            init.append("self.flatten = nn.Flatten(1)")
        init += ["self.drop = nn.Dropout(drop_rate)" if rng.random() < 0.6 else "self.drop = nn.Identity()",
                 f"self.fc = nn.Linear({width_var}, num_classes)"]
        head += [f"{a} = {flat}", f"{a} = self.drop({a})"]
        if s.head == "split":
            head.append(f"return {a} if pre_logits else self.fc({a})")
        else:
            head.append(f"return self.fc({a})")
    return init, head


MIXABLE = ("resnet", "mbconv", "regnet", "shuffle", "fire")
LOOP_FORMS = ("features", "stages_seq", "modulelist", "ordereddict", "add_module")


def _loop_head(k: Kind, tbl: str, start: str = "") -> list[str]:
    unpack = ", ".join(k.cols)
    en = f"enumerate({tbl}, start={start})" if start else f"enumerate({tbl})"
    head = [f"for idx, ({unpack}{',' if len(k.cols) == 1 else ''}) in {en}:"]
    if isinstance(k, DenseKind):
        head.append(f"    last = idx == len({tbl}) - 1")
    return head


def _pool_line(k: Kind) -> list[str]:
    return ["    if idx > 0:", "        blocks.append(nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True))"] \
        if k.pools_between else []


def cnn_network(rng: random.Random, p: V3Program, kind_name: str):
    s = p.style
    p.network = True
    kind = CONV_KINDS[kind_name](p, rng)
    a = s.arg
    cls = rng.choice(["Net", "Backbone", "ConvNet", "TinyNet", "Model", "Classifier", "Network"])
    if cls in p.have:
        cls += "V3"
    n_stages = rng.randint(3, 5)
    kind2 = rows2 = table2 = None
    if s.mix and kind_name in MIXABLE and s.container in LOOP_FORMS:
        # e.g. MobileNet/EfficientNet-V2/byobnet: early stages of one block type, later of another
        kind2 = CONV_KINDS[rng.choice([k for k in MIXABLE if k != kind_name])](p, rng)
        n_stages = rng.randint(2, 3)
    extra_methods = ""
    stem_init, stem_w, stem_fwd = _stem(p, kind, rng)
    inits, fwd = [], []
    if s.inject and not isinstance(kind, ConvNeXtKind):
        default = (f"partial(nn.BatchNorm2d, eps={rng.choice(['1e-5', '1e-3', '0.001'])})" if s.partial_norm
                   else "nn.BatchNorm2d")
        if s.partial_norm:
            p.need("from functools import partial")
        inits += ["if norm_layer is None:", f"    norm_layer = {default}"]
    inits += [f"self.num_classes = num_classes"] + stem_init + [f"in_ch = {stem_w}"]
    fwd += stem_fwd

    if isinstance(kind, InceptionKind):          # written out, block by block
        idx = 0
        for st in range(n_stages):
            if st:
                fwd.append(f"{a} = F.max_pool2d({a}, kernel_size=3, stride=2, padding=1)")
            for _ in range(rng.randint(1, 2)):
                idx += 1
                w = V2._inception_args(rng, kind.n_br, kind.pool)
                inits.append(f"self.mixed_{idx} = {kind.cls}(in_ch, {', '.join(map(str, w.values()))})")
                inits.append(f"in_ch = {kind.out_ch(w)}")
                fwd.append(f"{a} = self.mixed_{idx}({a})")
        body_defs = ""
    else:
        rows = []
        prev = stem_w
        for i in range(n_stages):
            rows.append(kind.row(i, prev))
        table_name = rng.choice(["cfg", "arch", "stage_cfg", "settings"])
        div = make_divisible_def(p) if (s.divisible and isinstance(kind, MBConvKind)) else None
        v = {"in": "in_ch", "blocks": "blocks", "div": div}
        if s.inject and isinstance(kind, ResNetKind):
            v["norm"] = "norm_layer"
        stage_code = kind.stage_lines(v)
        if kind2 is not None:
            table_name, table2 = rng.choice([("early_cfg", "late_cfg"), ("conv_cfg", "block_cfg"),
                                             ("cfg_a", "cfg_b"), ("stem_stages", "main_stages")])
            rows2 = [kind2.row(n_stages + j, None) for j in range(rng.randint(1, 5 - n_stages))]
            div2 = (div or make_divisible_def(p)) if (s.divisible and isinstance(kind2, MBConvKind)) else None
            v2 = {"in": "in_ch", "blocks": "blocks", "div": div2}
            if s.inject and isinstance(kind2, ResNetKind):
                v2["norm"] = "norm_layer"
            stage_code2 = kind2.stage_lines(v2)
        unpack = ", ".join(kind.cols)
        if isinstance(kind, ConvNeXtKind) and kind.has_dp:
            inits += [f"total = sum(r[1] for r in {table_name})",
                      "dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, total)]", "cur = 0"]
        if isinstance(kind, DenseKind):
            loop_head = [f"for idx, ({unpack},) in enumerate({table_name}):", f"    last = idx == len({table_name}) - 1"]
        else:
            loop_head = [f"for idx, ({unpack}{',' if len(kind.cols) == 1 else ''}) in enumerate({table_name}):"]
        pool_line = ["    if idx > 0:", "        blocks.append(nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True))"] \
            if kind.pools_between else []
        container = s.container
        if container == "method" and isinstance(kind, ConvNeXtKind) and kind.has_dp:
            container = "stages_seq"          # per-stage drop-path slices: kept to the loop forms
        extra_methods = ""
        if container == "method":
            # torchvision's ResNet: self.layerN = self._make_layer(...), written out per stage
            mname = rng.choice(["_make_layer", "_make_stage", "_build_stage"])
            code = list(stage_code)
            uses_self = False
            if "norm_layer" in "\n".join(code):
                code = [l.replace("norm_layer", "self._norm_layer") for l in code]
                inits.append("self._norm_layer = norm_layer")
                uses_self = True
            if "width_mult" in "\n".join(code):
                code = [l.replace("width_mult", "self.width_mult") for l in code]
                inits.append("self.width_mult = width_mult")
                uses_self = True
            for name in ("block", "down_type"):
                if re.search(rf"\b{name}\b", "\n".join(code)):
                    code = [re.sub(rf"\b{name}\b", f"self.{name}", l) for l in code]
                    inits.append(f"self.{name} = {name}")
                    uses_self = True
            static = not uses_self and rng.random() < 0.5
            pool_p = ", pool=False" if kind.pools_between else ""
            last_p = ", last=False" if isinstance(kind, DenseKind) else ""
            pool_l = ["if pool:", "    blocks.append(nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True))"] \
                if kind.pools_between else []
            deco = "    @staticmethod\n" if static else ""
            first = "" if static else "self, "
            extra_methods = f'''
{deco}    def {mname}({first}{unpack}, in_ch{pool_p}{last_p}):
        blocks = []
{_ind(pool_l + code, 8)}
        return nn.Sequential(*blocks), in_ch
'''
            nm = rng.choice(["layer", "stage"])
            loop = []
            for i in range(n_stages):
                args = f"*{table_name}[{i}], in_ch"
                if kind.pools_between:
                    args += f", pool={i > 0}"
                if isinstance(kind, DenseKind):
                    args += f", last={i == n_stages - 1}"
                loop.append(f"self.{nm}{i + 1}, in_ch = self.{mname}({args})")
                if s.feature_info:
                    loop.append(f'self.feature_info.append(dict(num_chs=in_ch, module="{nm}{i + 1}"))')
            if s.feature_info:
                loop = ["self.feature_info = []"] + loop
            stage_fwd = [f"{a} = self.{nm}{i + 1}({a})" for i in range(n_stages)]
            body_defs = ""
        elif container == "stage_class":
            body_defs = _stage_class(p, kind, v, stage_code, unpack)
            inner = [f"    stage = {p.stage_cls}({unpack}, in_ch{', idx > 0' if kind.pools_between else ''}"
                     f"{', last' if isinstance(kind, DenseKind) else ''})",
                     "    stages.append(stage)", "    in_ch = stage.out_channels"]
            loop = ["stages = []"] + loop_head + inner
            if isinstance(kind, ConvNeXtKind) and kind.has_dp:
                loop = ["stages = []"] + loop_head + [
                    f"    stage = {p.stage_cls}({unpack}, in_ch, dp_rates[cur:cur + n])",
                    "    cur += n", "    stages.append(stage)", "    in_ch = stage.out_channels"]
            seq_or_list = rng.choice(["seq", "list"])
            loop.append("self.stages = nn.Sequential(*stages)" if seq_or_list == "seq" else "self.stages = nn.ModuleList(stages)")
            stage_fwd = [f"{a} = self.stages({a})"] if seq_or_list == "seq" else \
                ["for stage in self.stages:", f"    {a} = stage({a})"]
        else:
            body_defs = ""
            parts = [(kind, table_name, stage_code, "")]
            if kind2 is not None:
                parts.append((kind2, table2, stage_code2, f"len({table_name})"))
            od_nm = rng.choice(["stage", "layer", "block"])
            am_nm = rng.choice(["layer", "stage"])
            fi_mod = {"features": 'f"features.{len(blocks) - 1}"', "stages_seq": 'f"stages.{idx}"',
                      "modulelist": 'f"stages.{idx}"', "ordereddict": f'f"stages.{od_nm}{{idx + 1}}"',
                      "add_module": f'f"{am_nm}{{idx + 1}}"'}[container]
            fi = [f"    self.feature_info.append(dict(num_chs=in_ch, module={fi_mod}))"] if s.feature_info else []
            loop = ["self.feature_info = []"] if s.feature_info else []
            loop += {"features": ["blocks = []"], "stages_seq": ["stages = []"],
                     "modulelist": ["self.stages = nn.ModuleList()"], "ordereddict": ["stages = OrderedDict()"],
                     "add_module": []}[container]
            for k, tbl, code, start in parts:
                head_l, pl = _loop_head(k, tbl, start), _pool_line(k)
                if container == "features":
                    loop += head_l + pl + _ind(code, 4).split("\n") + fi
                    continue
                per = ["    blocks = []"] + pl + _ind(code, 4).split("\n")
                store = {"stages_seq": ["    stages.append(nn.Sequential(*blocks))"],
                         "modulelist": ["    self.stages.append(nn.Sequential(*blocks))"],
                         "ordereddict": [f'    stages[f"{od_nm}{{idx + 1}}"] = nn.Sequential(*blocks)'],
                         "add_module": [f'    self.add_module(f"{am_nm}{{idx + 1}}", nn.Sequential(*blocks))']}[container]
                loop += head_l + per + store + fi
            seq = "features" if container == "features" else "stages"
            if container == "features":
                loop.append("self.features = nn.Sequential(*blocks)")
            elif container == "stages_seq":
                loop.append("self.stages = nn.Sequential(*stages)")
            elif container == "ordereddict":
                p.need("from collections import OrderedDict")
                loop.append("self.stages = nn.Sequential(stages)")
            elif container == "add_module":
                loop.append(f"self.num_stages = {' + '.join(f'len({t})' for _, t, _, _ in parts)}")
            if container in ("features", "stages_seq", "ordereddict"):
                call = f"{a} = self.{seq}({a})"
                if s.grad_ckpt:
                    p.need("from torch.utils.checkpoint import checkpoint_sequential")
                    loop.append("self.grad_checkpointing = False")
                    stage_fwd = ["if self.grad_checkpointing and not torch.jit.is_scripting():",
                                 f"    {a} = checkpoint_sequential(self.{seq}, len(self.{seq}), {a})",
                                 "else:", f"    {call}"]
                else:
                    stage_fwd = [call]
            elif container == "modulelist":
                stage_fwd = ["for stage in self.stages:", f"    {a} = stage({a})"]
            else:
                stage_fwd = ["for i in range(self.num_stages):", f'    {a} = getattr(self, f"{am_nm}{{i + 1}}")({a})']
            if s.feature_info:
                loop.append("self.num_features = in_ch")
        inits += loop
        fwd += stage_fwd
    if s.head_class and s.head != "classifier":
        hc = head_class_def(p)
        split = s.head == "split"
        head_init = [f"self.head = {hc}(in_ch, num_classes, drop_rate=drop_rate)"]
        head = [f"return self.head({a}{', pre_logits=pre_logits' if split else ''})"]
    else:
        head_init, head = _head(s, rng, "in_ch", p)
    if rng.random() < 0.35 and not isinstance(kind, InceptionKind):
        last = rng.choice([64, 128])
        if s.ops_imports:
            p.need("from torchvision.ops import Conv2dNormActivation")
            inits += [f"self.conv_head = Conv2dNormActivation(in_ch, {last}, kernel_size=1)", f"in_ch = {last}"]
            fwd += [f"{a} = self.conv_head({a})"]
        else:
            inits += [f"self.conv_head = nn.Conv2d(in_ch, {last}, kernel_size=1, bias=False)",
                      f"self.act_head = {s.act_ctor()}", f"in_ch = {last}"]
            fwd += [f"{a} = self.conv_head({a})", f"{a} = self.act_head({a})"]
    inits += head_init
    if s.assert_input:
        fwd = [f'torch._assert({a}.dim() == 4, f"expected a 4D input, got {{{a}.dim()}}D")'] + fwd
    return _network_class(p, rng, cls, inits, fwd, head, kind, rows if not isinstance(kind, InceptionKind) else None,
                          table_name if not isinstance(kind, InceptionKind) else None, extra_methods,
                          kind2, rows2, table2)


def _stage_class(p: V3Program, kind: Kind, v: dict, stage_code: list[str], unpack: str) -> str:
    s = p.style
    name = p.style.rng.choice(["Stage", "ResStage", "BlockStage", "NetStage"])
    p.stage_cls = name
    extra, pool = "", []
    if kind.pools_between:
        extra = ", pool=False"
        pool = ["if pool:", "    blocks.append(nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True))"]
    if isinstance(kind, DenseKind):
        extra += ", last=False"
    dp_param = ""
    code = stage_code
    if isinstance(kind, ConvNeXtKind) and kind.has_dp:
        dp_param = ", dp_rates=None"
        code = [l.replace("dp_rates[cur + j]", "dp_rates[j]") for l in stage_code if l != "cur += n"]
    if "width_mult" in "\n".join(code):
        extra += ", width_mult=1.0"
    norm = ", norm_layer=None" if "norm_layer" in "\n".join(code) else ""
    pre = ["if norm_layer is None:", "    norm_layer = nn.BatchNorm2d"] if norm else []
    hold = s.rng.choice(["blocks", "layers"])
    fwd_body = [f"return self.{hold}(x)"]
    p.add(name, f'''
class {name}(nn.Module):
    def __init__(self, {unpack}, in_ch{extra}{dp_param}{norm}):
        {s.super_line(name)}
{_ind(pre, 8)}
        blocks = []
{_ind(pool + code, 8)}
        self.{hold} = nn.Sequential(*blocks)
        self.out_channels = in_ch

    def forward(self, x):
{_ind(fwd_body, 8)}
''')
    return name


def _network_class(p, rng, cls, inits, fwd, head, kind, rows, table_name, extra_methods="",
                   kind2=None, rows2=None, table2=None):
    s: V3Style = p.style
    a = s.arg
    params = ["in_chans=3", f"num_classes={rng.choice([10, 100, 1000])}", "drop_rate=0.0"]
    if isinstance(kind, ConvNeXtKind) and kind.has_dp:
        params.append("drop_path_rate=0.1")
    if s.divisible and any(isinstance(k, MBConvKind) for k in (kind, kind2)):
        params.append("width_mult=1.0")
    if s.inject and not isinstance(kind, ConvNeXtKind):
        params.append(s.callable_arg("norm_layer"))
    if rows is not None:
        params.insert(0, table_name)
    if rows2 is not None:
        params.insert(1, table2)
    lead = [k.net_block for k in (kind, kind2) if k is not None and getattr(k, "net_block", None)]
    if lead:
        params.insert(0, "block")
    params += [k.net_kw for k in (kind, kind2) if k is not None and getattr(k, "net_kw", None)]
    if s.kwargs_net:
        params.append("**kwargs")
    if rng.random() < 0.5:
        params.insert(-1 if s.kwargs_net else len(params), "output_stride=32")
        inits = ["assert output_stride in (8, 16, 32)"] + inits
    if s.head == "split":
        feats = fwd + [f"return {a}"]
        methods = f'''
    def forward_features(self, {a}):
{_ind(feats, 8)}

    def forward_head(self, {a}, pre_logits: bool = False):
{_ind(head, 8)}

    def forward(self, {a}):
        {a} = self.forward_features({a})
        {a} = self.forward_head({a})
        return {a}
'''
    else:
        methods = f'''
    def forward{s.sig(f"self, {a}")}:
{_ind(fwd + head, 8)}
'''
    p.add(cls, f'''
class {cls}(nn.Module):
    def __init__(self, {", ".join(params)}):
        {s.super_line(cls)}
{_ind(inits, 8)}
{methods}{extra_methods}''')
    # how the instance is made: a builder with named configurations, or the class directly
    def fmt(rs):
        return "[" + ", ".join("(" + ", ".join(repr(x) for x in r) + ("," if len(r) == 1 else "") + ")"
                               for r in rs) + "]"

    def deeper(k, rs):                           # a deeper configuration: one more block per stage
        if "n" not in k.cols:
            return rs
        j = k.cols.index("n")
        return [tuple(x + 1 if c == j else x for c, x in enumerate(r)) for r in rs]

    tables = [(kind, rows)] + ([(kind2, rows2)] if rows2 is not None else [])
    if s.builder:
        fn = rng.choice(["build_model", "create_net", f"{cls.lower()}_small", "make_model", f"{cls.lower()}_tiny"])
        if rows is not None:
            pack = (lambda ts: fmt(ts[0])) if rows2 is None else (lambda ts: "(" + ", ".join(fmt(t) for t in ts) + ")")
            variants = {"small": pack([r for _, r in tables])}
            if any("n" in k.cols for k, _ in tables) and rng.random() < 0.6:
                variants["base"] = pack([deeper(k, r) for k, r in tables])
            cfgs = "{" + ", ".join(f'"{k}": {t}' for k, t in variants.items()) + "}"
            star = "*" if rows2 is not None else ""
            first = "".join(f"{b}, " for b in lead)
            if rng.random() < 0.5:
                # torchvision's _resnet(block, layers, weights, progress) / timm's _create_x(variant)
                inner = f"_{cls.lower()}"
                p.add(inner, f'''
def {inner}(variant, weights=None, progress=True, **kwargs):
    cfgs = {cfgs}
    model = {cls}({first}{star}cfgs[variant], **kwargs)
    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))
    return model
''')
                lines = [f"return {inner}({list(variants)[-1]!r}, weights, progress, **kwargs)"]
                sig = "weights=None, progress=True, **kwargs"
            else:
                lines = [f"cfgs = {cfgs}", f"return {cls}({first}{star}cfgs[variant], **kwargs)"]
                sig = 'variant="small", **kwargs'
        else:
            lines = [f"return {cls}(**kwargs)"]
            sig = "**kwargs"
        p.add(fn, f'''
def {fn}({sig}):
{_ind(lines, 4)}
''')
        ctor = f"{fn}(num_classes={rng.choice([10, 100])})"
    elif rows is not None:
        names = rng.choice([("ARCH", "ARCH_LATE"), ("CFG", "CFG_B"), ("STAGES", "LATE_STAGES"),
                            ("DEFAULT_CFG", "DEFAULT_CFG_2")])
        consts = []
        for (_, r), const in zip(tables, names):
            p.add(const, f"{const} = {fmt(r)}")
            consts.append(const)
        ctor = f"{cls}({', '.join(lead + consts)})"
    else:
        ctor = f"{cls}()"
    # ConvNeXt: a stride-4 stem and up to five stride-2 downsamplings -> at least 128 pixels
    size = 128 if isinstance(kind, ConvNeXtKind) else 64 if rows2 is not None else rng.choice([32, 64])
    return ctor, f"torch.randn(1, 3, {size}, {size})"


# --------------------------------------------------------------------------
# the ViT-style network
# --------------------------------------------------------------------------
def vit_network(rng: random.Random, p: V3Program):
    s: V3Style = p.style
    a = s.arg
    dp = drop_path_def(p) if s.drop_path else None
    explicit = rng.random() < 0.7
    attn_cls = rng.choice(["Attention", "SelfAttention", "MultiHeadAttention"])
    mlp_cls = rng.choice(["Mlp", "MLP", "FeedForward"])
    blk_cls = rng.choice(["Block", "TransformerBlock", "EncoderBlock"])
    pe_cls = rng.choice(["PatchEmbed", "PatchEmbedding", "Stem"])
    net_cls = rng.choice(["VisionTransformer", "ViT", "TinyViT", "PatchTransformer"])
    if explicit:
        sdpa = rng.random() < 0.4
        core = (["x = F.scaled_dot_product_attention(q, k, v)"] if sdpa else
                ["q = q * self.scale", "attn = q @ k.transpose(-2, -1)", "attn = attn.softmax(dim=-1)",
                 "attn = self.attn_drop(attn)", "x = attn @ v"])
        p.add(attn_cls, f'''
class {attn_cls}(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        {s.super_line(attn_cls)}
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
{_ind(core, 8)}
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
''')
    use_ops_mlp = s.ops_imports and rng.random() < 0.5
    mlp_defaults = _ind(["if out_features is None:", "    out_features = in_features",
                         "if hidden_features is None:", "    hidden_features = in_features"] if s.none_checks else
                        ["out_features = out_features or in_features",
                         "hidden_features = hidden_features or in_features"], 8)
    if use_ops_mlp:
        p.need("from torchvision.ops import MLP")
        mlp_expr = "MLP(dim, [hidden, dim], activation_layer=nn.GELU, dropout=drop)"
    else:
        p.add(mlp_cls, f'''
class {mlp_cls}(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        {s.super_line(mlp_cls)}
{mlp_defaults}
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
''')
        mlp_expr = f"{mlp_cls}(dim, hidden, drop=drop)"
    ls = rng.random() < 0.5
    if ls:
        p.add("LayerScale", f'''
class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5):
        {s.super_line("LayerScale")}
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x * self.gamma
''')
    attn_expr = (f"{attn_cls}(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)"
                 if explicit else "nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, batch_first=True)")
    attn_call = "self.attn(self.norm1(x))" if explicit else "self.attn(h, h, h, need_weights=False)[0]"
    pre = [] if explicit else ["h = self.norm1(x)"]
    ls_init = ["self.ls1 = LayerScale(dim, init_values) if init_values else nn.Identity()",
               "self.ls2 = LayerScale(dim, init_values) if init_values else nn.Identity()"] if ls else []
    dp_init = [f"self.drop_path1 = {dp_expr(dp, 'drop_path')}", f"self.drop_path2 = {dp_expr(dp, 'drop_path')}"] if dp else []
    wrap1 = (lambda e: f"self.drop_path1({e})") if dp else (lambda e: e)
    wrap2 = (lambda e: f"self.drop_path2({e})") if dp else (lambda e: e)
    sc1 = (lambda e: f"self.ls1({e})") if ls else (lambda e: e)
    sc2 = (lambda e: f"self.ls2({e})") if ls else (lambda e: e)
    use_partial = rng.random() < 0.6
    if use_partial:
        p.need("from functools import partial")
    nl = (lambda d: f"norm_layer({d})") if use_partial else (lambda d: f"nn.LayerNorm({d}, eps=1e-6)")
    blk_params = "dim, num_heads, mlp_ratio=4.0, qkv_bias=False, drop=0.0, attn_drop=0.0" + \
                 (", init_values=None" if ls else "") + (", drop_path=0.0" if dp else "") + \
                 (", norm_layer=nn.LayerNorm" if use_partial else "")
    p.add(blk_cls, f'''
class {blk_cls}(nn.Module):
    def __init__(self, {blk_params}):
        {s.super_line(blk_cls)}
        hidden = int(dim * mlp_ratio)
        self.norm1 = {nl("dim")}
        self.attn = {attn_expr}
{_ind(ls_init[:1], 8)}
        self.norm2 = {nl("dim")}
        self.mlp = {mlp_expr}
{_ind(ls_init[1:], 8)}
{_ind(dp_init, 8)}

    def forward(self, x):
{_ind(pre, 8)}
        x = x + {wrap1(sc1(attn_call))}
        x = x + {wrap2(sc2("self.mlp(self.norm2(x))"))}
        return x
''')
    if rng.random() < 0.5:
        return _staged_vit(rng, p, blk_cls, net_cls)
    norm_pe = rng.random() < 0.4
    p.add(pe_cls, f'''
class {pe_cls}(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=64, norm_layer=None):
        {s.super_line(pe_cls)}
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer{" is not None" if s.none_checks else ""} else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x
''')
    cls_tok = rng.random() < 0.7
    pool = "x[:, 0]" if cls_tok and rng.random() < 0.7 else "x.mean(dim=1)"
    depth = rng.randint(2, 6)
    dim = rng.choice([32, 48, 64])
    heads = rng.choice([2, 4])
    blocks_line = (f"self.blocks = {'nn.ModuleList([' if s.block_list else 'nn.Sequential(*['}{blk_cls}(embed_dim, num_heads, mlp_ratio, qkv_bias=qkv_bias, "
                   f"drop=drop_rate{', init_values=init_values' if ls else ''}{', drop_path=dpr[i]' if dp else ''}"
                   f"{', norm_layer=norm_layer' if use_partial else ''}) "
                   f"for i in range(depth)])")
    rep_size = s.od_head and rng.random() < 0.5
    inits = [f"self.patch_embed = {pe_cls}(img_size, patch_size, in_chans, embed_dim"
             f"{', norm_layer=nn.LayerNorm' if norm_pe else ''})",
             "num_patches = self.patch_embed.num_patches"]
    if cls_tok:
        inits += ["self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))",
                  "self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))"]
    else:
        inits += ["self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))"]
    inits += ["self.pos_drop = nn.Dropout(p=drop_rate)"]
    if dp:
        inits.append("dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]")
    if use_partial:
        inits = (["if norm_layer is None:", "    norm_layer = partial(nn.LayerNorm, eps=1e-6)"] if s.none_checks
                 else ["norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)"]) + inits
    gp = rng.random() < 0.6
    if gp:
        inits = ['assert global_pool in ("token", "avg")', "self.global_pool = global_pool"] + inits
    inits += [blocks_line]
    if s.feature_info:
        inits.append("self.feature_info = [dict(module=f\"blocks.{i}\", num_chs=embed_dim, reduction=patch_size) "
                     "for i in range(depth)]")
    if s.grad_ckpt and not s.block_list:
        p.need("from torch.utils.checkpoint import checkpoint_sequential")
        inits.append("self.grad_checkpointing = False")
    inits += [f"self.norm = {nl('embed_dim')}", "self.head_drop = nn.Dropout(drop_rate)"]
    if s.od_head:
        # torchvision's VisionTransformer: heads in an OrderedDict, a pre-logits layer if configured
        p.need("from collections import OrderedDict")
        inits += ["heads = OrderedDict()",
                  "if representation_size is None:",
                  '    heads["head"] = nn.Linear(embed_dim, num_classes)',
                  "else:",
                  '    heads["pre_logits"] = nn.Linear(embed_dim, representation_size)',
                  '    heads["act"] = nn.Tanh()',
                  '    heads["head"] = nn.Linear(representation_size, num_classes)',
                  "self.head = nn.Sequential(heads)"]
    else:
        inits.append("self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()")
    feats = ["x = self.patch_embed(x)"]
    if cls_tok:
        feats.append(rng.choice(["x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)",
                                 "cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)\n"
                                 "x = torch.cat([cls_tokens, x], dim=1)"]))
    if s.block_list:
        run = ["for blk in self.blocks:", "    x = blk(x)"]
    elif s.grad_ckpt:
        run = ["if self.grad_checkpointing and not torch.jit.is_scripting():",
               "    x = checkpoint_sequential(self.blocks, len(self.blocks), x)", "else:", "    x = self.blocks(x)"]
    else:
        run = ["x = self.blocks(x)"]
    feats += ["x = x + self.pos_embed", "x = self.pos_drop(x)"] + run + ["x = self.norm(x)", "return x"]
    feats = "\n".join(feats).split("\n")
    params = ("img_size=32, patch_size=4, in_chans=3, num_classes=10, embed_dim=64, depth=4, num_heads=4, "
              "mlp_ratio=4.0, qkv_bias=True, drop_rate=0.0" + (", init_values=None" if ls else "") +
              (", drop_path_rate=0.1" if dp else "") +
              (f", global_pool={'token' if cls_tok else 'avg'!r}" if gp else "") +
              (", norm_layer=None" if use_partial else "") + (", representation_size=None" if s.od_head else "") +
              (", **kwargs" if s.kwargs_net else ""))
    asrt = (['torch._assert(x.shape[1] == 3, "expected 3 input channels")'] if s.assert_input else [])
    head_pool = (f'x[:, 0] if self.global_pool == "token" else x.mean(dim=1)' if gp and cls_tok else
                 "x.mean(dim=1)" if gp else pool)
    p.add(net_cls, f'''
class {net_cls}(nn.Module):
    def __init__(self, {params}):
        {s.super_line(net_cls)}
        self.num_classes = num_classes
        self.embed_dim = embed_dim
{_ind(inits, 8)}

    def forward_features(self, x):
{_ind(asrt + feats, 8)}

    def forward_head(self, x, pre_logits: bool = False):
        x = {head_pool}
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x
''')
    size, patch = rng.choice([(32, 4), (32, 8), (64, 8)])
    args = f"img_size={size}, patch_size={patch}, embed_dim={dim}, depth={depth}, num_heads={heads}" + \
           (", init_values=1e-5" if ls and rng.random() < 0.7 else "") + \
           (", representation_size=32" if rep_size else "")
    if s.builder:
        fn = rng.choice(["vit_tiny", "create_vit", "build_vit", "vit_small"])
        p.add(fn, f'''
def {fn}(**kwargs):
    model_args = dict({args})
    return {net_cls}(**dict(model_args, **kwargs))
''')
        ctor = f"{fn}(num_classes={rng.choice([10, 100])})"
    else:
        ctor = f"{net_cls}({args})"
    return ctor, f"torch.randn(1, 3, {size}, {size})"


def _staged_vit(rng: random.Random, p: V3Program, blk_cls: str, net_cls: str):
    s: V3Style = p.style
    stage_cls = rng.choice(["TransformerStage", "Stage", "ViTStage", "EncoderStage"])
    down = rng.choice(["conv", "conv_norm"])
    down_init = ("nn.Conv2d(in_dim, dim, kernel_size=3, stride=2, padding=1)" if down == "conv" else
                 "nn.Sequential(nn.Conv2d(in_dim, dim, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(dim))")
    run = ["for blk in self.blocks:", "    x = blk(x)"] if s.block_list else ["x = self.blocks(x)"]
    blocks = ("nn.ModuleList([" if s.block_list else "nn.Sequential(*[") + \
        f"{blk_cls}(dim, num_heads, mlp_ratio, qkv_bias=qkv_bias, drop=drop) for _ in range(depth)])"
    p.add(stage_cls, f'''
class {stage_cls}(nn.Module):
    def __init__(self, in_dim, dim, depth, num_heads, mlp_ratio=4.0, qkv_bias=False, drop=0.0, downsample=True):
        {s.super_line(stage_cls)}
        if downsample:
            self.downsample = {down_init}
        else:
            assert in_dim == dim
            self.downsample = nn.Identity()
        self.blocks = {blocks}
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x):
        x = self.downsample(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
{_ind(run, 8)}
        x = self.norm(x)
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return x
''')
    n = rng.randint(3, 4)
    dims = rng.choice([(16, 32, 64, 128), (32, 64, 128, 256), (24, 48, 96, 192)])[:n]
    heads = tuple(max(1, d // rng.choice([16, 32])) for d in dims)
    depths = tuple(rng.randint(1, 3) for _ in range(n))
    stage_loop = ["stages = []", "in_dim = embed_dims[0]",
                  "for i, (dim, depth, heads) in enumerate(zip(embed_dims, depths, num_heads)):",
                  f"    stages.append({stage_cls}(in_dim, dim, depth, heads, mlp_ratio, qkv_bias=qkv_bias, "
                  "drop=drop_rate, downsample=i > 0))",
                  "    in_dim = dim"]
    if s.feature_info:
        stage_loop.insert(0, "self.feature_info = []")
        stage_loop.append('    self.feature_info.append(dict(num_chs=dim, module=f"stages.{i}"))')
    if s.container == "modulelist":
        stage_loop.append("self.stages = nn.ModuleList(stages)")
        st_fwd = ["for stage in self.stages:", "    x = stage(x)"]
    else:
        stage_loop.append("self.stages = nn.Sequential(*stages)")
        st_fwd = ["x = self.stages(x)"]
    head = ["self.num_features = embed_dims[-1]", "self.head_drop = nn.Dropout(drop_rate)"]
    if s.od_head:
        p.need("from collections import OrderedDict")
        head += ["self.head = nn.Sequential(OrderedDict([",
                 '    ("flatten", nn.Flatten(1)),',
                 '    ("fc", nn.Linear(self.num_features, num_classes)),',
                 "]))"]
    else:
        head += ["self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()"]
    stem = rng.choice(["patch", "conv2"])
    stem_init = (["self.stem = nn.Conv2d(in_chans, embed_dims[0], kernel_size=patch_size, stride=patch_size)"]
                 if stem == "patch" else
                 ["self.stem = nn.Sequential(",
                  "    nn.Conv2d(in_chans, embed_dims[0] // 2, kernel_size=3, stride=2, padding=1),",
                  "    nn.BatchNorm2d(embed_dims[0] // 2),",
                  "    nn.GELU(),",
                  "    nn.Conv2d(embed_dims[0] // 2, embed_dims[0], kernel_size=3, stride=2, padding=1),",
                  ")"])
    params = ("img_size=32, patch_size=4, in_chans=3, num_classes=10, embed_dims=(32, 64, 128), "
              "depths=(2, 2, 2), num_heads=(1, 2, 4), mlp_ratio=4.0, qkv_bias=True, drop_rate=0.0" +
              (", **kwargs" if s.kwargs_net else ""))
    asrt = ['torch._assert(x.shape[1] == 3, "expected 3 input channels")'] if s.assert_input else []
    p.add(net_cls, f'''
class {net_cls}(nn.Module):
    def __init__(self, {params}):
        {s.super_line(net_cls)}
        assert len(embed_dims) == len(depths) == len(num_heads)
        self.num_classes = num_classes
{_ind(stem_init + stage_loop + head, 8)}

    def forward_features(self, x):
{_ind(asrt + ["x = self.stem(x)"] + st_fwd + ["return x"], 8)}

    def forward_head(self, x, pre_logits: bool = False):
        x = x.mean(dim=(2, 3))
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x
''')
    size = 64 if n == 4 else rng.choice([32, 64])
    args = f"embed_dims={dims}, depths={depths}, num_heads={heads}, mlp_ratio={rng.choice([2.0, 4.0])}"
    if s.builder:
        fn = rng.choice(["vit_staged_tiny", "create_model", "staged_vit_small", "build_hvit"])
        p.add(fn, f'''
def {fn}(**kwargs):
    model_args = dict({args})
    return {net_cls}(**dict(model_args, **kwargs))
''')
        ctor = f"{fn}(num_classes={rng.choice([10, 100])})"
    else:
        ctor = f"{net_cls}({args})"
    return ctor, f"torch.randn(1, 3, {size}, {size})"


# --------------------------------------------------------------------------
# the VGG-style network: a width list turned into layers by a function
# --------------------------------------------------------------------------
def vgg_network(rng: random.Random, p: V3Program):
    s: V3Style = p.style
    a = s.arg
    fn = rng.choice(["make_layers", "_make_features", "build_layers"])
    p.add(fn, f'''
def {fn}(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), {s.act_ctor()}]
            else:
                layers += [conv2d, {s.act_ctor()}]
            in_channels = v
    return nn.Sequential(*layers)
''')
    widths = [8, 16, 32, 32]
    cfg = []
    for i, w in enumerate(widths[:rng.randint(3, 4)]):
        cfg += [w] * rng.randint(1, 2) + ["M"]
    last = [v for v in cfg if v != "M"][-1]
    pool = rng.choice([1, 2])
    hidden = rng.choice([32, 64])
    cls = rng.choice(["VGG", "PlainNet", "VGGLike", "ConvClassifier"])
    init_w = rng.random() < 0.5
    init_lines = ['if init_weights:', '    for m in self.modules():', '        if isinstance(m, nn.Conv2d):',
                  '            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")',
                  '        elif isinstance(m, nn.Linear):', '            nn.init.normal_(m.weight, 0, 0.01)'] if init_w else []
    p.add(cls, f'''
class {cls}(nn.Module):
    def __init__(self, features, num_classes=10, init_weights=True, dropout=0.5):
        {s.super_line(cls)}
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d(({pool}, {pool}))
        self.classifier = nn.Sequential(
            nn.Linear({last} * {pool} * {pool}, {hidden}),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear({hidden}, {hidden}),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear({hidden}, num_classes),
        )
{_ind(init_lines, 8)}

    def forward{s.sig(f"self, {a}")}:
        {a} = self.features({a})
        {a} = self.avgpool({a})
        {a} = torch.flatten({a}, 1)
        {a} = self.classifier({a})
        return {a}
''')
    cfgs = {"A": cfg, "B": [x for v in cfg for x in ([v, v] if v != "M" and rng.random() < 0.3 else [v])]}
    p.add("cfgs", "cfgs = {" + ", ".join(f'"{k}": {v!r}' for k, v in cfgs.items()) + "}")
    bn = rng.random() < 0.5
    p.add("_" + cls.lower(), f'''
def _{cls.lower()}(cfg, batch_norm, **kwargs):
    return {cls}({fn}(cfgs[cfg], batch_norm=batch_norm), **kwargs)
''')
    name = rng.choice(["vgg_small", f"{cls.lower()}_a", "plain_tiny"])
    p.add(name, f'''
def {name}(**kwargs):
    return _{cls.lower()}("{rng.choice(['A', 'B'])}", {bn}, **kwargs)
''')
    size = rng.choice([32, 64])
    return f"{name}(num_classes={rng.choice([10, 100])})", f"torch.randn(1, 3, {size}, {size})"


# --------------------------------------------------------------------------
# the MLP-Mixer-style network: token mixing through transpose
# --------------------------------------------------------------------------
def mixer_network(rng: random.Random, p: V3Program):
    s: V3Style = p.style
    dp = drop_path_def(p) if s.drop_path else None
    mlp = rng.choice(["Mlp", "MlpBlock", "FeedForward"])
    blk = rng.choice(["MixerBlock", "MixerLayer", "TokenMixingBlock"])
    net = rng.choice(["MlpMixer", "Mixer", "TinyMixer"])
    p.add(mlp, f'''
class {mlp}(nn.Module):
    def __init__(self, in_features, hidden_features, act_layer={"None" if s.none_checks else "nn.GELU"}, drop=0.0):
        {s.super_line(mlp)}
{_ind(["if act_layer is None:", "    act_layer = nn.GELU"] if s.none_checks else [], 8)}
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
''')
    dp_init = f"self.drop_path = {dp_expr(dp, 'drop_path')}" if dp else ""
    w = (lambda e: f"self.drop_path({e})") if dp else (lambda e: e)
    tok_style = rng.random() < 0.5
    tok = (w("self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2)") if tok_style else None)
    fwd = ([f"x = x + {tok}"] if tok_style else
           ["y = self.norm1(x).transpose(1, 2)", "y = self.mlp_tokens(y).transpose(1, 2)", f"x = x + {w('y')}"])
    fwd += [f"x = x + {w('self.mlp_channels(self.norm2(x))')}", "return x"]
    p.add(blk, f'''
class {blk}(nn.Module):
    def __init__(self, dim, seq_len, mlp_ratio=(0.5, 4.0), drop=0.0{", drop_path=0.0" if dp else ""}):
        {s.super_line(blk)}
        tokens_dim, channels_dim = [int(r * dim) for r in mlp_ratio]
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp_tokens = {mlp}(seq_len, tokens_dim, drop=drop)
        {dp_init}
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp_channels = {mlp}(dim, channels_dim, drop=drop)

    def forward(self, x):
{_ind(fwd, 8)}
''')
    dpr = "dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_blocks)]" if dp else ""
    p.add(net, f'''
class {net}(nn.Module):
    def __init__(self, num_classes=10, img_size=32, in_chans=3, patch_size=4, num_blocks=4, embed_dim=64,
                 mlp_ratio=(0.5, 4.0), drop_rate=0.0, drop_path_rate=0.0{", **kwargs" if s.kwargs_net else ""}):
        {s.super_line(net)}
        self.num_classes = num_classes
        self.stem = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (img_size // patch_size) ** 2
        {dpr}
        self.blocks = {"nn.ModuleList([" if s.block_list else "nn.Sequential(*["}
            {blk}(embed_dim, num_patches, mlp_ratio, drop=drop_rate{", drop_path=dpr[i]" if dp else ""})
            for i in range(num_blocks)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.stem(x)
        x = x.flatten(2).transpose(1, 2)
{_ind(["for blk in self.blocks:", "    x = blk(x)"] if s.block_list else ["x = self.blocks(x)"], 8)}
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        x = x.mean(dim=1)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x
''')
    size, patch = rng.choice([(32, 4), (32, 8), (64, 8)])
    args = f"img_size={size}, patch_size={patch}, num_blocks={rng.randint(2, 6)}, embed_dim={rng.choice([32, 48, 64])}"
    if dp:
        args += ", drop_path_rate=0.1"
    if s.builder:
        fn = rng.choice(["mixer_tiny", "create_mixer", "mixer_small"])
        p.add(fn, f'''
def {fn}(**kwargs):
    model_args = dict({args})
    return {net}(**dict(model_args, **kwargs))
''')
        return f"{fn}(num_classes={rng.choice([10, 100])})", f"torch.randn(1, 3, {size}, {size})"
    return f"{net}({args})", f"torch.randn(1, 3, {size}, {size})"


NETWORK_ONLY = {"vit": vit_network, "vgg": vgg_network, "mixer": mixer_network}


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------
def generate_v3_program(rng: random.Random, kind: str = "network", family: str | None = None) -> str:
    """One program: kind "network" (a whole network) or "block" (one block, v3 style)."""
    family = family or rng.choice(KINDS)
    style = V3Style(rng)
    p = V3Program(style)
    if kind == "network":
        ctor, inp = NETWORK_ONLY[family](rng, p) if family in NETWORK_ONLY else cnn_network(rng, p, family)
    elif family in NETWORK_ONLY:
        ctor, inp = NETWORK_ONLY[family](rng, p)   # these blocks only make sense inside their network
    else:
        k = CONV_KINDS[family](p, rng)
        ctor, cin = k.block_ctor()
        inp = f"torch.randn(1, {cin}, 16, 16)"
    return p.render(ctor, inp)
