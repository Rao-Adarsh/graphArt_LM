"""Build the real-code evaluation set from torchvision's own source (RQ4).

    python scripts/build_torchvision_set.py

Writes data/generated/torchvision.jsonl in the same shape as the test split, so the
whole pipeline runs on it unchanged (`run_prompting --split torchvision`).

Each sample is torchvision's real source, pulled with `inspect`: the block's class,
the helper classes and functions it uses, and the import lines it needs, followed by
one line that builds it and an example input. Nothing is rewritten. The ground truth
comes from tracing that source, and every sample is checked against tracing the
genuine torchvision object with the same arguments -- so the assembled code is known
to define exactly the graph torchvision does.

Two kinds of sample, so style and size can be told apart:
  block    real building blocks; graphs about the size of the synthetic test set
  network  two whole small networks (~70 nodes), the length stress test

    python scripts/build_torchvision_set.py --set torchvision_holdout

builds the held-out set instead: other model families (MobileNetV3, EfficientNet,
ConvNeXt, RegNet, DenseNet, VGG, DeepLabV3, video ResNets, S3D, RAFT), frozen before
the generator was changed using the first 21 samples' errors. The generator is never
fitted to it, and it is scored once, at the end -- so it stays a fair test of code
the project has not looked at. Its sha256 is recorded in data/generated/FROZEN.json.
"""

import argparse
import hashlib
import importlib
import inspect
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import networkx as nx
import torch
from torch import nn

# By full path: `torchvision.models.googlenet` is also the name of a builder function,
# which shadows the module on attribute access.
(alexnet, googlenet, inception, mnasnet, mobilenetv2, resnet, shufflenetv2, squeezenet,
 vision_transformer) = (
    importlib.import_module(f"torchvision.models.{m}")
    for m in ("alexnet", "googlenet", "inception", "mnasnet", "mobilenetv2", "resnet", "shufflenetv2",
              "squeezenet", "vision_transformer"))
(convnext, deeplabv3, densenet, efficientnet, mobilenetv3, raft, regnet, s3d, vgg, video_resnet) = (
    importlib.import_module(f"torchvision.models.{m}")
    for m in ("convnext", "segmentation.deeplabv3", "densenet", "efficientnet", "mobilenetv3",
              "optical_flow.raft", "regnet", "video.s3d", "vgg", "video.resnet"))

from graphart.extract import graph_from_code, graph_from_module
from graphart.schema import to_text
from graphart.validate import check, to_networkx

DATA = ROOT / "data" / "generated"
FROZEN = DATA / "FROZEN.json"

BASE_IMPORTS = """\
from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
"""

# (id, kind, torchvision module, objects whose source is shown, extra imports,
#  construction expression, input shape, the same construction on the real module)
ENTRIES = [
    ("tv-resnet-basicblock", "block", resnet, ["conv3x3", "conv1x1", "BasicBlock"], "",
     "BasicBlock(64, 64)", [1, 64, 56, 56], lambda: resnet.BasicBlock(64, 64)),
    ("tv-resnet-basicblock-down", "block", resnet, ["conv3x3", "conv1x1", "BasicBlock"], "",
     "BasicBlock(64, 128, stride=2, downsample=nn.Sequential(conv1x1(64, 128, 2), nn.BatchNorm2d(128)))",
     [1, 64, 56, 56],
     lambda: resnet.BasicBlock(64, 128, stride=2, downsample=nn.Sequential(resnet.conv1x1(64, 128, 2), nn.BatchNorm2d(128)))),
    ("tv-resnet-bottleneck", "block", resnet, ["conv3x3", "conv1x1", "Bottleneck"], "",
     "Bottleneck(256, 64)", [1, 256, 56, 56], lambda: resnet.Bottleneck(256, 64)),
    ("tv-squeezenet-fire", "block", squeezenet, ["Fire"], "",
     "Fire(96, 16, 64, 64)", [1, 96, 55, 55], lambda: squeezenet.Fire(96, 16, 64, 64)),
    ("tv-googlenet-inception", "block", googlenet, ["BasicConv2d", "Inception"], "",
     "Inception(192, 64, 96, 128, 16, 32, 32)", [1, 192, 28, 28],
     lambda: googlenet.Inception(192, 64, 96, 128, 16, 32, 32)),
    ("tv-inceptionv3-a", "block", inception, ["BasicConv2d", "InceptionA"], "",
     "InceptionA(192, pool_features=32)", [1, 192, 35, 35], lambda: inception.InceptionA(192, pool_features=32)),
    ("tv-inceptionv3-b", "block", inception, ["BasicConv2d", "InceptionB"], "",
     "InceptionB(288)", [1, 288, 35, 35], lambda: inception.InceptionB(288)),
    ("tv-inceptionv3-c", "block", inception, ["BasicConv2d", "InceptionC"], "",
     "InceptionC(768, channels_7x7=128)", [1, 768, 17, 17], lambda: inception.InceptionC(768, channels_7x7=128)),
    ("tv-shufflenet-unit", "block", shufflenetv2, ["channel_shuffle", "InvertedResidual"], "",
     "InvertedResidual(116, 116, 1)", [1, 116, 28, 28], lambda: shufflenetv2.InvertedResidual(116, 116, 1)),
    ("tv-shufflenet-unit-down", "block", shufflenetv2, ["channel_shuffle", "InvertedResidual"], "",
     "InvertedResidual(24, 116, 2)", [1, 24, 56, 56], lambda: shufflenetv2.InvertedResidual(24, 116, 2)),
    ("tv-vit-encoderblock", "block", vision_transformer, ["MLPBlock", "EncoderBlock"],
     "from functools import partial\nfrom torchvision.ops.misc import MLP\n",
     "EncoderBlock(num_heads=4, hidden_dim=64, mlp_dim=128, dropout=0.0, attention_dropout=0.0)",
     [1, 10, 64],
     lambda: vision_transformer.EncoderBlock(num_heads=4, hidden_dim=64, mlp_dim=128, dropout=0.0, attention_dropout=0.0)),
    ("tv-inceptionv3-d", "block", inception, ["BasicConv2d", "InceptionD"], "",
     "InceptionD(768)", [1, 768, 17, 17], lambda: inception.InceptionD(768)),
    ("tv-inceptionv3-e", "block", inception, ["BasicConv2d", "InceptionE"], "",
     "InceptionE(1280)", [1, 1280, 8, 8], lambda: inception.InceptionE(1280)),
    ("tv-inceptionv3-aux", "block", inception, ["BasicConv2d", "InceptionAux"], "",
     "InceptionAux(768, num_classes=10)", [1, 768, 17, 17], lambda: inception.InceptionAux(768, num_classes=10)),
    ("tv-googlenet-aux", "block", googlenet, ["BasicConv2d", "InceptionAux"], "",
     "InceptionAux(512, num_classes=10)", [1, 512, 14, 14], lambda: googlenet.InceptionAux(512, num_classes=10)),
    ("tv-mnasnet-residual", "block", mnasnet, ["_InvertedResidual"], "",
     "_InvertedResidual(24, 24, 3, 1, 3)", [1, 24, 56, 56], lambda: mnasnet._InvertedResidual(24, 24, 3, 1, 3)),
    ("tv-mnasnet-down", "block", mnasnet, ["_InvertedResidual"], "",
     "_InvertedResidual(24, 40, 5, 2, 3)", [1, 24, 56, 56], lambda: mnasnet._InvertedResidual(24, 40, 5, 2, 3)),
    ("tv-mobilenetv2-residual", "block", mobilenetv2, ["InvertedResidual"],
     "from torchvision.ops.misc import Conv2dNormActivation\n",
     "InvertedResidual(32, 32, 1, 6)", [1, 32, 28, 28], lambda: mobilenetv2.InvertedResidual(32, 32, 1, 6)),
    ("tv-alexnet", "network", alexnet, ["AlexNet"],
     "from torchvision.utils import _log_api_usage_once\n",
     "AlexNet(num_classes=1000)", [1, 3, 224, 224], lambda: alexnet.AlexNet(num_classes=1000)),
    ("tv-squeezenet1_1", "network", squeezenet, ["Fire", "SqueezeNet"],
     "import torch.nn.init as init\nfrom torchvision.utils import _log_api_usage_once\n",
     'SqueezeNet(version="1_1", num_classes=1000)', [1, 3, 224, 224],
     lambda: squeezenet.SqueezeNet(version="1_1", num_classes=1000)),
    # ResNet's type hints name Bottleneck, so its (unused here) source is shown too, as
    # it is in the real file.
    ("tv-resnet18", "network", resnet, ["conv3x3", "conv1x1", "BasicBlock", "Bottleneck", "ResNet"],
     "from torchvision.utils import _log_api_usage_once\n",
     "ResNet(BasicBlock, [2, 2, 2, 2])", [1, 3, 224, 224], lambda: resnet.ResNet(resnet.BasicBlock, [2, 2, 2, 2])),
]

# The held-out set: no family above appears here. DenseNet was tried and left out: its
# source carries torch.jit._overload_method stubs, which do not run once assembled.
MISC = "from torchvision.ops.misc import Conv2dNormActivation, Conv3dNormActivation, MLP, Permute, SqueezeExcitation\n"
SE = "from torchvision.ops.misc import SqueezeExcitation as SElayer\n"
PARTIAL = "from functools import partial\n"
DIVISIBLE = "from torchvision.models._utils import _make_divisible\n"
DEPTH = "from torchvision.ops import StochasticDepth\n"
LOG = "from torchvision.utils import _log_api_usage_once\n"
V3_NORM = "partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)"
V3_NORM_REAL = lambda: __import__("functools").partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)
S3D_NORM = "partial(nn.BatchNorm3d, eps=0.001, momentum=0.001)"
S3D_NORM_REAL = lambda: __import__("functools").partial(nn.BatchNorm3d, eps=0.001, momentum=0.001)
HOLDOUT = [
    ("tvh-mobilenetv3-se", "block", mobilenetv3, ["InvertedResidualConfig", "InvertedResidual"],
     PARTIAL + MISC + SE + DIVISIBLE,
     f'InvertedResidual(InvertedResidualConfig(40, 5, 120, 40, True, "RE", 1, 1, 1.0), {V3_NORM})',
     [1, 40, 28, 28],
     lambda: mobilenetv3.InvertedResidual(mobilenetv3.InvertedResidualConfig(40, 5, 120, 40, True, "RE", 1, 1, 1.0),
                                          V3_NORM_REAL())),
    ("tvh-mobilenetv3-hs-down", "block", mobilenetv3, ["InvertedResidualConfig", "InvertedResidual"],
     PARTIAL + MISC + SE + DIVISIBLE,
     f'InvertedResidual(InvertedResidualConfig(40, 3, 240, 80, False, "HS", 2, 1, 1.0), {V3_NORM})',
     [1, 40, 28, 28],
     lambda: mobilenetv3.InvertedResidual(mobilenetv3.InvertedResidualConfig(40, 3, 240, 80, False, "HS", 2, 1, 1.0),
                                          V3_NORM_REAL())),
    ("tvh-efficientnet-mbconv", "block", efficientnet, ["_MBConvConfig", "MBConvConfig", "MBConv"],
     "import math\nfrom dataclasses import dataclass\n" + PARTIAL + MISC + DEPTH + DIVISIBLE,
     "MBConv(MBConvConfig(6, 3, 1, 40, 40, 1), 0.0, nn.BatchNorm2d)", [1, 40, 28, 28],
     lambda: efficientnet.MBConv(efficientnet.MBConvConfig(6, 3, 1, 40, 40, 1), 0.0, nn.BatchNorm2d)),
    ("tvh-efficientnet-mbconv-down", "block", efficientnet, ["_MBConvConfig", "MBConvConfig", "MBConv"],
     "import math\nfrom dataclasses import dataclass\n" + PARTIAL + MISC + DEPTH + DIVISIBLE,
     "MBConv(MBConvConfig(6, 5, 2, 40, 80, 1), 0.0, nn.BatchNorm2d)", [1, 40, 28, 28],
     lambda: efficientnet.MBConv(efficientnet.MBConvConfig(6, 5, 2, 40, 80, 1), 0.0, nn.BatchNorm2d)),
    ("tvh-efficientnet-fused", "block", efficientnet, ["_MBConvConfig", "FusedMBConvConfig", "FusedMBConv"],
     "import math\nfrom dataclasses import dataclass\n" + PARTIAL + MISC + DEPTH + DIVISIBLE,
     "FusedMBConv(FusedMBConvConfig(4, 3, 1, 48, 48, 1), 0.0, nn.BatchNorm2d)", [1, 48, 28, 28],
     lambda: efficientnet.FusedMBConv(efficientnet.FusedMBConvConfig(4, 3, 1, 48, 48, 1), 0.0, nn.BatchNorm2d)),
    ("tvh-convnext-block", "block", convnext, ["CNBlock"], PARTIAL + MISC + DEPTH,
     "CNBlock(96, 1e-6, 0.0)", [1, 96, 14, 14], lambda: convnext.CNBlock(96, 1e-6, 0.0)),
    ("tvh-regnet-bottleneck", "block", regnet, ["BottleneckTransform", "ResBottleneckBlock"],
     "import math\nfrom collections import OrderedDict\n" + PARTIAL + MISC + DIVISIBLE,
     "ResBottleneckBlock(48, 104, 2, nn.BatchNorm2d, nn.ReLU, group_width=8, bottleneck_multiplier=1.0, se_ratio=0.25)",
     [1, 48, 28, 28],
     lambda: regnet.ResBottleneckBlock(48, 104, 2, nn.BatchNorm2d, nn.ReLU, group_width=8,
                                       bottleneck_multiplier=1.0, se_ratio=0.25)),
    ("tvh-deeplabv3-aspp", "block", deeplabv3, ["ASPPConv", "ASPPPooling", "ASPP"],
     "from collections.abc import Sequence\n",
     "ASPP(256, [12, 24, 36], 64)", [1, 256, 32, 32], lambda: deeplabv3.ASPP(256, [12, 24, 36], 64)),
    ("tvh-r3d-block", "block", video_resnet, ["Conv3DSimple", "BasicBlock"], "",
     "BasicBlock(64, 64, Conv3DSimple)", [1, 64, 4, 28, 28],
     lambda: video_resnet.BasicBlock(64, 64, video_resnet.Conv3DSimple)),
    ("tvh-r2plus1d-block", "block", video_resnet, ["Conv2Plus1D", "BasicBlock"], "",
     "BasicBlock(64, 64, Conv2Plus1D)", [1, 64, 4, 28, 28],
     lambda: video_resnet.BasicBlock(64, 64, video_resnet.Conv2Plus1D)),
    ("tvh-mc3-bottleneck", "block", video_resnet, ["Conv3DNoTemporal", "Bottleneck"], "",
     "Bottleneck(256, 64, Conv3DNoTemporal)", [1, 256, 4, 14, 14],
     lambda: video_resnet.Bottleneck(256, 64, video_resnet.Conv3DNoTemporal)),
    ("tvh-s3d-sepinception", "block", s3d, ["TemporalSeparableConv", "SepInceptionBlock3D"], PARTIAL + MISC,
     f"SepInceptionBlock3D(192, 64, 96, 128, 16, 32, 32, {S3D_NORM})", [1, 192, 4, 28, 28],
     lambda: s3d.SepInceptionBlock3D(192, 64, 96, 128, 16, 32, 32, S3D_NORM_REAL())),
    ("tvh-raft-residual-down", "block", raft, ["ResidualBlock"], MISC,
     "ResidualBlock(64, 96, norm_layer=nn.InstanceNorm2d, stride=2)", [1, 64, 32, 32],
     lambda: raft.ResidualBlock(64, 96, norm_layer=nn.InstanceNorm2d, stride=2)),
    ("tvh-raft-bottleneck", "block", raft, ["BottleneckBlock"], MISC,
     "BottleneckBlock(64, 64, norm_layer=nn.BatchNorm2d, stride=1)", [1, 64, 32, 32],
     lambda: raft.BottleneckBlock(64, 64, norm_layer=nn.BatchNorm2d, stride=1)),
    ("tvh-vgg11", "network", vgg, ["VGG", "make_layers"], "from typing import cast\n" + LOG,
     'VGG(make_layers([64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]), num_classes=1000)',
     [1, 3, 224, 224],
     lambda: vgg.VGG(vgg.make_layers([64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]),
                     num_classes=1000)),
    # Type hints name every conv builder and stem, so their source is shown too, as in the file.
    ("tvh-r3d-small", "network", video_resnet,
     ["Conv3DSimple", "Conv2Plus1D", "Conv3DNoTemporal", "BasicBlock", "Bottleneck", "BasicStem", "R2Plus1dStem",
      "VideoResNet"],
     "from collections.abc import Sequence\n" + LOG, "VideoResNet(BasicBlock, [Conv3DSimple] * 4, [1, 1, 1, 1], BasicStem, num_classes=10)",
     [1, 3, 8, 56, 56],
     lambda: video_resnet.VideoResNet(video_resnet.BasicBlock, [video_resnet.Conv3DSimple] * 4, [1, 1, 1, 1],
                                      video_resnet.BasicStem, num_classes=10)),
    ("tvh-mobilenetv3-small", "network", mobilenetv3,
     ["InvertedResidualConfig", "InvertedResidual", "MobileNetV3", "_mobilenet_v3_conf"],
     "from collections.abc import Sequence\n" + PARTIAL + MISC + SE + DIVISIBLE + LOG,
     'MobileNetV3(*_mobilenet_v3_conf("mobilenet_v3_small"), num_classes=1000)', [1, 3, 224, 224],
     lambda: mobilenetv3.MobileNetV3(*mobilenetv3._mobilenet_v3_conf("mobilenet_v3_small"), num_classes=1000)),
]

SETS = {"torchvision": ENTRIES, "torchvision_holdout": HOLDOUT}


def source(module, names: list[str], extra_imports: str, build: str, shape: list[int]) -> str:
    parts = [BASE_IMPORTS + extra_imports]
    parts += [inspect.getsource(getattr(module, name)) for name in names]
    parts.append(f"model = {build}\nexample_input = torch.randn({', '.join(map(str, shape))})\n")
    return "\n\n".join(p.rstrip() + "\n" for p in parts)


def same_graph(a: dict, b: dict) -> bool:
    match = lambda x, y: x["label"] == y["label"]
    return nx.is_isomorphic(to_networkx(a), to_networkx(b), node_match=match, edge_match=match)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--set", choices=sorted(SETS), default="torchvision")
    ap.add_argument("--freeze", action="store_true",
                    help="record the written file's sha256 in FROZEN.json (once; it refuses to overwrite)")
    args = ap.parse_args()
    out = DATA / f"{args.set}.jsonl"
    frozen = json.loads(FROZEN.read_text(encoding="utf-8")) if FROZEN.exists() else {}
    if args.set in frozen and not args.freeze:
        print(f"note: {out.name} is frozen; rebuilding only to check it is reproduced byte for byte")

    rows, failures = [], []
    for sid, kind, module, names, imports, build, shape, real in SETS[args.set]:
        code = source(module, names, imports, build, shape)
        try:
            graph, _ = graph_from_code(code)
        except Exception as exc:
            failures.append(f"{sid}: does not trace ({type(exc).__name__}: {exc})")
            continue
        torch.manual_seed(0)
        reference = graph_from_module(real(), example_input=torch.randn(*shape))
        if not same_graph(graph, reference):
            failures.append(f"{sid}: assembled source traces differently from torchvision's own object")
            continue
        if not graph["shapes_available"] or check(graph):
            failures.append(f"{sid}: no shapes, or fails our own structural checks {check(graph)}")
            continue
        rows.append({"id": sid, "tier": 5 if kind == "block" else 6, "kind": kind, "code": code,
                     "target": to_text(graph), "graph": graph, "num_nodes": len(graph["nodes"]),
                     "code_lines": len(code.strip().splitlines())})
        print(f"{sid:28s} {kind:8s} {len(graph['nodes']):3d} nodes  {rows[-1]['code_lines']:3d} lines")

    for f in failures:
        print("SKIPPED", f)
    text = "".join(json.dumps(r) + "\n" for r in rows)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if args.set in frozen:
        same = digest == frozen[args.set]["sha256"]
        print(f"\n{len(rows)} samples, sha256 {digest[:12]} -- "
              f"{'matches the frozen file' if same else 'DIFFERS from the frozen file; not written'}")
        if not same:
            raise SystemExit(1)
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"\n{len(rows)} samples -> {out.relative_to(ROOT)}  (sha256 {digest[:12]})")
    if args.freeze:
        import torchvision
        frozen[args.set] = {"sha256": digest, "samples": len(rows), "ids": [r["id"] for r in rows],
                            "torchvision": torchvision.__version__}
        FROZEN.write_text(json.dumps(frozen, indent=1) + "\n", encoding="utf-8")
        print(f"frozen in {FROZEN.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
