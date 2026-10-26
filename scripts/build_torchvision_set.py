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
"""

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

from graphart.extract import graph_from_code, graph_from_module
from graphart.schema import to_text
from graphart.validate import check, to_networkx

OUT = ROOT / "data" / "generated" / "torchvision.jsonl"

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


def source(module, names: list[str], extra_imports: str, build: str, shape: list[int]) -> str:
    parts = [BASE_IMPORTS + extra_imports]
    parts += [inspect.getsource(getattr(module, name)) for name in names]
    parts.append(f"model = {build}\nexample_input = torch.randn({', '.join(map(str, shape))})\n")
    return "\n\n".join(p.rstrip() + "\n" for p in parts)


def same_graph(a: dict, b: dict) -> bool:
    match = lambda x, y: x["label"] == y["label"]
    return nx.is_isomorphic(to_networkx(a), to_networkx(b), node_match=match, edge_match=match)


def main() -> None:
    rows, failures = [], []
    for sid, kind, module, names, imports, build, shape, real in ENTRIES:
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

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    print(f"\n{len(rows)} samples -> {OUT.relative_to(ROOT)}")
    for f in failures:
        print("SKIPPED", f)


if __name__ == "__main__":
    main()
