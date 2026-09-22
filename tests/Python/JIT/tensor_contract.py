# RUN: %PYTHON %s
# SPDX-License-Identifier: Apache-2.0
"""User-visible Tensor contracts through Dynamo and AOTAutograd."""

import torch
from buddy.compiler import _runtime
from buddy.compiler.frontend import DynamoCompiler, TorchCompileBackend
from buddy.compiler.ops import tosa
from torch._inductor.decomposition import decompositions

calls = 0
original = _runtime._TorchExecution.__call__


def tracked(self, *args):
    global calls
    calls += 1
    return original(self, *args)


_runtime._TorchExecution.__call__ = tracked


def compile_model(model, dynamic=False):
    compiler = DynamoCompiler(
        primary_registry=tosa.ops_registry,
        aot_autograd_decomposition=decompositions,
    )
    return torch.compile(
        model,
        backend=TorchCompileBackend(compiler),
        fullgraph=True,
        dynamic=dynamic,
    )


class Add(torch.nn.Module):
    def forward(self, x):
        return x + 1


for x in (
    torch.randn(4, 8).t(),
    torch.randn(2, 3, 4, 5).contiguous(memory_format=torch.channels_last),
    torch.empty(0, 8),
    torch.randn(8).bfloat16(),
):
    before = calls
    torch.testing.assert_close(
        compile_model(Add())(x), x + 1, check_stride=True
    )
    assert calls > before


class Constant(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.dtype = dtype

    def forward(self, x):
        return x + 1, torch.tensor([0, 1, 2], dtype=self.dtype)


for dtype in (
    torch.float32,
    torch.float64,
    torch.int64,
    torch.bool,
    torch.bfloat16,
):
    before = calls
    print("Constant", dtype, flush=True)
    model = Constant(dtype)
    x = torch.randn(8)
    compiled = compile_model(model)
    output = compiled(x)
    torch.testing.assert_close(output, model(x))
    output[1].zero_()
    # Returning a tensor constant must not expose mutable JIT global storage.
    torch.testing.assert_close(compiled(x), model(x))
    assert calls > before


class Train(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.arange(1, 9, dtype=torch.float32)
        )

    def forward(self, x, unused):
        return (x * self.weight).square()


model, reference = Train(), Train()
x = torch.randn(8, requires_grad=True)
y = x.detach().clone().requires_grad_()
unused = torch.randn(8, requires_grad=True)
compile_model(model)(x, unused).sum().backward()
reference(y, unused).sum().backward()
torch.testing.assert_close(x.grad, y.grad)
torch.testing.assert_close(model.weight.grad, reference.weight.grad)
assert unused.grad is None

compiled = compile_model(Add(), dynamic=True)
for size in (8, 12, 8):
    before = calls
    x = torch.randn(size, size)
    torch.testing.assert_close(compiled(x), x + 1)
    assert calls > before
print("Tensor contracts PASS")
