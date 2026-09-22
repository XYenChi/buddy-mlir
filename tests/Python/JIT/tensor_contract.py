# RUN: %PYTHON %s
# SPDX-License-Identifier: Apache-2.0
"""User-visible Tensor contracts through Dynamo and AOTAutograd."""

from unittest.mock import patch

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


def test_layouts_and_dtypes():
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


def test_constant_isolation():
    for dtype in (
        torch.float32,
        torch.float64,
        torch.int64,
        torch.bool,
        torch.bfloat16,
    ):
        before = calls
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


def test_gradients():
    model, reference = Train(), Train()
    x = torch.randn(8, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    unused = torch.randn(8, requires_grad=True)
    compile_model(model)(x, unused).sum().backward()
    reference(y, unused).sum().backward()
    torch.testing.assert_close(x.grad, y.grad)
    torch.testing.assert_close(model.weight.grad, reference.weight.grad)
    assert unused.grad is None


def test_dynamic_shapes():
    compiled = compile_model(Add(), dynamic=True)
    for size in (8, 12, 8):
        before = calls
        x = torch.randn(size, size)
        torch.testing.assert_close(compiled(x), x + 1)
        assert calls > before


def test_lazy_input_flags():
    for x in (
        torch._neg_view(torch.randn(8)),
        torch.randn(8, dtype=torch.complex64),
        torch.randn(8, dtype=torch.complex64).conj(),
        torch.randn(8, dtype=torch.complex128).conj(),
    ):
        torch._dynamo.reset()
        before = calls
        torch.testing.assert_close(
            compile_model(Add())(x), x + 1, check_stride=True
        )
        assert calls > before


def test_expanded_output():
    def expand(x):
        return (x + 1).expand(4, 8)

    x = torch.randn(8)
    result = compile_model(expand)(x)
    torch.testing.assert_close(result, expand(x), check_stride=True)
    result[0].add_(2)
    torch.testing.assert_close(result[0], result[3])


def test_complex_add():
    def add(x, y):
        return torch.add(x, y, alpha=2 + 3j)

    x = torch.randn(2, 1, dtype=torch.complex64)
    y = torch.randn(1, 3, dtype=torch.complex64)
    before = calls
    torch.testing.assert_close(
        compile_model(add)(x, y), add(x, y), check_stride=True
    )
    assert calls > before


def test_split_output():
    def split(x):
        return (x + 1).split(3)

    x = torch.randn(8)
    result = compile_model(split)(x)
    torch.testing.assert_close(result, split(x), check_stride=True)
    assert all(torch._C._is_alias_of(result[0], y) for y in result[1:])


def test_dtype_view():
    def reinterpret(x):
        y = x + 1
        return y, y.view(torch.int32)

    x = torch.randn(8)
    result = compile_model(reinterpret)(x)
    torch.testing.assert_close(result, reinterpret(x), check_stride=True)
    assert torch._C._is_alias_of(*result)
    result[1].zero_()
    torch.testing.assert_close(result[0], torch.zeros_like(x))


def test_empty_sum():
    for dtype in (torch.float32, torch.int32):
        for dim in (0, 1):

            def reduce(x, dim=dim):
                return x.sum(dim=dim)

            x = torch.empty(0, 8, dtype=dtype)
            torch.testing.assert_close(compile_model(reduce)(x), reduce(x))


if __name__ == "__main__":
    with patch.object(_runtime._TorchExecution, "__call__", tracked):
        test_layouts_and_dtypes()
        test_constant_isolation()
        test_gradients()
        test_dynamic_shapes()
        test_lazy_input_flags()
        test_complex_add()
        test_expanded_output()
        test_split_output()
        test_dtype_view()
        test_empty_sum()
    print("Tensor contracts PASS")
