# RUN: %PYTHON %s
# SPDX-License-Identifier: Apache-2.0

import torch
from buddy.compiler.frontend import DynamoCompiler, TorchCompileBackend
from buddy.compiler.ops import tosa
from torch._inductor.decomposition import decompositions


def backend():
    return TorchCompileBackend(
        DynamoCompiler(
            primary_registry=tosa.ops_registry,
            aot_autograd_decomposition=decompositions,
        )
    )


class Strided(torch.nn.Module):
    def __init__(self, size, stride, offset):
        super().__init__()
        self.size, self.stride, self.offset = size, stride, offset

    def forward(self, x):
        return torch.as_strided(x, self.size, self.stride, self.offset) + 1


class Overlapping(torch.nn.Module):
    def forward(self, x, y):
        x.add_(1)
        return y, x + y


class Slice(torch.nn.Module):
    def forward(self, x):
        return x[1:, 1::2], x + 1


for size, stride, offset in [
    ((8,), (1,), 4),
    ((3, 2), (1, 4), 2),
    ((2, 3), (0, 2), 1),
]:
    model = Strided(size, stride, offset)
    x = torch.arange(12, dtype=torch.float32)
    compiled = torch.compile(model, backend=backend(), fullgraph=True)
    torch.testing.assert_close(compiled(x), model(x))

model = Overlapping()
expected_base = torch.arange(12, dtype=torch.float32)
actual_base = expected_base.clone()
expected = model(expected_base[:8], expected_base[4:])
actual = torch.compile(model, backend=backend(), fullgraph=True)(
    actual_base[:8], actual_base[4:]
)
torch.testing.assert_close(actual, expected)
torch.testing.assert_close(actual_base, expected_base)
actual[0].add_(10)
expected[0].add_(10)
torch.testing.assert_close(actual_base, expected_base)

model = Slice()
x = torch.arange(64, dtype=torch.float32).reshape(8, 8)
y = x.clone()
actual = torch.compile(model, backend=backend(), fullgraph=True)(x)
expected = model(y)
torch.testing.assert_close(actual, expected, check_stride=True)
actual[0].add_(5)
expected[0].add_(5)
torch.testing.assert_close(x, y)
print("Strided semantics PASS")
