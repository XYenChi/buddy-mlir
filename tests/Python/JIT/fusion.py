# RUN: %PYTHON %s

import unittest

import torch
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa
from torch._inductor.decomposition import decompositions


class SharedProducer(torch.nn.Module):
    def forward(self, a, b):
        product = a * b
        return product, torch.relu(product + a)


class TestFusion(unittest.TestCase):
    def test_shared_producer_and_broadcast(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                torch.manual_seed(42)
                a = torch.randn(16, 32).to(dtype)
                b = torch.randn(32).to(dtype)
                model = SharedProducer()
                compiler = DynamoCompiler(
                    primary_registry=tosa.ops_registry,
                    aot_autograd_decomposition=decompositions,
                )
                compiler.importer_by_export(model, a, b)
                execute = compiler.dynamo_run()
                retained = execute(a, b)
                expected = list(model(a, b))
                torch.testing.assert_close(retained, expected)
                backing = torch.zeros(16, 64, dtype=dtype)
                backing[:, ::2] = a + 1
                other = execute(backing[:, ::2], b)
                torch.testing.assert_close(other, list(model(a + 1, b)))
                torch.testing.assert_close(retained, expected)


if __name__ == "__main__":
    unittest.main()
