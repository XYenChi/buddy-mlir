# RUN: %PYTHON %s

import platform
import unittest

import torch
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import Graph
from buddy.compiler.ops import tosa
from torch._inductor.decomposition import decompositions


class Matmul(torch.nn.Module):
    def forward(self, a, b):
        return a @ b


class TestMatmulConfiguration(unittest.TestCase):
    def run_matmul(self, size, vector_type):
        torch.manual_seed(42)
        a, b = torch.randn(31, 37), torch.randn(37, 43)
        compiler = DynamoCompiler(
            primary_registry=tosa.ops_registry,
            aot_autograd_decomposition=decompositions,
        )
        compiler.importer_by_export(Matmul(), a, b)
        execute = compiler.dynamo_run(
            matmul_vector_size=size, matmul_vector_type=vector_type
        )
        torch.testing.assert_close(execute(a, b)[0], a @ b, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(
            execute(a + 1, b)[0], (a + 1) @ b, atol=1e-4, rtol=1e-4
        )
        spelling = f"[{size}]" if vector_type == "scalable" else str(size)
        self.assertIn(f"vector<{spelling}xf32>", str(execute.graph._imported_module))

    def test_fixed_widths(self):
        for size in (4, 8, 16, 32):
            with self.subTest(size=size):
                self.run_matmul(size, "fixed")

    @unittest.skipUnless(
        platform.machine() == "riscv64"
        and torch.backends.cpu.get_cpu_capability() == "RVV",
        "requires an RVV execution target",
    )
    def test_scalable_rvv(self):
        self.run_matmul(4, "scalable")

    def test_invalid_configuration(self):
        graph = Graph.__new__(Graph)
        for size in (0, -1, True, "4"):
            with self.subTest(size=size), self.assertRaisesRegex(ValueError, "positive integer"):
                graph.lower_to_llvm_ir(matmul_vector_size=size)
        with self.assertRaisesRegex(ValueError, "fixed or scalable"):
            graph.lower_to_llvm_ir(matmul_vector_type="unknown")


if __name__ == "__main__":
    unittest.main()
