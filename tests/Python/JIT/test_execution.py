# RUN: %PYTHON %s

import ctypes
import gc
import threading
import unittest
import weakref
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from buddy.compiler._runtime import _TorchExecution
from buddy_mlir import runtime as rt


class AliasEngine:
    def __init__(self, array, outputs=1):
        self.descriptor = type(rt.get_ranked_memref_descriptor(array))
        self.output_type = type(
            "Outputs", (ctypes.Structure,),
            {"_fields_": [(str(i), self.descriptor) for i in range(outputs)]},
        )
        self.graph = SimpleNamespace(
            _func_name="alias", _output_descriptor=self.output_type,
            _output_memref=[None] * outputs,
        )
        self.lookups = 0
        self.callback = None

    def lookup(self, name):
        self.lookups += 1
        return self.call

    def call(self, packed):
        source = ctypes.cast(
            packed[1], ctypes.POINTER(ctypes.POINTER(self.descriptor))
        ).contents.contents
        output = ctypes.cast(
            packed[0], ctypes.POINTER(ctypes.POINTER(self.output_type))
        ).contents.contents
        if self.callback is not None:
            self.callback(packed)
        for i in range(len(self.graph._output_memref)):
            target = getattr(output, str(i))
            ctypes.memmove(
                ctypes.addressof(target), ctypes.addressof(source),
                ctypes.sizeof(source),
            )


class TestExecution(unittest.TestCase):
    def make_execution(self, tensor, outputs=1):
        array = tensor.numpy() if tensor.dtype != torch.bfloat16 else tensor.float().numpy().view(np.uint32).astype(np.uint16)
        engine = AliasEngine(array, outputs)
        return _TorchExecution(engine, engine.graph), engine

    def test_reuse_updates_addresses_and_preserves_old_results(self):
        x = torch.arange(8, dtype=torch.float32)
        execute, engine = self.make_execution(x)
        with patch.object(rt, "get_ranked_memref_descriptor", wraps=rt.get_ranked_memref_descriptor) as build:
            first = execute(x)[0]
            frame = execute.local.frame
            second = execute(x + 20)[0]
            self.assertIs(execute.local.frame, frame)
            self.assertEqual(build.call_count, 1)
        self.assertEqual(engine.lookups, 1)
        torch.testing.assert_close(first, x)
        torch.testing.assert_close(second, x + 20)
        first.add_(100)
        torch.testing.assert_close(x, torch.arange(8, dtype=torch.float32))
        torch.testing.assert_close(second, x + 20)

    def test_shape_change_replaces_frame(self):
        x = torch.arange(3, dtype=torch.float32)
        execute, _ = self.make_execution(x)
        old = execute(x)[0]
        frame = execute.local.frame
        new = execute(torch.arange(7, dtype=torch.float32))[0]
        self.assertIsNot(execute.local.frame, frame)
        torch.testing.assert_close(old, x)
        torch.testing.assert_close(new, torch.arange(7, dtype=torch.float32))

    def test_noncontiguous_and_dtypes(self):
        for dtype in (torch.float32, torch.float64, torch.int32, torch.int64, torch.bfloat16):
            with self.subTest(dtype=dtype):
                x = torch.arange(16).to(dtype).reshape(4, 4).t()
                execute, _ = self.make_execution(x)
                actual = execute(x)[0]
                expected = x
                torch.testing.assert_close(actual, expected)

    def test_bfloat16_all_bit_patterns_and_output_isolation(self):
        # Include signed zeros, subnormals, infinities and every NaN payload.
        bits = np.arange(65536, dtype=np.uint16).reshape(256, 256)
        original = torch.from_numpy(bits.copy().view(np.int16)).view(torch.bfloat16)
        for x in (original, original.t(), original[:, ::2], original[:1].expand(8, 256)):
            with self.subTest(shape=x.shape, stride=x.stride()):
                execute, _ = self.make_execution(x, outputs=2)
                expected = x.contiguous().view(torch.int16).numpy().copy()
                first, second = execute(x)
                self.assertEqual(first.dtype, torch.bfloat16)
                np.testing.assert_array_equal(first.view(torch.int16).numpy(), expected)
                self.assertTrue(torch._C._is_alias_of(first, second))
                execute(torch.zeros_like(x))
                gc.collect()
                np.testing.assert_array_equal(second.view(torch.int16).numpy(), expected)
                first.zero_()
                self.assertEqual(torch.count_nonzero(second).item(), 0)
                np.testing.assert_array_equal(original.view(torch.int16).numpy().view(np.uint16), bits)

    def test_strided_inputs_are_isolated_and_c_order(self):
        base = torch.arange(120, dtype=torch.float32).reshape(10, 12)
        inputs = (
            base.t(), base[1::2, 2::3], base[:1].expand(8, 12),
            torch.arange(120, dtype=torch.float32).reshape(2, 3, 4, 5)
            .contiguous(memory_format=torch.channels_last),
            base[:0],
        )
        for x in inputs:
            with self.subTest(shape=x.shape, stride=x.stride()):
                execute, _ = self.make_execution(x)
                saved = x.clone()
                actual = execute(x)[0]
                self.assertTrue(actual.is_contiguous())
                torch.testing.assert_close(actual, saved)
                actual.add_(1)
                torch.testing.assert_close(x, saved)
                # The returned alias must survive reuse of the call frame.
                execute(torch.ones_like(x))
                gc.collect()
                torch.testing.assert_close(actual, saved + 1)

    def test_input_copies_live_until_last_output_view(self):
        x = torch.arange(8, dtype=torch.float32)
        execute, _ = self.make_execution(x, outputs=2)
        refs = []
        original_array = np.array
        def track_array(*args, **kwargs):
            array = original_array(*args, **kwargs)
            refs.append(weakref.ref(array))
            return array
        with patch.object(np, "array", side_effect=track_array):
            outputs = execute(x)
        view = outputs[1][::2]
        del outputs
        gc.collect()
        self.assertIsNotNone(refs[0]())
        torch.testing.assert_close(view, x[::2])
        del view
        gc.collect()
        self.assertIsNone(refs[0]())

    def test_threads_have_separate_frames(self):
        x = torch.arange(8, dtype=torch.float32)
        execute, engine = self.make_execution(x)
        barrier = threading.Barrier(2)
        frames = []
        def callback(packed):
            frames.append(packed[0])
            barrier.wait(timeout=10)
        engine.callback = callback
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(execute, x)
            second = pool.submit(execute, x + 20)
            outputs = first.result(), second.result()
        self.assertEqual(len(set(frames)), 2)
        torch.testing.assert_close(outputs[0][0], x)
        torch.testing.assert_close(outputs[1][0], x + 20)

    def test_reentrant_call_does_not_overwrite_outer_frame(self):
        x = torch.arange(8, dtype=torch.float32)
        execute, engine = self.make_execution(x)
        frames = []
        nested = []
        def callback(packed):
            frames.append(packed[0])
            if len(frames) == 1:
                nested.extend(execute(x + 20))
        engine.callback = callback
        outer = execute(x)[0]
        self.assertEqual(len(set(frames)), 2)
        torch.testing.assert_close(outer, x)
        torch.testing.assert_close(nested[0], x + 20)
        self.assertFalse(execute.local.frame.active)

    def test_exception_allows_frame_reuse(self):
        x = torch.arange(8, dtype=torch.float32)
        execute, engine = self.make_execution(x)
        def fail(packed):
            raise RuntimeError("test callback")
        engine.callback = fail
        with self.assertRaisesRegex(RuntimeError, "test callback"):
            execute(x)
        frame = execute.local.frame
        self.assertFalse(frame.active)
        engine.callback = None
        torch.testing.assert_close(execute(x)[0], x)
        self.assertIs(frame, execute.local.frame)

    def test_torch_compile_uses_cached_execution(self):
        from buddy.compiler.frontend import DynamoCompiler, TorchCompileBackend
        from buddy.compiler.ops import tosa
        from torch._inductor.decomposition import decompositions

        class AddRelu(torch.nn.Module):
            def forward(self, a, b):
                return torch.relu(a + b)

        model = AddRelu().eval()
        compiler = DynamoCompiler(
            primary_registry=tosa.ops_registry,
            aot_autograd_decomposition=decompositions,
        )
        with patch("buddy.compiler.frontend._TorchExecution", wraps=_TorchExecution) as create:
            compiled = torch.compile(model, backend=TorchCompileBackend(compiler))
            a, b = torch.randn(32), torch.randn(32)
            first = compiled(a, b)
            torch.testing.assert_close(first, model(a, b))
            torch.testing.assert_close(compiled(a + 1, b), model(a + 1, b))
            torch.testing.assert_close(first, model(a, b))
            self.assertGreater(create.call_count, 0)


if __name__ == "__main__":
    unittest.main()
