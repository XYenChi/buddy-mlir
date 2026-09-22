# RUN: %PYTHON %s
# SPDX-License-Identifier: Apache-2.0

import ctypes
import gc
import unittest
import weakref
from unittest.mock import patch

import numpy as np
import torch
from buddy.compiler._runtime import _TorchExecution
from buddy_mlir import runtime as rt
from test_execution import AliasEngine


class AllocatingEngine(AliasEngine):
    def __init__(self, array):
        super().__init__(array, outputs=2)
        self.graph._runtime_allocators = {"malloc", "free"}
        self.callbacks = {}

    def register_runtime(self, name, callback):
        self.callbacks[name] = callback

    def call(self, packed):
        super().call(packed)
        source = ctypes.cast(
            packed[1], ctypes.POINTER(ctypes.POINTER(self.descriptor))
        ).contents.contents
        output = ctypes.cast(
            packed[0], ctypes.POINTER(ctypes.POINTER(self.output_type))
        ).contents.contents
        address = self.callbacks["buddy_malloc"](32)
        ctypes.memmove(address, source.aligned, 32)
        for i in range(2):
            target = getattr(output, str(i))
            target.allocated = address
            target.aligned = ctypes.cast(address, type(target.aligned))


class TestOutputOwnership(unittest.TestCase):
    def make_execution(self):
        engine = AllocatingEngine(np.zeros(8, dtype=np.float32))
        execute = _TorchExecution(engine, engine.graph)
        released = []
        free = execute.allocators.libc.free

        def counted_free(address):
            released.append(address)
            free(address)

        execute.allocators.libc.free = counted_free
        return execute, engine, released

    def test_shared_storage_keeps_allocation_and_engine_alive(self):
        execute, engine, released = self.make_execution()
        x = torch.arange(8, dtype=torch.float32)
        first, second = execute(x)
        torch.testing.assert_close(first, x)
        self.assertTrue(torch._C._is_alias_of(first, second))
        view = second[2:]
        engine_ref = weakref.ref(engine)
        del first, second, execute, engine
        gc.collect()
        self.assertFalse(released)
        self.assertIsNotNone(engine_ref())
        torch.testing.assert_close(view, x[2:])
        del view
        gc.collect()
        self.assertEqual(len(released), 1)
        self.assertIsNone(engine_ref())

    def test_conversion_failure_releases_outputs(self):
        execute, _, released = self.make_execution()
        x = torch.arange(8, dtype=torch.float32)
        with (
            patch.object(
                rt,
                "ranked_memref_to_numpy",
                side_effect=RuntimeError("conversion failed"),
            ),
            self.assertRaisesRegex(RuntimeError, "conversion failed"),
        ):
            execute(x)
        gc.collect()
        self.assertEqual(len(released), 1)
        self.assertFalse(execute.allocators.live)

    def test_borrowed_input_is_not_freed(self):
        execute, engine, released = self.make_execution()
        execute.function = lambda packed: AliasEngine.call(engine, packed)
        x = torch.arange(8, dtype=torch.float32)
        first, second = execute(x)
        torch.testing.assert_close(first, x)
        del first, second
        gc.collect()
        self.assertFalse(released)


if __name__ == "__main__":
    unittest.main()
