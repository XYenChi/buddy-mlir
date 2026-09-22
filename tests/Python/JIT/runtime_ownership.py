# RUN: %PYTHON %s
# SPDX-License-Identifier: Apache-2.0

import ctypes
import gc
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


x = torch.arange(8, dtype=torch.float32)
engine = AllocatingEngine(np.zeros(8, dtype=np.float32))
execute = _TorchExecution(engine, engine.graph)
released = []
free = execute.allocators.libc.free


def counted_free(address):
    released.append(address)
    free(address)


execute.allocators.libc.free = counted_free
first, second = execute(x)
torch.testing.assert_close(first, x)
assert torch._C._is_alias_of(first, second)
view = second[2:]
ref = weakref.ref(engine)
del first, second, execute, engine
gc.collect()
assert not released and ref() is not None
torch.testing.assert_close(view, x[2:])
del view
gc.collect()
assert len(released) == 1, released
assert ref() is None
print("Native shared output ownership PASS")

# A failed result conversion must release already-adopted native allocations.
engine = AllocatingEngine(np.zeros(8, dtype=np.float32))
execute = _TorchExecution(engine, engine.graph)
released.clear()
execute.allocators.libc.free = counted_free
with patch.object(
    rt, "ranked_memref_to_numpy", side_effect=RuntimeError("conversion failed")
):
    try:
        execute(x)
    except RuntimeError as error:
        assert str(error) == "conversion failed"
    else:
        raise AssertionError("conversion failure was swallowed")
gc.collect()
assert len(released) == 1
assert not execute.allocators.live

# A borrowed input is never adopted or freed by the native allocator tracker.
engine.call = lambda packed: AliasEngine.call(engine, packed)
execute.function = engine.call
released.clear()
first, second = execute(x)
torch.testing.assert_close(first, x)
del first, second
gc.collect()
assert not released
print("Conversion failure cleanup and borrowed input exclusion PASS")
