# SPDX-License-Identifier: Apache-2.0

import ctypes
import threading

import numpy as np
import torch
from buddy_mlir import runtime as rt


def prepare_runtime_allocators(graph):
    """Give this JIT explicit allocator provenance without guessing pointers.

    Only direct calls emitted by the standard lowering are redirected. External
    runtime allocators are not adopted. Exported/AOT modules are unchanged.
    """
    from buddy_mlir import ir

    names = {
        name: "_mlir_ciface_buddy_" + name
        for name in ("malloc", "aligned_alloc", "free")
    }
    used = set()

    def visit(operation):
        if operation.name == "llvm.func" and "sym_name" in operation.attributes:
            name = ir.StringAttr(operation.attributes["sym_name"]).value
            if name in names:
                operation.attributes["sym_name"] = ir.StringAttr.get(
                    names[name]
                )
                used.add(name)
        if operation.name == "llvm.call" and "callee" in operation.attributes:
            name = ir.FlatSymbolRefAttr(operation.attributes["callee"]).value
            if name in names:
                operation.attributes["callee"] = ir.FlatSymbolRefAttr.get(
                    names[name]
                )
        for region in operation.regions:
            for block in region.blocks:
                for child in block.operations:
                    visit(child.operation)

    with graph._ctx:
        visit(graph._imported_module.operation)
    graph._runtime_allocators = used


class _NativeAllocation:
    def __init__(self, address, free, engine):
        self.address, self.free, self.engine = address, free, engine

    def __del__(self):
        if self.address:
            self.free(self.address)
            self.address = None


class _OwnedArray(np.ndarray):
    """Retain an allocation owner through NumPy views and Torch storage."""

    def __array_finalize__(self, parent):
        self.owner = getattr(parent, "owner", None)


class _RuntimeAllocators:
    def __init__(self, engine, names):
        self.live = set()
        self.lock = threading.Lock()
        self.libc = ctypes.CDLL(None)
        self.libc.malloc.argtypes = [ctypes.c_size_t]
        self.libc.malloc.restype = ctypes.c_void_p
        self.libc.free.argtypes = [ctypes.c_void_p]
        self.libc.free.restype = None
        self.callbacks = []

        def allocate(size):
            address = self.libc.malloc(size)
            if address:
                with self.lock:
                    self.live.add(address)
            return address

        def aligned_allocate(alignment, size):
            address = self.libc.aligned_alloc(alignment, size)
            if address:
                with self.lock:
                    self.live.add(address)
            return address

        def release(address):
            with self.lock:
                self.live.discard(address)
            self.libc.free(address)

        functions = {
            "malloc": (
                ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_size_t),
                allocate,
            ),
            "free": (ctypes.CFUNCTYPE(None, ctypes.c_void_p), release),
        }
        if "aligned_alloc" in names:
            self.libc.aligned_alloc.argtypes = [
                ctypes.c_size_t,
                ctypes.c_size_t,
            ]
            self.libc.aligned_alloc.restype = ctypes.c_void_p
            functions["aligned_alloc"] = (
                ctypes.CFUNCTYPE(
                    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t
                ),
                aligned_allocate,
            )
        for name in names:
            prototype, function = functions[name]
            callback = prototype(function)
            self.callbacks.append(callback)
            engine.register_runtime("buddy_" + name, callback)

    def adopt(self, address, engine):
        with self.lock:
            if address not in self.live:
                return None
            self.live.remove(address)
        return _NativeAllocation(address, self.libc.free, engine)


class _InputBackedArray(np.ndarray):
    """Keep copied inputs alive when a result aliases their storage."""

    inputs: tuple[np.ndarray, ...]


class _CallFrame:
    def __init__(self, graph, arrays, signature):
        self.signature = signature
        self.active = False
        self.descriptors = []
        self.aligned_slots = []
        for array in arrays:
            template = rt.get_ranked_memref_descriptor(array)
            # Copy metadata without retaining the first invocation's arrays.
            descriptor = type(template).from_buffer_copy(template)
            self.descriptors.append(descriptor)
            self.aligned_slots.append(
                ctypes.c_void_p.from_buffer(
                    descriptor, type(descriptor).aligned.offset
                )
            )
        self.input_slots = [
            ctypes.pointer(ctypes.pointer(d)) for d in self.descriptors
        ]
        self.output = graph._output_descriptor()
        self.output_slot = ctypes.pointer(ctypes.pointer(self.output))
        self.output_descriptors = [
            getattr(self.output, str(i))
            for i in range(len(graph._output_memref))
        ]
        self.output_pointers = [
            ctypes.pointer(descriptor) for descriptor in self.output_descriptors
        ]
        slots = [self.output_slot, *self.input_slots]
        self.packed = (ctypes.c_void_p * len(slots))(
            *(ctypes.cast(slot, ctypes.c_void_p).value for slot in slots)
        )


class _TorchExecution:
    """Execute a Buddy graph with one reusable call frame per Python thread."""

    def __init__(self, engine, graph):
        self.engine = engine
        self.graph = graph
        names = getattr(graph, "_runtime_allocators", ())
        self.allocators = _RuntimeAllocators(engine, names) if names else None
        self.function = engine.lookup(graph._func_name)
        self.local = threading.local()

    def __call__(self, *args):
        arrays = []
        for tensor in args:
            if tensor.device.type != "cpu":
                tensor = tensor.cpu()
            if tensor.dtype == torch.bfloat16:
                # Transfer raw BF16 bits without widening or canonicalizing NaNs.
                array = (
                    tensor.contiguous()
                    .view(torch.int16)
                    .numpy()
                    .view(np.uint16)
                )
                arrays.append(array.copy())
            elif not tensor.is_contiguous():
                # contiguous() already creates an isolated buffer. NumPy keeps
                # its owning Tensor alive, so a second input copy is redundant.
                arrays.append(tensor.contiguous().numpy())
            else:
                arrays.append(np.array(tensor.numpy(), copy=True))

        signature = tuple(
            (array.dtype, array.shape, array.strides) for array in arrays
        )
        frame = getattr(self.local, "frame", None)
        if frame is None or frame.active or frame.signature != signature:
            previous = frame
            frame = _CallFrame(self.graph, arrays, signature)
            # A nested invocation must not replace the outer thread's frame.
            if previous is None or not previous.active:
                self.local.frame = frame
        frame.active = True
        try:
            addresses = [array.ctypes.data for array in arrays]
            for descriptor, aligned, address in zip(
                frame.descriptors, frame.aligned_slots, addresses
            ):
                descriptor.allocated = address
                aligned.value = address
            self.function(frame.packed)
            outputs = []
            owners_by_address = {}
            if self.allocators is not None:
                # Adopt all returned allocations before any conversion can fail.
                for descriptor in frame.output_descriptors:
                    address = descriptor.allocated
                    if (
                        address not in addresses
                        and address not in owners_by_address
                    ):
                        owners_by_address[address] = self.allocators.adopt(
                            address, self.engine
                        )
            for descriptor, pointer in zip(
                frame.output_descriptors, frame.output_pointers
            ):
                out = rt.ranked_memref_to_numpy(pointer)
                if isinstance(out, np.ndarray):
                    owners = tuple(
                        array
                        for array, address in zip(arrays, addresses)
                        if descriptor.allocated == address
                    )
                    if owners:
                        out = out.view(_InputBackedArray)
                        out.inputs = owners
                    elif (
                        owners_by_address.get(descriptor.allocated) is not None
                    ):
                        out = out.view(_OwnedArray)
                        out.owner = owners_by_address[descriptor.allocated]
                    else:
                        # Borrowed globals must keep their JIT engine alive too.
                        out = out.view(_OwnedArray)
                        out.owner = self.engine
                outputs.append(out)
            # Repeated from_numpy calls create distinct StorageImpl objects even
            # for the same allocation. Build one storage for shared native
            # outputs so Torch can observe aliasing, not just matching pointers.
            groups = {}
            for index, (out, descriptor) in enumerate(
                zip(outputs, frame.output_descriptors)
            ):
                if isinstance(out, np.ndarray):
                    address = descriptor.allocated
                    if (
                        address not in addresses
                        and owners_by_address.get(address) is None
                    ):
                        address = ctypes.cast(
                            descriptor.aligned, ctypes.c_void_p
                        ).value
                    groups.setdefault((address, out.dtype), []).append(index)
                else:
                    outputs[index] = torch.tensor(out)
            for indices in groups.values():
                arrays_out = [outputs[i] for i in indices]
                address = frame.output_descriptors[indices[0]].allocated
                borrowed_global = (
                    address not in addresses
                    and owners_by_address.get(address) is None
                )
                if len(indices) == 1:
                    # A tensor constant must be writable without mutating (or
                    # writing into read-only) JIT global storage.
                    array = (
                        arrays_out[0].copy()
                        if borrowed_global
                        else arrays_out[0]
                    )
                    tensors = [torch.from_numpy(array)]
                else:
                    start = min(a.ctypes.data for a in arrays_out)
                    end = max(
                        a.ctypes.data
                        + (
                            sum(
                                (n - 1) * stride
                                for n, stride in zip(a.shape, a.strides)
                            )
                            + a.itemsize
                            if a.size
                            else 0
                        )
                        for a in arrays_out
                    )
                    buffer = (ctypes.c_byte * (end - start)).from_address(start)
                    storage = np.frombuffer(
                        buffer, dtype=arrays_out[0].dtype
                    ).view(_OwnedArray)
                    storage.owner = tuple(arrays_out)
                    base = torch.from_numpy(storage)
                    if borrowed_global:
                        base = base.clone()
                    tensors = [
                        base.as_strided(
                            a.shape,
                            tuple(s // a.itemsize for s in a.strides),
                            (a.ctypes.data - start) // a.itemsize,
                        )
                        for a in arrays_out
                    ]
                for index, array, result in zip(indices, arrays_out, tensors):
                    outputs[index] = (
                        result.view(torch.bfloat16)
                        if array.dtype == np.uint16
                        else result
                    )
            return outputs
        finally:
            frame.active = False
