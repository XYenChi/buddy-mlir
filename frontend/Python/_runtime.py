# SPDX-License-Identifier: Apache-2.0

import ctypes
import threading

import numpy as np
import torch
from buddy_mlir import runtime as rt


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
        self.function = engine.lookup(graph._func_name)
        self.local = threading.local()

    def __call__(self, *args):
        arrays = []
        for tensor in args:
            if tensor.device.type != "cpu":
                tensor = tensor.cpu()
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.contiguous()
                f32 = tensor.to(dtype=torch.float32).numpy()
                # Widening BF16 creates independent storage, so its bits can
                # be shifted in place before narrowing to the ABI buffer.
                bits = f32.view(np.uint32)
                bits >>= 16
                array = bits.astype(np.uint16)
                # astype already owns an isolated buffer.
                arrays.append(array)
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
            for descriptor, pointer in zip(
                frame.output_descriptors, frame.output_pointers
            ):
                out = rt.ranked_memref_to_numpy(pointer)
                if isinstance(out, np.ndarray) and out.dtype == np.uint16:
                    # Widening allocates independent storage; shift in place
                    # without another full-size uint32 temporary.
                    out = np.asarray(out, dtype=np.uint32)
                    out <<= 16
                    out = out.view(np.float32)
                elif isinstance(out, np.ndarray):
                    owners = tuple(
                        array
                        for array, address in zip(arrays, addresses)
                        if descriptor.allocated == address
                    )
                    if owners:
                        out = out.view(_InputBackedArray)
                        out.inputs = owners
                if isinstance(out, np.ndarray):
                    outputs.append(torch.from_numpy(out))
                else:
                    outputs.append(torch.tensor(out))
            return outputs
        finally:
            frame.active = False
