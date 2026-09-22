# Torch JIT runtime tests

Run these scripts with the same Python environment and Buddy/LLVM runtime
libraries used by the Python lit suite:

```sh
python tests/Python/JIT/test_execution.py
python tests/Python/JIT/runtime_ownership.py
python tests/Python/JIT/strided_semantics.py
python tests/Python/JIT/tensor_contract.py
python tests/Python/JIT/fusion.py
python tests/Python/JIT/matmul_configuration.py
```

The runtime tests cover reusable thread-local call frames, isolated input buffers,
BF16 bit preservation, shared output storage and allocation lifetime. Ownership
checks include conversion failures and exclusion of borrowed input allocations.
The integration tests exercise strided indexing, mutation propagation, output
layouts, writable constants, gradients and concrete shape specialization through
`torch.compile`.

Symbolic inputs currently use an LRU cache of at most 32 concrete Buddy
compilations. Output allocation tracking applies to standard allocator calls in
the JIT; external/custom allocator ownership is not inferred. Layout restoration
and allocation tracking may add runtime overhead.

Validation on the `sg2044` and `v100` SSH hosts used PyTorch
`2.15.0a0+git91cc85c`. Both hosts reported riscv64/RVV CPU and no CUDA device.
The Tensor differential suite passed 27 cases per host, with only CUDA skipped.
The 100-call output-discarding memory check no longer showed the previous
approximately 101 MiB excess RSS growth. These results do not establish CUDA
support, exhaustive operator coverage or a latency improvement.

Raw benchmark samples, host-specific launchers and compressed logs are local
experiment artifacts, not source files for this test suite.
