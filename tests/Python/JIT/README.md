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
BF16 bit preservation, lazy conjugate/negative input flags, shared output storage
across dtypes and allocation lifetime. Ownership checks include conversion
failures, NumPy/DLPack consumers and exclusion of borrowed input allocations.
The integration tests exercise strided indexing, mutation propagation, output
layouts, writable constants, gradients and concrete shape specialization through
`torch.compile`. They also check absolute storage offsets, expanded output
strides, default split dimensions, dtype-view mutation propagation, empty sums
and complex addition with broadcasting and complex alpha.

Symbolic inputs currently use an LRU cache of at most 32 concrete Buddy
compilations. Output allocation tracking applies to standard allocator calls in
the JIT; external/custom allocator ownership is not inferred. Layout restoration
and allocation tracking may add runtime overhead.

Strided gathers require a dense represented input storage; reaching outside a
sliced input's represented storage remains unsupported. Dtype-view lowering
supports equal-width integer/float reinterpretation and complex/component views.
Other width-changing reinterpretations and general complex operator coverage
remain outside these tests. AOT output alias metadata restores shared storage
when value-only tensor lowering creates separate output allocations.

Validation on the `sg2044` and `v100` SSH hosts used PyTorch
`2.15.0a0+git91cc85c`. Both hosts reported riscv64/RVV CPU and no CUDA device.
The original and expanded Tensor differential suites passed 43 cases per host,
with only CUDA skipped. Seven additional controlled ABI/ownership cases passed
on each host, including mixed-dtype aliases and release after the last consumer.
The 100-call output-discarding memory check no longer showed the previous
approximately 101 MiB excess RSS growth. These results do not establish CUDA
support, exhaustive operator coverage or a latency improvement.

Raw benchmark samples, host-specific launchers and compressed logs are local
experiment artifacts, not source files for this test suite.
