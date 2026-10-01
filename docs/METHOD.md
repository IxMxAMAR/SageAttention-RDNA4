# How measurements were taken

Every speed or accuracy claim in this repo was measured under the rules below. Most of them exist
because a careless measurement produced a confident, wrong number first. Those mistakes are listed
at the end, because they are the real reason for the rules.

Hardware: AMD Radeon RX 9070 XT (`gfx1201`, RDNA4, 64 CUs, 16 GB), Windows 11, a ROCm build of
PyTorch.

---

## 1. Speed

1. **Compare inside one process, interleaved.** Every candidate runs once per round, the rounds
   rotate the order, and a result is the median of at least 10 rounds (8 at the largest shapes).
   This is the most important rule. Clock drift is monotonic, so a sequential "A, then B"
   experiment can report a spurious 18 % win for B. Interleaving keeps the *ratio* trustworthy even
   while the absolute clock moves.
2. **Every block carries an A/A twin:** the same binary measured as if it were a second
   candidate. Its deviation from 1.0 is that block's noise floor. An effect smaller than the twin's
   deviation is not an effect.
3. **Warm up by time, not by call count:** at least 4 s and at least 40 calls, with a hard call cap
   so a blocking call cannot hang the loop. A 200-call warm-up once left 25 identical measurements
   drifting by 17.8 %.
4. **Time batches, not single calls:** one event pair per batch of about 10 calls, then divide.
   Roughly 0.2–1 % of single-call event pairs return nonsense.
5. **Never compile inside the timed process.** Triton and the loaders JIT-compile on first use.
   Each timed process runs only binaries a separate warm-up process has already built.
6. **Check the clock before and after every block.** A known fp16 GEMM is timed before and after
   each block. A block whose two probes disagree by more than about 3 % is void. On this card the
   probe can show the clock was stable, but it cannot pin the absolute clock, so results are
   reported as ratios.
7. **Nothing else on the GPU.** No browser video, no other GPU program, no second benchmark.
   Every block checks for foreign GPU clients before and after it runs.
8. **Reject impossible numbers.** Any implied throughput above the measured instruction ceiling
   (see §4) is re-measured, not reported.
9. **Large models need `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True`.** With about 13 of
   16 GB in use, PyTorch's caching allocator thrashed: step times doubled and became bimodal,
   while repeated renders of identical code still produced bit-identical images. With expandable
   segments, repeated renders agree within 0.5 %.

## 2. Correctness

1. **Bit-identity for changes that should not change the math.** A reordering, a new memory
   layout or a new dispatch path must produce *exactly* the same output (`torch.equal`) as the
   kernel it replaces, on every test cell. "Close enough" is not accepted for changes that claim to
   be pure.
2. **An fp64 reference for changes that do change the math.** The reference is computed in
   chunks of query rows; a full N×N fp32 score matrix once fragmented the allocator badly enough to
   halve a baseline. The bar is relative to the kernel being replaced: the new error divided by the
   old error, on the same inputs.
3. **Edge shapes every time:** N in {1, 15, 63, 64, 65, 127, 128, 129, 300, 1000}, batch 1 and 2,
   1/3/48 heads, causal and non-causal, smooth_k off and on.
4. **Outlier sweeps on every conversion.** Every quantized value (Q, K, V and the attention weights
   P) is pushed to, just above and far above the format's maximum (fp8 e4m3 tops out at 448), with
   tokens at |x| up to the fp16 maximum. This rule was added after a real render found an overflow
   that every earlier test had missed (see F150 in [FINDINGS.md](FINDINGS.md)).
5. **Coverage checks must use values the kernel cannot produce.** Pre-fill output buffers with a
   sentinel, never with zeros, or "untouched" looks like "correct".
6. **Every launch goes through a contract check.** The kernel is launched through raw `ctypes`, so
   nothing downstream bounds-checks. The loader rejects any tensor whose shape, dtype, stride or
   device does not match the contract, before the GPU sees it.

## 3. Hand-written kernels: gates before any GPU run

1. **Register and LDS budget, offline.** At most 240 VGPRs (6 waves per SIMD32 on this chip), no
   spills, and LDS small enough for 3 workgroups per WGP. These are read from the compiled code
   object, not assumed.
2. **Barrier liveness.** A checker walks the assembly and rejects any `s_barrier` that only some
   waves of a workgroup can reach. That is the class of bug that hangs the GPU.
3. **Read the assembly.** Count instructions per matrix instruction in the hot loop and compare
   against a budget. A change that should reduce work but does not reduce the count in the listing
   is void, and is not timed.

## 4. The ceiling

Measured with register-resident microbenchmarks on this card:

| | measured | documented |
|---|---|---|
| fp8 WMMA (`v_wmma_f32_16x16x16_fp8_fp8`) | 359 TFLOP/s | 2048 FLOP/clk/CU |
| fp16 WMMA | 199.6 TFLOP/s | 1024 FLOP/clk/CU |
| dependent WMMA latency | 27 clock ticks | |
| LDS read throughput | 14.6 TB/s | |

Never quote a ratio of two *achieved* numbers (1.8× here) as the instruction rate (2.0×).

## 5. How a result is accepted

1. **Predict first.** Before a measurement, write down what is expected, with an interval, and
   the decision rule. For example: "kernel_vs_base ≤ 0.98 at both non-causal sizes". The prediction
   is frozen. If it turns out to need changing, the change is recorded next to it, never in place.
2. **A miss is a result.** A refuted prediction is recorded as refuted and the route is closed
   with its numbers, so it is not tried again by accident.
3. **Every number in a write-up is re-checked against the raw logs** before the write-up is
   accepted. That check has caught arithmetic slips, a wrong line citation, and twice a conclusion
   the data did not support.
4. **Resume safely.** After an interrupted run, here usually a power cut, every artifact is
   verified by hash and by a clean rebuild before it is reused. A timing block that did not finish
   is treated as never run.

## 6. Measurement mistakes that produced these rules

| wrong result | cause | rule |
|---|---|---|
| SDPA at 33 TFLOP/s (really 65) | a full N×N fp32 reference fragmented the allocator | chunked reference (§2.2) |
| 0.0056 ms, i.e. >12 000 TFLOP/s | unknown; appeared on the first config of a sweep | plausibility check (§1.8) |
| "fp8 gives no tensor-core speed-up" | the test matmul was memory-bound | register-resident roof loop (§4) |
| the same configuration at 67.1 vs 49.1 TFLOP/s | clock drift across two processes | one process, interleaved (§1.1) |
| "gfx12 has no hardware fp8 convert" | an inline-asm constraint aborted the compiler, and the abort was read as the answer | check the instruction set directly |
| an 18 % "win" | sequential A-then-B timing | interleaving (§1.1) |
| a speed-up from a variant that changed nothing | the timing harness silently overwrote one A/A twin with the other | A/A ratios derived from raw medians |
| a "new kernel" end-to-end result | the model ran in bf16; the kernel accepts fp16 only and fell back silently | prove which kernel served each call (§2.6, F152) |
| a black image in a real render | P × V-scale overflowed fp8 for one large-V token; no test had |V| > 448 | outlier sweep (§2.4) |
