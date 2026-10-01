# Findings catalogue

This is the record of what was tried, measured and decided while porting SageAttention's quantized attention to AMD RDNA4 (gfx1201, RX 9070 XT) on Windows, with the aim of beating the existing gfx12 port in SageAttention PR #368. It lists every numbered finding and every work track. Refuted and closed results are kept next to the wins: they are what stops the next person from repeating them.

The work went through four phases.

1. **A Triton fp8 kernel.** The gap to PR #368 was diagnosed, levers were tried and closed, accuracy work was done (per-token fp8 scaling, `smooth_k`, `smooth_v`), and correctness bugs were found and fixed.
2. **Attempts to hand-edit Triton's compiled output.**
3. **A hand-written HIP kernel, SK1, and its variants.** `sk1_t1` (one softmax rescale per 64 keys) and `sk1_t4a1` (mask and skip work guarded by one branch) were wins; many other variants were refuted.
4. **Shipping.** Packaging behind a flag, then default-on with fallback; a real-world fp8 overflow found when a render went black, and its fix; end-to-end tests in a diffusion model; widening the input envelope.

## How to read an entry

Each entry gives the question that was asked, the result with its key numbers as the project's records state them, and a status.

| status | meaning |
|---|---|
| confirmed | the claim or measurement stands |
| refuted | the hypothesis or prediction was shown false, or the claim was later retracted or overturned |
| closed | a route judged not worth pursuing, without a cleaner refutation |
| shipped | the change went into the shipped default or the package |
| open | unresolved, never run, or void and not redone |

Where a later result overturned or corrected an earlier one, both entries say so.

**Speed numbers are ratios.** The protocol the project settled on times the candidates inside one process, interleaved, with an A/A twin (the same binary measured as if it were a second candidate) to give the noise floor of each block; the earliest findings predate parts of it, and the entries say where a number was later found to be affected. Absolute milliseconds depend on the GPU's clock state (see F073) and are not quoted. Unless an entry says "speedup" or "faster by", a ratio is candidate time divided by reference time, so below 1 means the candidate is faster. A few early entries use the opposite convention ("speed relative to PR #368") and say so. A ratio is labelled kernel-only (the attention kernel alone) or full-call (the whole call, including quantization); where an entry does not say, the source did not.

**Terms.**

- *PR #368, the incumbent (also "the fork"):* the gfx12 port in SageAttention PR #368 (HIP kernels), which this work set out to beat.
- *SDPA:* PyTorch's `scaled_dot_product_attention`, which runs AOTriton on this stack. *Prologue:* the quantization pass or passes that run before the attention kernel.
- *Shipped:* the project's own kernel in the state it was in at the time of the finding. *Production:* the SageAttention kernel that ComfyUI actually ran on this machine.
- *SK1:* the hand-written HIP kernel.
- *(1,48,8771,128):* batch, heads, sequence length N and head dimension of Krea2's self-attention, the "real" or "production" shape. N is the sequence length throughout.
- *`smooth_k`, `smooth_v`:* subtract the sequence-axis mean of K (or V) before quantizing, and account for it afterwards.
- *Frozen, pre-registered:* the question, the predictions and the decision rule were written down before the run and not edited afterwards; deviations are recorded separately. *Lever, non-lever:* a change that did, or did not, clear its frozen threshold. *Void:* a block or cell whose controls failed, so its numbers are not to be quoted.
- *Bracket:* an identical copy of a kernel carried alongside it in the same block to measure the timing artifact; "bracket-corrected" means corrected by it. *Slots, slots per `wmma`:* instruction-issue slots in the KV loop body, counted per `v_wmma` matrix instruction.

**Numbering.** F-numbers are the project's own finding numbers. Work tracks were numbered separately (H for the hand-edit and hand-kernel tracks, S for "stage" reports); here they have descriptive names with the original code in small text. Where a finding exists only as the pre-registration of a track, its number appears on the track's entry. There is no F048, no F103, and there are no stages S4, S26 or S29. Within a group, entries are in the order the work was done.

## Contents

**Phase 1: the Triton fp8 kernel**

1. [Baseline and early corrections](#1-baseline-and-early-corrections) (17 entries)
2. [Levers on the Triton loop](#2-levers-on-the-triton-loop) (17 entries)
3. [Build, driver and clock environment](#3-build-driver-and-clock-environment) (4 entries)
4. [INT4, sparsity and other routes that were costed first](#4-int4-sparsity-and-other-routes-that-were-costed-first) (7 entries)
5. [The production shape and the gap to PR #368](#5-the-production-shape-and-the-gap-to-pr-368) (19 entries)
6. [Accuracy: per-token fp8, smooth_k and int8 QK](#6-accuracy-per-token-fp8-smooth_k-and-int8-qk) (7 entries)
7. [Odd lengths, the EVEN_N mask and a shipped-kernel bug](#7-odd-lengths-the-even_n-mask-and-a-shipped-kernel-bug) (8 entries)
8. [Causal, long-sequence and end-to-end measurements](#8-causal-long-sequence-and-end-to-end-measurements) (6 entries)
9. [smooth_v: a lever that kept flipping](#9-smooth_v-a-lever-that-kept-flipping) (14 entries)
10. [The sv fold and the per-channel V scale](#10-the-sv-fold-and-the-per-channel-v-scale) (12 entries)
11. [The long-N gap and the last Triton stages](#11-the-long-n-gap-and-the-last-triton-stages) (17 entries)

**Phase 2: hand edits**

12. [Hand-editing Triton's compiled output](#12-hand-editing-tritons-compiled-output) (13 entries)

**Phase 3: the hand-written kernel**

13. [SK1: a hand-written HIP kernel](#13-sk1-a-hand-written-hip-kernel) (14 entries)

**Phase 4: shipping**

14. [Packaging, shipping and end to end](#14-packaging-shipping-and-end-to-end) (6 entries)

## 1. Baseline and early corrections

The first findings set the baseline: whether the incumbent builds and runs, what the hardware can do, where the Triton fp8 kernel stands against PR #368 and SDPA, and which early claims had to be corrected. Several entries here were corrected by findings further down; the correction is marked on both sides.

### F001 — The prebuilt PR #368 wheel does not load

**Question.** Can the prebuilt PR #368 gfx12 wheel be imported and used as the baseline on the current PyTorch/ROCm stack?

**Result.** No. The import fails with "DLL load failed while importing _qattn_gfx12_native". The first diagnosis blamed a PyTorch C++ ABI break and counted 26 of 53 `torch_cpu.dll` symbols and 1 of 65 `c10.dll` symbols as missing. The `torch_cpu` count was a `pefile` false positive (the DLL has 60,175 named exports and `pefile` reported 8,192). The real failure is a single missing `c10` symbol.

**Status:** confirmed (the wheel does not load). Its mechanism was corrected by F027.

### F002 — fp8 WMMA runs at about twice the fp16 rate

**Question.** Does fp8 WMMA on RDNA4 have a compute-rate advantage over fp16, or does quantization buy nothing?

**Result.** A register-resident WMMA loop measured fp16 at 108.3, bf16 at 97.3 and fp8 e4m3 at 195.6 TFLOP/s: an achieved fp8/fp16 ratio of 1.81 against a documented instruction-rate ratio of 2.0. The earlier reading that fp8 gives no advantage (equal instruction counts per kernel) was wrong, and an earlier 2048³ matmul test that showed 1.13× was memory-bound. A dedicated microbenchmark later measured the fp8 WMMA peak at 359.1 TFLOP/s against 199.6 for fp16 (ratio 1.799) with marginal issue costs of 8 and 16 ticks, so 195.6 was an achieved loop rate, not a ceiling (track H15).

**Status:** confirmed (fp8 is about 2× fp16 at the instruction level).

### F003 — Corrections on int8 dot, fp8 rate and the V-permute

**Question.** Which early assumptions were wrong: int8 `tl.dot`, the fp8 rate against peak, and the incumbent's V-permute?

**Result.** int8 `tl.dot` does compile and run when given `out_dtype=tl.int32` (`v_wmma_i32_16x16x16_iu8`, max abs error 0.0 against an integer reference), which corrects F002's statement. The measured 108.3 and 195.6 TFLOP/s are 69% and 62% of the 2400 MHz figure, so 195.6 is an achievable-loop rate, not a spec peak. The hypothesis that the incumbent applies NVIDIA's V-permute is falsified: its source has 0 matches for the permute, so the 1.31× regression reported against it is not a V-layout cost.

**Status:** confirmed. The finding's ranking of pipeline and LDS weak spots was later superseded (see F018).

### F004 — Folding V's scale into P

**Question.** Can per-channel V quantization, a sequence-axis reduction, be replaced by a per-row scale folded into P?

**Result.** The fold is algebraically exact. The per-tensor prologue became about 20–30× faster, and max error fell from 0.00615 to 0.00117 (B=1, H=8, N=8192, D=128). The file's "no accuracy downside" claim was partly falsified by F005 for V with per-channel offsets.

**Status:** confirmed (speed-up), with the accuracy claim limited by F005.

### F005 — Per-token V scaling fails on offset V

**Question.** Does per-token V quantization lose accuracy against per-channel scaling and `smooth_v` when V has a per-channel offset?

**Result.** On benign V, per-token is best (error 2.53e-2 against 2.88e-2 for per-channel and 3.38e-2 for `smooth_v`). With a per-channel offset of 6.0, `smooth_v` is 170× more accurate (4.54e-5 against 7.74e-3), and 29× with offset plus scale spread. With a single ×40 outlier token, per-token is 2.8× worse than per-channel. F086 later found that this last case does not reproduce in the project's kernel (0.936×, `smooth_v` better), because F005's table confounded centring with the change from per-token to per-channel scale; the two offset cases did reproduce.

**Status:** confirmed for the offset cases; the outlier case was overturned by F086. It falsifies the "no downside" claim in F004 and opens the `smooth_v` thread (F085 onward).

### F006 — Can Triton reach the hardware fp8 convert?

**Question.** Can Triton use the gfx1201 hardware fp8 convert instruction, or must the kernel move to HIP?

**Result.** The first version said it was unreachable, which was wrong. Inline asm with the `=v` constraint and an int16 dtype with `pack=2` works and is bit-exact against torch e4m3 (0 mismatches in 2^20 samples). The loop body drops from 2670 to 1108 instructions (BM=64, BN=64), a cut of 48–59%. At N=8192 with the 64/64/4 tile the time ratio against the default cast is 0.780 (default cast = 1.000), the fastest variant in 7 of 8 config/shape sets.

**Status:** refuted for the original conclusion ("Triton cannot reach the fp8 convert"); the corrected result stands and the hardware convert became the default.

### F007 — The incumbent is the fastest fp8 path

**Question.** Is PR #368's kernel really 1.31× slower than plain fp16 at N=8192, as issue #389 reported, once built from source here?

**Result.** Built from source and measured, it is 1.76× faster than SDPA at N=8192 non-causal, not 1.31× slower. Our fp8 kernel-only was 4% behind it at N=2048 and 35% behind at N=4096, with 50% end-to-end overhead at N=2048. Non-causal accuracy was comparable; the incumbent looked about 1.4× better on causal, which F014 later traced to a smoothing confound.

**Status:** confirmed. The premise that beating the incumbent would be easy was falsified.

### F008 — Fusing Q quantization into the kernel; the prologue is memory-bound

**Question.** Does quantizing Q inside the kernel and combining the K and V launches close the prologue deficit, and is the prologue instruction-bound?

**Result.** Fused Q is bit-identical (max difference 0.000e+00) and cuts the full-call time at N=2048 by 13% non-causal and 7% causal. Using the hardware convert in the prologue changed nothing, so the prologue is memory/latency-bound, not instruction-bound. Dropping the V prologue (fp16 P·V) is a net loss at causal=0, though 4.3× more accurate non-causally and 8.3× causally.

**Status:** confirmed.

### F009 — The prologue cost gap

**Question.** Why does the prologue cost far more end to end than its kernel time suggests at N=2048?

**Result.** A launch-geometry sweep found BLOCK_R=16 with 4 warps already optimal at N=2048 and within 3% at N=8192. Reusing buffers was not faster. A cold-L2 test explained the large-N gap (cold/hot 2.09× at N=8192) but not N=2048, where cold ran faster (0.50×) and the gap is host-side.

**Status:** closed (geometry and allocator levers are null). The host-overhead explanation was taken over and confirmed by F013.

### F010 — Our kernel against PR #368 at N=8192 causal

**Question.** At N≥4096, where does the fp8 kernel stand against PR #368, and can Triton pipelining close the non-causal gap?

**Result.** Speed relative to PR #368 (above 1 means ours is faster): at N=8192 causal our kernel-only is 1.05× and our end-to-end call 0.99× (parity); at N=8192 non-causal it is only 0.62× end to end (0.64× kernel-only); N=4096 is 0.68× non-causal and 0.83× causal end to end. A sweep of 12 tile configurations left none within 1.6× of the incumbent at N=8192 non-causal, and a `num_stages` retest closed Triton pipelining.

**Status:** confirmed for the measured numbers. The causal explanation was retracted by F016, and one resource row was a data error (corrected in F018).

### F011 — A 16-wide KV tile is worth about 20%

**Question.** Does a narrower KV tile (BN=16) recover speed at N=8192 non-causal, and is register pressure the lever?

**Result.** The 128×16 (4-warp) tile is 20% faster than 128×32 at N=8192 non-causal, moving speed relative to PR #368 from 0.63× to 0.75× (1.00× is parity). At N=8192 causal it reaches 1.11× of PR #368. The register-pressure hypothesis was refuted: a 64×32 tile with 165 registers and 0 spills was slower.

**Status:** confirmed. F018 later refuted the occupancy/LDS-capacity form of the explanation.

### F012 — A single-process table: an end-to-end win at N=8192 causal

**Question.** With all shapes re-measured in one interleaved process and the auto-selected tile, where do we stand against PR #368?

**Result.** In the first table, at N=8192 causal our end-to-end call takes 0.88× of PR #368's time (a 12% win) and is 1.59× faster than SDPA; the other five shapes lose (for example 1.30× at N=2048 non-causal and 1.32× at N=8192 non-causal, as ours/PR #368 time). A second run in the same file, after the kernel-only timing path was repaired, gives 0.90× end to end and 1.57× against SDPA at N=8192 causal, and adds kernel-only wins at both causal shapes (0.84× at N=2048, 0.83× at N=8192). In that second run the prologue is 23.5% of the end-to-end time at N=2048 causal.

**Status:** confirmed. Superseded by the definitive table in F015; the direction of the results did not change.

### F013 — The short-N prologue cost is host overhead

**Question.** Is the N=2048 prologue gap host-side launch overhead that CUDA-graph capture removes, with no effect at N=8192?

**Result.** Graph replay recovered 51.4 µs at N=2048 causal, 45.6 µs at N=2048 non-causal and 3.6 µs at N=8192 causal, matching the predicted gap of about 49 µs. End to end against PR #368 the ratio is 0.85× at N=2048 causal (a 15% win) and 0.88× at N=8192 causal, but 1.15× at N=2048 non-causal. Graphs need fixed shapes and static input pointers.

**Status:** confirmed. Graphs are not universal: they help at N≤4096 and F015 found them worse at N=8192.

### F014 — Correctness against a chunked fp32 reference

**Question.** Is the fp8 kernel numerically correct against PR #368 and an fp32 reference built from the same fp16 inputs?

**Result.** Non-causal max-abs error ratio ours/PR #368 is 0.99× (N=1024), 1.03×, 1.06× and 0.99× (N=8192): parity. Causal ratios were 1.41×, 1.71× and 1.57×, first read as a real 1.4–1.7× gap caused by `smooth_k`. fp16 SDPA is 50–200× more accurate than either fp8 path. Afterwards the causal comparison was found to be confounded (a smoothed incumbent against an unsmoothed kernel), and a "2.2× better" replacement claim did not reproduce (F021). The current position is parity non-causal and about 1.5× worse causal.

**Status:** confirmed for non-causal parity. The causal claim was corrected.

### F015 — The definitive six-shape table

**Question.** With every adopted change in one process, how do we compare to PR #368 and SDPA at each shape, and do CUDA graphs always help?

**Result.** Kernel-only ours/PR #368 time ratios (below 1 means ours is faster): 0.84 (N=2048 causal), 0.82 (N=8192 causal), 1.03 (N=4096 causal), 1.06 (N=2048 non-causal), 1.23 (N=4096 non-causal), 1.25 (N=8192 non-causal). Graph replay helps at N≤4096 (−16% and −17% at N=2048) and is worse at N=8192 (+1% non-causal, +6% causal). With graph replay the call beats SDPA at five of six shapes, by up to 1.50×.

**Status:** confirmed (reproduced twice). F022 corrected the "15–18% on causal shapes" summary: it covers two shapes, and at N=4096 causal the kernel is 5% slower (1.053×) where this file lists 1.03×.

### F016 — What the incumbent actually does

**Question.** How does PR #368's gfx12 kernel feed LDS, and is software pipelining with an async global-to-LDS copy possible on gfx1201?

**Result.** Assembling with `llvm-mc` shows that both `buffer_load ... lds` and `global_load_lds_dword` are rejected on gfx1201, and the incumbent's source has 0 matches for either. Its mechanism is `global_load_tr_b128` transpose loads plus lane-major LDS staging, with no async copy. The explanation given in F010 for the causal result is withdrawn.

**Status:** confirmed. Corrects F010.

### F022 — Baseline audit

**Question.** Is torch SDPA the strongest reachable attention baseline on gfx1201, and do the reported SDPA wins survive?

**Result.** The four SDPA variants are the same AOTriton `attn_fwd` kernel. At N=8192 non-causal the strongest SDPA moves the ratio from 1.459× quoted to 1.456× (1.478× measured), a correction of 0.2 percentage points. All five SDPA wins survive (1.100×, 1.324×, 1.406×, 1.478×, 1.699×) and the single loss deepens to 0.964× at N=2048 non-causal. N=4096 causal against PR #368 is 1.053× (about 5% slower). Because AOTriton has no fp8 keys, the comparison is fp8 against fp16, which any headline must state.

**Status:** confirmed. Corrects F015's causal summary.

## 2. Levers on the Triton loop

The Triton kernel was then tried against every lever the project could name: instruction counts, LDS layout, loop structure, scheduling hints, inline assembly and compiler settings. Most were closed by measurement or by reading the generated code. The one lever that converted to wall time was splitting the causal loop into two ranges (F025, F029, F030, F038).

### F017 — Removing P-conversion instructions gains nothing

**Question.** Does removing the reshape, split and interleave tail of the fp32-to-fp8 P conversion speed up the Triton kernel?

**Result.** `tl.reshape` and `tl.split` are free (byte-identical loop body); `tl.interleave` costs 66 `v_perm_b32` per iteration against a floor of 64. Extracting bytes before the split (PCAST=2) leaves the loop at 575 instructions and is within noise, at 1.011 / 1.008 / 1.010 against PCAST=1 in three replications (inside the 4–7% spread), and deleting the whole 78-instruction conversion tail (loop 575 → 497, an upper bound) is stated to buy 0% of wall time. A software cast (PCAST=0) is 1.48× slower than the hardware pack.

**Status:** closed (a negative result; instruction count of the P conversion is not the limiter).

### F018 — LDS capacity is not the non-causal limiter

**Question.** Does LDS capacity, through occupancy, explain the non-causal deficit against PR #368 on gfx1201?

**Result.** Taking LDS as a 64 KB per-workgroup limit that caps the kernel at 4 waves/SIMD, doubling occupancy still makes it slower: 4 waves/SIMD beats 8 by 1.1% (128×16 w4 against 64×16 w2), and the 7- and 8-wave configurations reach 0.38–0.64 of PR #368's throughput against 0.79 for the shipped tile. The earlier `num_stages` puzzle is solved: stage 3 loses to spills (256 registers and 48 spills against 235 and 0 at stage 1), and the earlier "ISA-identical" row was a transcription error. Its `v_fma` column is void (a whole-kernel count; the loop body has 18 FMA-family operations).

**Status:** refuted (the LDS-capacity hypothesis). S30 later showed that the premise "LDS binds at 4 waves" was an artifact of a misreported LDS pool: the VGPR file binds at 6 waves/SIMD32 and the effective LDS is 131 072 B per WGP.

### F019 — Non-MMA loop cost

**Question.** Can removing non-MMA loop instructions (`psc`, `adefer`, `qfold`, `vtrans`) beat the incumbent at non-causal shapes?

**Result.** As first written, `psc` gave −48 loop instructions and a 3.7–3.8% wall gain at N=8192, `adefer` removed 96 instructions but gave 1.00× and has a 34744× error, and `vtrans` failed its gate. The `psc` argument was later found to have been accepted and never forwarded, so every `psc` row is a same-binary self-comparison; its gain and its "bit-identical" claim are void. What stands: `vtrans` fails its gate, `adefer` is numerically broken, and instruction count does not order wall time.

**Status:** refuted (the central `psc` result is void). Corrected by F023.

### F020 — The first HIP kernel: correct, but 11.3× slower

**Question.** Once its non-determinism is fixed, is a hand-written HIP fp8 flash-attention kernel competitive with PR #368 or the Triton kernel?

**Result.** Non-determinism was fixed (run-to-run difference exactly 0) and the correctness gate passes, but at N=8192 non-causal the kernel is 11.3× slower than PR #368 and 8.9× slower than the Triton fp8 kernel. The cause is 256 VGPRs, the architectural maximum, with 464 spills. Stopping the PV loop from being fully unrolled (`#pragma unroll 1`) cut spills to 63 and bought 2.81× over its own earlier version, but the kernel still sits at the VGPR ceiling with 54784 B of LDS per workgroup, one workgroup per CU.

**Status:** closed ("a redesign, not a tweak"). The project returned to hand-written HIP with a different design in the SK1 kernel (tracks H6–H9).

### F021 — IEEE division cost, and a scheduling hint

**Question.** Does removing the 1361-instruction IEEE `fdiv` sequence give the ~15% the static count suggests, and are two free compiler levers worth enabling?

**Result.** The division fix is live as a default (`FDIV=1`): text 6392 → 2355 instructions, division sequence 1361 → 171, `s_delay_alu` 1072 → 110. The wall-clock gain is only 0.85–3.3% (0.0% at N=8192 causal) because all 1361 instructions were in the prologue and epilogue. `schedule_hint="attention"` is the larger lever, 0.946 of the time at N=8192 for both causal flags and a gain at five of six shapes (5–11% in the file's body, 5–13% in its summary), but it regresses N=4096 causal by 3.7% on the auto tile and needs a tile guard. A quoted "2.2× better" non-causal accuracy did not reproduce: ours/incumbent (`smooth_k=False`) is 0.85, 0.93 and 0.98 at N=2048, 4096 and 8192.

**Status:** shipped (the division fix); the hint was measured but not made a default. Corrects the replacement claim in F014.

### F023 — `psc` and `schedule_hint`: the gap did not close

**Question.** Do `psc` and `schedule_hint` close the non-causal gap to PR #368, and can an ISA predicate decide when the hint is safe?

**Result.** The best non-causal configuration (`schedule_hint="attention"`, shipped kernel, N=8192) is 1.2169× slower than PR #368 measured in the same process, so the gap did not close. `psc` had never actually been applied in F019; applied correctly it is worth about ±1.5% with an unstable sign. `schedule_hint` gives a real, bit-identical 4.8% at N=8192 non-causal. The best ISA-derived guard predicate scores 4 of 6, and the recommendation is to gate the hint on `BLOCK_N == 16`.

**Status:** confirmed (the gap did not close). Corrects F019.

### F024 — LDS bank conflicts and barriers

**Question.** Are LDS bank conflicts or the barrier structure behind the non-causal gap to PR #368?

**Result.** The conflicts are real and exactly 2.00× on all 64 gathers, but removing them (a transposed V load, 64 → 0 `ds_load_u8`) makes the kernel 7.9% / 9.5% slower: other waves hide the LDS-pipe cost. `psc` in the shipped kernel is worth 1.7%; an earlier 4.7% lead came from a codegen restructuring (1266 → 3025 instructions) and is withdrawn. The barrier structure (4 rounds per iteration against the incumbent's 2) is left as the sole surviving candidate.

**Status:** closed (real, but hidden).

### F025 — `PRE_LOAD_V` and the two-range loop split

**Question.** Does AOTriton's `PRE_LOAD_V`, or splitting the causal loop into two ranges, speed up the fp8 kernel?

**Result.** `PRE_LOAD_V` reaches the ISA but is 1.016×, i.e. 1.6% slower, at N=8192 non-causal (kernel-only). `SPLIT_LOOP=1` is a real causal win: 1.60× / 1.52× / 1.28× speedup at N=2048 / 4096 / 8192, bit-identical at N=8192, taking the causal ratio against PR #368 from 0.83× / 0.87× / 0.86× to 0.57× / 0.76× / 0.65× at those three shapes (below 1 is faster). Non-causally the split changes nothing.

**Status:** confirmed. The split numbers were re-measured in F029, F030 and F038.

### F026 — Reaching the incumbent's mechanisms through inline asm

**Question.** Can the incumbent's `global_load_tr` and its row-distributed rescale be reached from Triton through inline asm?

**Result.** `global_load_tr` is unreachable from Triton 3.7.1: every 64-bit operand variant aborted the compiler, including a control that used the native `global_load_b128`. The row-distributed rescale is a layout trade, not a missing mechanism. A later analysis found the stated reason for the abort was wrong (the aborts were LLVM constraint-count mismatches plus one unknown-constraint error); the conclusion that the instruction is unreachable stands.

**Status:** closed.

### F029 — `SPLIT_LOOP` benchmark

**Question.** How much does splitting the KV loop into two ranges speed up the causal kernel at each shape, and does it affect non-causal?

**Result.** Causal speedup (base time over split time, on the variant skeleton) is 1.60× at N=2048, 1.51× at N=4096 and 1.34× at N=8192. Against PR #368 the ratio moves from 1.62× to 1.01× at N=2048, from 1.58× to 1.03× at N=4096 and from 0.84× to 0.63× at N=8192 (below 1 is faster). Non-causal, `SPLIT_LOOP=0` and `=1` compile to the same AMDGCN, so there is no effect.

**Status:** confirmed. F030 found that the HIP event-pair timer under-reports with batch size, so the N=2048 and N=4096 parity claims here rest on an under-reported incumbent time.

### F030 — Porting `SPLIT_LOOP` into the shipped kernel

**Question.** Does the causal split-loop win survive the port into the shipped kernel?

**Result.** Yes, but smaller than on the variant skeleton at every shape, as pre-registered: 1.32–1.40× at N=2048, 1.35–1.46× at N=4096 and 1.26–1.29× at N=8192. Causally the kernel is now 1.1–1.75× faster than PR #368, where before the port it was 1.25× slower. The finding also showed the HIP event-pair timer under-reports as the batch size grows (wall/event 1–7% at `inner=10`, 1.4–4.1 at `inner=200–500`), which invalidates any table taken at `inner` ≥ 200.

**Status:** shipped. Corrects F029. F038 later re-derived the speedups, and N=4096 shrank to about 1.33×.

### F031 — Swapping the operand order (K·Qᵀ)

**Question.** Does computing S = K·Qᵀ avoid a transpose of P in the PV GEMM?

**Result.** No: it moves the transpose from P (once per KV iteration) to O (once, in the epilogue), as FlyDSL does. A compile-only census at six shapes shows 256 registers at all six (the cap, against 182–240 before), spills of 32–112 at five of six (against 0), twice the shared memory and a loop 22–116% longer. No timing was taken.

**Status:** closed (a loss by construction).

### F032 — The in-thread transpose setting

**Question.** Does `TRITON_HIP_USE_IN_THREAD_TRANSPOSE=1` change the compiled kernel on gfx1201, and what does it do to the PV-operand LDS byte gathers?

**Result.** Compile-only, nothing launched. At 4 of the 6 shipped shapes it drives the PV `ds_load_u8` count to exactly 0; the loop body shrinks 1182 → 701 (−40.7%), 648 → 470 (−27.5%) and 557 → 427 (−23.3%) with no new spills. At the N=8192 tile (128×16, 4 warps, causal and non-causal) the binary is byte-identical, because the pass does not fire.

**Status:** confirmed (the codegen change). The speed question was answered by F035 (null).

### F033 — Barrier census and the global V-load width

**Question.** Does a per-loop barrier-count difference explain the causal/non-causal asymmetry, and is the global V load already full width?

**Result.** The barrier hypothesis is dead: at the shipped default, causal and non-causal both emit exactly 4 × `s_barrier_signal` and 4 × `s_barrier_wait` per loop body at every shape. In the `SPLIT_LOOP=1` build, the faster one, the count is higher (14/14 against 4/4 at N=2048 and N=4096, 8/8 against 4/4 at N=8192). The global V load is `buffer_load_b128` only, 2048 B per iteration, which is exactly the V tile (ratio 1.000×) and coalesced.

**Status:** closed (barriers do not explain the asymmetry; the V load is already optimal).

### F035 — The in-thread transpose, timed

**Question.** Is the in-thread-transpose setting faster at run time, and is it numerically identical?

**Result.** Null. The corrected ITT/baseline time ratios are 0.9925 / 0.9932 / 0.9948 (event) and 0.9869 / 0.9847 / 0.9902 (wall) at three shapes, about 0.65% from 1 and inside the benchmark's own bracket (0.9960–1.0175×); the file's headline calls this "0.65% slower", which does not match the sign of its ratios. Output is bit-identical to the baseline at all four shapes tested. An earlier cross-process design had shown 13% faster; that was a measurement-order artifact, with the same baseline binary measuring differently after 4 s and 22.6 s of warm-up.

**Status:** confirmed (null). F038 did not reproduce this file's stated mechanism for the artifact (a boost decay) and found a queue-state effect instead.

### F037 — Emulated `exp2`

**Question.** What does replacing the hardware `exp2` with an exponent split plus a polynomial do to the emitted code?

**Result.** Codegen census only; nothing was launched or timed. With the emulation (degree 3) at N=8192 non-causal, `v_exp_f32` goes 18 → 0, `v_dual_mul_f32` 26 → 100 and the loop 551 → 561 (+1.8%) with 0 spills; at N=2048 and N=4096 non-causal the loop grows 557 → 764 and 865 (+37…+55%) with 32 spills. The prediction that a pairable split would dual-issue better missed (93 against 100 dual-issue multiplies), so the floor split stayed the default.

**Status:** open (codegen measured, run time never timed).

### F038 — Re-verifying the `SPLIT_LOOP` win

**Question.** Does the shipped causal speedup survive re-measurement with the warm-up defect fixed?

**Result.** Yes at all three causal shapes, and one number shrinks: N=2048 1.37–1.40× → 1.33–1.38×; N=4096 1.38× → 1.30–1.36× (pooled about 1.33×); N=8192 1.27× → 1.29–1.31×. The corrected headline is that `SPLIT_LOOP` is a 1.29–1.38× causal win. PR #368 was not measured here, so the earlier "1.13–1.75× faster than PR #368" is not re-confirmed.

**Status:** confirmed. Corrects F030 (N=4096 down, N=8192 slightly up) and F035's stated mechanism.

## 3. Build, driver and clock environment

Findings about the machine rather than the kernel: the first wheel (F027), GPU crashes and wedges (F028, F034), and a clock-state problem that voided a day of absolute timings but not the ratios (F073). They are the reason this catalogue quotes ratios.

### F027 — Building and verifying the wheel

**Question.** Can the gfx12 SageAttention port be built into a verified wheel that works on gfx1201?

**Result.** The wheel `sageattention-2.2.0+amd.gfx12.1-cp312-cp312-win_amd64.whl` was built and passed 22/22 checks. A real ComfyUI-length call at N=16384 ran with max abs error 3.5554e-03. A static ABI check found the project's own extension compatible (`torch_cpu.dll` 53/53, `c10.dll` 63/63), while the third-party wheel is missing 1 of 65 `c10` symbols, which corrects F001's count of 26.

**Status:** shipped (the wheel). Corrects F001.

### F028 — Driver crash forensics

**Question.** Was the GPU wedge a display crash, and can it be attributed to the project's kernels?

**Result.** The machine had bugchecked with `0x116` VIDEO_TDR_ERROR nine times since 2026-07-08, with zero Display 4101–4104 events: timeout recovery had never succeeded. Attribution to the project's kernels is not established: no Windows-side fault record ties a kernel launch to a wedge. The health probe had returned "ok" on a wedged device (a false pass) and was fixed.

**Status:** confirmed (a display crash, not attributable to the kernels).

### F034 — The gfx1201 wedge reproduces from compiles alone

**Question.** What triggers the device wedge seen even when no kernel is launched?

**Result.** With only Triton meta-tensor compiles, three compiles were fine, the fourth gave HIP error 719, and a second run wedged at the identical point. The trigger is cumulative per-process state, the number of compiles already done, not any particular shape. The mitigation is one compile, or a small batch, per fresh process.

**Status:** confirmed.

### F073 — The clock-state probe: absolutes invalid, ratios validated

**Question.** Were a day's timings affected by a mis-set GPU clock, and which results survive?

**Result.** The GPU had been running underclocked and undervolted. Its effective clock depends on workload duration (a 4096³ fp16 matmul implies 2.817 GHz, a 16384³ one 1.873 GHz). Absolute milliseconds from that day were inflated by about 12–13% and are void; ratios moved by less than 1%: ours/production 0.8600 → 0.8607, ours/PR #368 0.9600 → 0.9540, PR #368/production 0.8959 → 0.9022, and F072's 1.0010 → 0.9974. The comparators were flat against the earlier record while ours moved −22.5%, which turned F071's inference about an arm-specific change into a measurement.

**Status:** confirmed. It voids the absolute times in F070–F072 and confirms their ratios. The rule adopted: probe the clock before and after every block.

## 4. INT4, sparsity and other routes that were costed first

Routes that promised a large step and were tested or costed before being built: INT4 WMMA, a kernel-override route to INT4 instructions, and block-sparse attention. Each ended in a decision, and the INT4 direction was dropped.

### F036 — INT4 WMMA instruction rate on gfx1201

**Question.** Does `v_wmma_i32_16x16x32_iu4` deliver more MACs per second than the fp8 WMMA path?

**Result.** It issues at the same instruction rate as the fp8 WMMA (49.28 against 48.19 G WMMA/s), so it delivers 2.0× the MACs per second (ratio 2.020). The column labelled "TOPS" (372.0 for IU4, 184.2 for the fp8 control) is in tera-MAC/s; in AMD's convention these are 744.0 and 368.4 (F039). The file still carries two statements later shown wrong: that `v_wmma_i32_16x16x16_iu8` is unavailable (it assembles), and an unresolved discrepancy with F002 (resolved in F039).

**Status:** confirmed (the 2.0× rate). Units corrected by F039; the sign-selector detail corrected by F045.

### F039 — INT4 verdict, and SageAttention 3 on gfx1201

**Question.** Is the INT4 WMMA lever worth pursuing given F036's numbers, and can SageAttention 3 (FP4) be expressed on gfx1201?

**Result.** The verdict was PURSUE. F036's "TOPS" column was in tera-MAC/s; in AMD's convention F036 measured IU4 at 744.0 TOPS, 95.6% of the 778.6 line, and the fp8 control at 368.4, 94.6% of AMD's published dense FP8 389.3, while F002's Triton loop reached 195.6 (50.3%). The earlier discrepancy of about 2× is a unit-and-count artefact. gfx1201 has no FP4 or microscaling instruction (0 matches in AMD's RDNA4 ISA XML), so SageAttention 3 (NVFP4) cannot be expressed.

**Status:** closed. F089 later decided to drop the INT4 direction without retracting this finding's ceiling result; the unit correction and the SageAttention 3 conclusion stand.

### F045 — An INT4 QK kernel against fp8: pre-registered

**Question.** Would an INT4 WMMA QK kernel be about 2× faster than a matched fp8 QK kernel at the real shape, and at what accuracy cost?

**Result.** The kernel was never compiled or run. A GPU-free simulation gave softmax total variation of 0.0133 / 0.0640 / 0.2275 for per-row INT4 on uniform, normal and heavy-tailed inputs against 0.0047 / 0.0142 / 0.0267 for fp8, which is 2.80× / 4.51× / 8.52× the fp8 value (2.85× / 4.08× / 7.08× for the best INT4 variant), and a minimum argmax survival of 69.2% against 94.8%. The end-to-end ceiling from a 2× faster QK is about 1.33×.

**Status:** open (never run; F089 later dropped the INT4 direction). Corrects F036's description of the operand sign selectors.

### F049 — Reaching `iu4` through `TRITON_KERNEL_OVERRIDE`

**Question.** Can `TRITON_KERNEL_OVERRIDE` carry the `iu4` WMMA instruction into a gfx1201 binary, via the `.llir` or the `.amdgcn` stage?

**Result.** Reachable through the `.amdgcn` stage: `v_wmma_i32_16x16x32_iu4` assembled and linked into a gfx1201 code object with an encoding byte-identical to F036's. The `.llir` stage is blocked at the IR parser ("expected number in address space"). The binary was never executed, and no speed claim is made.

**Status:** confirmed (the route exists; nothing was run).

### F075 — Block sparsity on real Krea2 attention

**Question.** Is real Krea2 attention sparse enough at L=1056 to justify a block-sparse kernel?

**Result.** No. Layer 26 is dense: at τ=0.99 it needs 97.0% of K/V blocks, and even the per-key oracle needs 70.6%, which fired the pre-registered hard stop (oracle above about 40%). Layer 0 is sparse (18.2% of blocks retain 90% of the mass, 42.4% retain 99%), but 8 of 48 heads are essentially one-hot, and the text-sink explanation was refuted. The average is 69.7%, MARGINAL, so a block-sparse kernel is not to be built.

**Status:** refuted (the sparsity lever). F090 re-ran it at the production length.

### F089 — Build or drop the INT4 QK direction

**Question.** Should INT4 QK be built, or the direction dropped (F039 had said PURSUE)?

**Result.** Drop, with a stated confidence of about 0.7. A build would give only a QK-only ceiling on a kernel with no shippable integration path; the end-to-end ceiling is at most 1.33×; and the accuracy cost is already priced at 2.85× / 4.08× / 7.08× the fp8 softmax total variation for the best INT4 variant, with argmax survival of 69.2% against 94.8%. The INT4 rate itself (2.020× at the instruction) is real, so the decision rests on integration and accuracy.

**Status:** closed. Overturns F039's PURSUE direction (its ceiling finding is not retracted); F045 stays unrun.

### F090 — F075's block-sparsity test at the production length

**Question.** Does F075's block-sparsity falsification hold at L=8771?

**Result.** The aggregate verdict is unchanged, MARGINAL: 64.2% of K/V blocks at τ=0.99, against 69.7% at L=1056. Layer 26 is still a hard stop (95.9%, was 97.0%); layer 0 improves to 32.4% (was 42.4%). Only 2 of 28 layers and one timestep were tested, so the short-N-artifact explanation is falsified for those two layers only.

**Status:** confirmed (the marginal verdict holds; do not build a block-sparse kernel).

## 5. The production shape and the gap to PR #368

Krea2's attention runs at (1,48,8771,128), non-causal, so the project measured the gap to PR #368 at that shape, audited the incumbent's kernel, and tested what might explain the difference. The headline moved several times as the kernel and the measurements improved. Where a later entry reverses an earlier one, both entries say so.

### F040 — Fusing K/V staging: pre-registered, and advised against

**Question.** Would fusing the K/V staging in the non-causal kernel remove barriers and close the roughly 1.20× non-causal deficit?

**Result.** Pre-registered but not recommended: 5 of 6 routes were closed from source, and the estimated probability that the rest fails was 0.85–0.90. Triton places barriers greedily per store site, so a rewrite does not remove them. The file's §9.3 recommended a census and an A/B at the real shape (1,48,8771,128) instead; that was run as F041.

**Status:** open (a frozen design that was never run).

### F041 — The real-shape deficit at Krea2's (1,48,8771,128)

**Question.** Does the roughly 1.20× non-causal deficit against PR #368 exist at the shape Krea2 actually runs?

**Result.** Yes. Kernel-only ours/PR #368 is 1.2241× on the event timer and 1.2160× on wall time (1.2250× and 1.2202× bracket-corrected; above 1 means ours is slower), a signal 24.17× the artifact floor. It is 1.224×, not the 1.05× that would have meant the earlier shape was unrepresentative. Causally at this shape the shipped split-loop arm beats the incumbent by 1.635× and the unshipped default only by 1.066×.

**Status:** confirmed. It executes F040's §9.3. F071 later reversed the headline after the kernel changed.

### F042 — The online-softmax rescale: a GPU-free census

**Question.** Can the output-accumulator rescale be skipped behind a headroom branch, as in the external HIP kernel?

**Result.** A static census of cached binaries finds 60–65 in-place multiplies plus one alpha `exp` per loop iteration, 5.5–11.1% of the loop; at the real shape (1,48,8771,128) non-causal it is 65 + 6 = 71 of 670 instructions, 10.6%. Every measured loop body has a single branch, the back-edge, so the rescale is unconditional. The file recommended running no A/B.

**Status:** closed. S31 later overturned the premise that Triton cannot emit a conditional skip, but confirmed the conclusion that skipping does not pay.

### F043 — Porting another project's int8-attention optimisations

**Question.** Which of four int8-attention optimisations from comfy-kitchen PR #194 carry over to the Triton fp8 kernel, and with what effect?

**Result.** Design and a frozen pre-registration only; no GPU work. The `exp2` builtin is already used (effect exactly 0). The softmax-headroom branch is expressible but structurally inert (predicted null, 0.0 to −0.5%). The int32 bias trick cannot exist here, and 16-byte granularity is already exceeded. A `sched_barrier` hint is already reachable and the most promising, but it was never measured at the real shape.

**Status:** open (frozen, never run). F042 and S31 later addressed the headroom branch.

### F044 — Re-baselining against stock SageAttention 1.0.6

**Question.** How do ours and the PR #368 fork compare with stock SageAttention 1.0.6 at H=48 and N=8771?

**Result.** Frozen but unexecuted. Stock 1.0.6 is pure Python (no extensions), so no compilation is needed. The prediction was that our non-causal deficit against stock would be materially smaller than 1.20× (60%) or at or below parity (40%), and it would be falsified if ours/stock ≥ 1.20×. Nothing was measured here.

**Status:** open. F047 corrected one of its citations, and F051 found that its "fork" arms were actually the production package.

### F046 — Register live-range census against an external study

**Question.** Does the kernel suffer the producer/consumer register pressure reported in an external study of a gfx1201 attention kernel (ROCm/rocm-libraries issue #11055)?

**Result.** No. The non-causal loop uses 232 VGPRs with 0 spills, 0 B private and 16 384 B shared at (1,48,8771,128), against 231 VGPRs for the external combined kernel. The causal real shape needs 256 VGPRs with 16 spills (`SPLIT_LOOP` brings it to 249 and 0), but the file treats the causal path as dead code in production.

**Status:** closed (zero spills at 232 registers).

### F047 — Which incumbent kernel produced the headline?

**Question.** Was the headline comparison measured on the fork's intended HND path, or on a fallback?

**Result.** Cleared. The headline harnesses call the native function with `tensor_layout="HND"`, the dispatch census shows `qk_rawq_int8_sv_f8_scaled_native_attn` ran and `sage_fp8_nhd_short_mha` did not, and the incumbent does not mutate `k`. One residual remains: if the NHD wrapper were faster than the HND path, the honest deficit would be larger than 1.20×.

**Status:** confirmed. Corrects a citation and a premise in F044.

### F050 — Production against the fork against ours, at the real shape

**Question.** Is the roughly 1.22× deficit against the PR #368 fork also the gap to the production SageAttention that ComfyUI actually runs?

**Result.** One process, non-causal (1,48,8771,128), five arms; ours is measured kernel-only while the other arms are full calls, which favours ours. Ours/PR #368 is 1.2303× (event) and 1.2204× (wall), reproducing F041's 1.2241× to 0.5%. Ours/production is 1.1096× and 1.1094× bracket-corrected: production is faster than ours. The pre-registered bar production/PR #368 ≥ 1.15× was missed (1.1089×), so the operand-bound mechanism is attenuated, neither confirmed nor refuted.

**Status:** confirmed for the kernel as it stood then. F071 reversed these ratios after the kernel changed.

### F051 — The first compliant in-process four-arm comparison

**Question.** At the real production shape, in one process, how do our kernel, stock SageAttention 1.0.6, torch SDPA and the production package compare?

**Result.** Non-causal (1,48,8771,128), ours kernel-only against full calls for the other arms (the same design as F050): ours/stock 1.0.6 is 0.7017× (we win 1.425×), ours/SDPA is 0.7164× (we win 1.396×), ours/production is 1.1094× bracket-corrected (we lose 1.11×), and production/stock 1.0.6 is 0.6339× (production wins 1.5775×). An arm first labelled "PR #368 fork" was actually the production package, and the label was corrected the same day.

**Status:** confirmed. Corrects F044's fork arms.

### F052 — Tile-width sweep at the real shape

**Question.** Is `BLOCK_N` the cause of the roughly 1.23× deficit against the fork?

**Result.** No, and the hypothesis fails in the opposite direction: the 128×32 tile is 1.1406× slower and 128×64 is 3.5821× slower than the shipped 128×16 (4 warps). The register census is 232 registers and 0 spills for 128×16, 256 and 40 spills for 128×32, and 256 and 216 spills for 128×64; the slowdown tracks the spills. Predictions P1 to P3 were refuted (P4 was not tested), although the file's header says all four.

**Status:** refuted (the shipped tile is already the best). S28 later confirmed this at long N.

### F053 — What the fork's HIP kernel does that ours does not

**Question.** What does the PR #368 fork's HIP kernel do structurally that the Triton kernel does not?

**Result.** A source-only audit finds three asymmetries: 4× the work per iteration (BC=64 with 128 threads gives 138 iterations at N=8771 against 549 for ours, 128 `wmma` per iteration against 32); no pipelining (no async copy, no `s_waitcnt`); and reuse of the A operand across two query groups. The LDS (about 19 464 B) and a VGPR lower bound (≥141) are source estimates.

**Status:** confirmed (a structural reading; no performance implication was established).

### F054 — The fork kernel's resources, from code-object metadata

**Question.** What are the fork kernel's real VGPR, spill, LDS and occupancy figures?

**Result.** For the 2q instance (BC=64, HD=128, BR=128, VT=1, non-causal) the metadata gives 256 VGPRs, 29 spilled (96 B per lane), 19 456 B of LDS and 128 threads, an occupancy of 3.0 waves/SIMD against 4.0 for ours (232 VGPRs, 0 spills, 16 384 B).

**Status:** refuted. A later census (S27) showed that this finding and F055 examined an instantiation the dispatch does not launch.

### F055 — Instruction census: the fork's loop against ours

**Question.** Does the fork's loop body exceed about 167 instructions per column, the falsifier set in F053?

**Result.** On the instantiation it examined, the fork loop was 3 389 instructions and 128 `v_wmma` over 64 columns, ours 670 and 32 over 16. `v_wmma` per column is 2.00 on both; instructions per `v_wmma` are 26.48 for the fork and 20.94 for ours, i.e. ours 1.26× better. S27 later found this was the wrong instantiation: for the launched one the fork's loop is 19.7 instructions per column against ours at 24.8, so the fork is 1.26× leaner.

**Status:** refuted (the comparison direction reversed; see S27).

### F056 — The production Triton loop taken apart

**Question.** Why is production SageAttention, built with the same compiler, 1.109× faster than ours?

**Result.** Production's loop is 694 instructions against ours at 670; instructions per `v_wmma` are 10.84 against 20.94 (ours 1.93× worse). Both emit 8 `s_barrier` per iteration with the same 16 384 B shared. It concluded that barrier count is not the differentiator and pointed at the fp8 P·V pack and shuffle composition.

**Status:** refuted. Its two central conclusions were normalisation errors (raw per-iteration counts compared across BLOCK_N=32 and 16 tiles), corrected by F057.

### F057 — Per-column recount, and registers

**Question.** After per-column normalisation, are F056's conclusions right, and what do the registers show?

**Result.** Per column both kernels pay equal barriers (0.250 `s_barrier_signal` per column), but ours runs twice the iterations: 4 392 barriers per workgroup against 2 200. At 128×32 with 4 warps, production uses 256 VGPRs with 2 spills and 12 B private; the file gives ours as 256 with 57 spills and 232 B, a figure F059 later corrected to 40 spills. The extra multiplies trace to the per-token-scale outer product, a speed/accuracy trade.

**Status:** confirmed. Corrects F056; its own spill figure for ours was corrected by F059.

### F059 — Full-call timing, and the tile-width trade

**Question.** Does the quantization prologue explain ours losing to PR #368 end to end, and is the tile width limited by per-token scales?

**Result.** Full call with both arms quantizing inside the timed region, N=8771, H=48: non-causal ours/PR #368 is 1.2708× (a loss) against 1.2303× kernel-only, so the prologue adds only +0.04×; causal is 0.6700× (a 1.49× win) full call against a 1.635× win kernel-only for the shipped split-loop arm (ours 8.9026 ms against PR #368's 14.5540 ms), so counting the prologue reduced the causal win. The 0.9401× that the first table gives as kernel-only is the unshipped non-split arm. Spills at the 128×32 tile with 4 warps: production 2, ours 40 (corrected from 57, which came from the wrong kernel), against 0 at 128×16 and 216 at 128×64. That per-token scales cause the spills is a hypothesis; the ablation was not run.

**Status:** confirmed (the prologue is not the main cost). Corrects F057's spill count for ours.

### F070 — `schedule_hint` at the real shape

**Question.** Does `schedule_hint="attention"` speed up the shipped kernel at the real shape (1,48,8771,128), non-causal?

**Result.** Null. The hinted/unhinted ratio is 1.00024 raw and 0.99999 bracket-corrected with bit-identical output, and the hinted kernel's AMDGCN contains no scheduling-barrier instructions at all. An unplanned finding: the shipped kernel measured very differently from the earlier record (a cross-process gap, not a result), so earlier real-shape ratios were of unknown validity; F071 and F073 resolved this.

**Status:** closed (the lever is null at this shape).

### F071 — The real-shape baseline, re-measured

**Question.** With the current kernel in one process, how do ours, PR #368, production SageAttention and SDPA compare at (1,48,8771,128) non-causal?

**Result.** Ours/PR #368 is 0.9600× and ours/production 0.8600× (bracket-corrected), against 1.2303× and 1.1094× recorded earlier in F050. PR #368/production barely moved (0.9017 to 0.8959, bracket-corrected), so the change is specific to our kernel (1.2303 / 0.9600 = 1.2816, ours about 22% faster relative to PR #368). Ours is kernel-only here while PR #368 and production are full calls.

**Status:** confirmed (the ratios, re-validated in F073 as 0.9540 and 0.8607). Overturns F050's loss ratios. F072 showed the kernel-only edge over PR #368 was the prologue.

### F072 — A like-for-like full call at the real shape

**Question.** When every arm pays its own prologue, how do ours and PR #368 compare at (1,48,8771,128) non-causal?

**Result.** Ours/PR #368 is 0.9993 raw and 1.0010 bracket-corrected: a tie. F071's roughly 4% kernel-only edge was the prologue (about 0.754 ms, 4.2% of the call). Bridged ours/production is 0.8968 (about 10% faster) and ours/SDPA 0.5838 (1.71× faster). At default clocks (F073) ours/PR #368 is 0.9974.

**Status:** confirmed. Corrects the framing of F071.

## 6. Accuracy: per-token fp8, `smooth_k` and int8 QK

Accuracy work on real activations captured from Krea2 and Flux2-Klein: how per-token fp8 compares with PR #368's per-block int8 K, what `smooth_k` buys, and what an int8-QK variant costs and gains. The earliest entries score torch re-implementations of the schemes; later ones score the kernel itself.

### F058 — Per-token fp8 against PR #368 on real activations

**Question.** How do per-token fp8 (ours) and PR #368's per-block int8 K compare in softmax error on real Krea2 and Flux2-Klein Q/K/V?

**Result.** The picture is split, not a win. PR #368 is more accurate at Krea2 block 26 by 14–27× at every latent scale, and on Klein (by 59× and 101× at two blocks, which the file says to treat as upper bounds). Ours wins at Krea2 block 0, by up to 55× at one setting, but that win is regime-fragile: PR #368 wins there by 5× at ×0.25 latent scale. Ours is flat across scales (KL 3.7e-3 to 4.7e-3); worst-case KL is 4.7e-3 for ours against 1.8e-1 for PR #368. The schemes are fp32 torch re-implementations, not the shipped kernels, and the text context is a random surrogate of the right width.

**Status:** confirmed. F061 later showed the `dlogit` magnitudes were overstated by 1.09–3.17×; the rankings survive.

### F061 — Per-token int8 Q/K against fp8 and PR #368

**Question.** Does per-token int8 QK land near PR #368 on benign captures, and at or below ours on Krea2 block 0?

**Result.** The headline was corrected on 2026-09-27. The arm pre-registered without mean subtraction (`int8tok`) fails on all 10 captures: 2.8–7.1× worse than PR #368 on the 6 benign captures and 39–54× worse than ours at Krea2 block 0. The file's own arm `int8tok_smseq` (the `smooth_k` mean removed first, then per-token int8) beats PR #368 on 7 of 10 (ties 1, loses 3 by 1.1–1.8×) and beats ours on 10 of 10, with worst-case KL 1.51e-3 against 5.18e-3 for ours and 1.81e-1 for PR #368.

**Status:** confirmed (the corrected headline). Corrects F058; it led to the int8-QK work in F081 and F083.

### F063 — `smooth_k` accuracy on real captures

**Question.** Is `smooth_k` (removing K's sequence-axis mean) free accuracy for the per-token fp8 scheme?

**Result.** It improves ours on 10 of 10 real captures, by 1.44× to 20.88× (geometric mean 3.19×); worst-case KL falls from 5.175e-3 to 3.022e-3 (1.71×). In exact arithmetic it changes nothing (KL ≤ 1.86e-6 against the uncentred reference). It does not make ours the best scheme: per-token int8 with `smooth_k` still wins.

**Status:** confirmed. Its end-to-end gain was found to depend on scale in F065 and F066.

### F081 — The shipping decision: quantization split and toolchain

**Question.** Can int8 and fp8 both ship, should the kernel be rewritten in HIP, and is the causal win sellable?

**Result.** The recommendation was to ship int8 QK with fp8 PV as one kernel (int8's relative L1 error is 4.1× lower than e4m3 for QK; int8 PV collapses to a worst-case cosine similarity of 19.52% against 96.70% for e4m3), to adopt `smooth_k`, and not to rewrite in HIP, citing one direct HIP-against-Triton attention comparison on gfx1201 in which Triton was about 32% faster (103.4 against 136.3 ms). The causal claim is narrower than assumed: autoregressive video models are block-causal and their speed-ups come from the KV cache, while Wan, LTX-2 and MiniMax-H3 are non-causal.

**Status:** open (a decision record). The int8 split was built in F083 and kept opt-in, and the cosine-similarity evidence it cited was found invalid there. The project later wrote the hand-written HIP kernel this entry advised against (SK1, tracks H6 onward).

### F082 — `smooth_k` reaches the shipped kernel

**Question.** Can `smooth_k`, implemented earlier but unreachable, be reached from `flash_attn_fp8`, is the shipped path unchanged, and does it help on real captures?

**Result.** It is wired into the fused K/V path, and with it off the compiled kernel is the identical 835-instruction binary. On 8 real Krea2 and Klein captures the output rms improved on 7 of 8: median +18.35%, minimum −6.97%, maximum +24.76%. It is not free: +41.9% to +94.6% of the prologue (ratios only; clock caveat).

**Status:** shipped (as an option in the kernel; off by default at this point).

### F083 — int8 QK built and scored (the 8+8 split)

**Question.** Does an int8-QK with fp8-PV path in `flash_attn_fp8` leave the fp8 path untouched and improve accuracy?

**Result.** The identity gate holds: with the flag off, the 835-instruction prologue and the 2290-instruction attention kernel are identical and outputs are `torch.equal` (verified twice). P2 was falsified: divergence from fp8 is 4.35%, not under 1%. P3 passed exactly at its line (6 of 8, geometric mean 1.109×); the 4.1× seen in simulation does not reproduce in the kernel. P4 was falsified: pre-softmax cosine similarity is 0.999996–1.000000 on all 8 even without `smooth_k`, so it is an invalid metric for softmax-neutral transforms; on the output metric `smooth_k` improves int8 on 7 of 8 captures, with geometric mean 2.08×. A quantizer bug was caught before shipping: `v_cvt_pk_i16_f32` truncates toward zero, differing from `torch.round` on 48.85% of elements, and was replaced with `v_rndne_f32`.

**Status:** confirmed (built and scored; the gate holds, the full accuracy win does not transfer). Corrects the cosine-similarity evidence in F081 (a reference to it as being in F079 was also wrong).

### F084 — What int8 QK and `smooth_k` cost at the production shape

**Question.** What do the int8-QK split and `smooth_k` cost at the production shape and at long N, and does the split beat the incumbent?

**Result.** int8 QK is speed-neutral: the int8/fp8 time ratio is 1.0074 / 1.0159 / 1.0061 / 1.0075 at the four shapes (the prediction was a null in [0.95, 1.05]), so F083's accuracy win is free. Against PR #368 (our arms kernel-only against full calls, which favours ours) the fp8 arm is 0.9525 at N=8771 and 1.0681 / 1.0825 / 1.0829 at N=16384 / 32768 / 47520, and the int8 arm is 1.0795 and 1.0859 at 32768 and 47520. int8/production is 0.86–0.87 at every shape; N=16384 is void and 32768 and 47520 are ratio-only.

**Status:** confirmed (int8 QK is speed-neutral, +0.6% to +1.6%).

## 7. Odd lengths, the `EVEN_N` mask and a shipped-kernel bug

The non-causal kernel carried a mask that cost about 20% (F060, F064). Putting it behind a uniform branch removed the odd-length penalty at N=8771, and forcing the same path at aligned lengths removed what had looked like an aligned-length deficit (F065). Making that path the default went through an accuracy gate, a rejection and a reversal (F066 to F069). The group also holds a correctness bug found in the shipped causal kernel (F062).

### F060 — The non-causal `EVEN_N` mask cost, and a tail peel

**Question.** What does the non-causal `EVEN_N` mask cost, and is a tail peel worth building?

**Result.** A first comparison suggested the mask costs +8.45% (odd/even normalised 1.08454). The peel was then built and was slower: peel/shipped time 1.1858 at N=8771 non-causal and 1.2260 at N=8768 non-causal, with 3 of 4 configurations regressing. Later work showed the +8.45% was a lower bound and the peel was the wrong instrument.

**Status:** refuted. F064 recovered the cost with a uniform branch (+20.1% non-causal), and F065 put the true mask cost at +19.3%. The peel stays behind a default-off flag.

### F062 — The shipped causal kernel returned NaN in some configurations

**Question.** Does the shipped kernel return NaN, and where does it come from?

**Result.** 4 of 20 (N, BLOCK_N) causal configurations gave NaN in the shipped kernel (0 of 20 in the peeled one), and N=64 with BLOCK_N=32 gave silent garbage without NaN (rms 5.542e+00 against 3.6999e-03). The cause: the causal `hi` bound was not clamped to existing key blocks, so loops read past the K/V buffers. The fix is `hi = tl.minimum(cdiv((start_m+1)*BLOCK_M, BLOCK_N), cdiv(n_pad, BLOCK_N))`. The failure depends on process history: at N=8768 causal with BLOCK_N=16 it was 0 on a fresh process and 1920 after smaller shapes. After the fix there are 0 NaN in all 40 cells, and the fixed/frozen time ratios are 0.99800 / 0.99973 / 0.98685 / 1.00525, within the bracket residual.

**Status:** shipped (the clamp is applied by default and costs nothing measurable).

### F064 — The score mask behind a uniform branch

**Question.** Can the odd-length non-causal cost be recovered by gating the score mask behind a uniform branch?

**Result.** At N=8771 non-causal, gating the mask (`GATE_MASK`) alone is a 20.11% gain and the full default configuration a 22.71% gain over the shipped kernel. Ours/PR #368 goes from 1.2224 (a loss) to 0.9448 (a win). The odd-length penalty is a cliff at N=8769, and PR #368 pays a matching +8.95% odd-length penalty (the shipped kernel +8.88%).

**Status:** shipped (`GATE_MASK`, `PAD_KV` and `LOAD_UNMASKED` default to 1). Overturns F060 (the peel failed, the branch succeeds, and the mask is worth +20.1%, not +8.45%). The aligned-deficit statement was overturned by F065.

### F065 — The aligned non-causal deficit is an `EVEN_N` artifact

**Question.** Is the N=8771 win real, and why does the 1.2242× aligned N=8768 non-causal deficit persist?

**Result.** The N=8771 win is gated: bit-identical to the shipped kernel at N=8769–8772, both causal modes, with 0 NaN. Forcing the ragged code path at N=8768 recovers 19.29%, which the file says more than closes the 1.2242× aligned non-causal deficit. Triton's divisibility-16 attribute changes the binary but nothing measurable (−0.04%). Fused `smooth_k` costs +2.0% of a full call against +14.9% for the torch form, with only a 0.05% output-error gain on the random surrogate (F066 reversed this on real captures).

**Status:** confirmed (the `EVEN_N` mechanism). Overturns F064's statement that the aligned deficit was untouched and F060's mask cost. F067 later retracted a similar "beats PR #368" ratio claim made for N=8192 in F066.

### F066 — Making the fast path the default: the gate rejected it

**Question.** Can `even_n_override=0` (the ragged path at aligned shapes) become the default under a bit-identity gate?

**Result.** The gate rejected it. The non-causal win is 15.6% to 22.7% at every aligned shape (120 cells, 20 real shapes) and the causal case ties (±1.3%), but non-causal output is not bit-identical to the shipped path (differences of 3e-5 to 1.2e-4, 0.5–0.9% of output scale, accuracy-neutral to 6 significant figures); causal is bit-identical. On real Krea2 block 0, `smooth_k` reduces output error by 17.5% / 19.2% / 24.8%, which reverses F065's 0.05% on the surrogate.

**Status:** refuted as a default decision. F067 retracted its N=8192 "beats PR #368" headline, and F068 then promoted the fast path to the default.

### F067 — The accuracy gate for the fast-path default <sub>(stage S1)</sub>

**Question.** Does the fast path pass a pre-registered accuracy gate so that it can become the default?

**Result.** The gate failed at 2 of 204 family-A cells, both a Krea2 capture at N=288, non-causal: `rel_max` 10.83% at BLOCK_N=16 and 19.10% at BLOCK_N=32, with `rel_mean` 0.514% and 0.631% in the fast path's favour (236 cells, 0 NaN). The default was left unchanged. It also retracted F066's N=8192 headline as an arm/bracket category error: fast/PR #368 is 1.01869 and shipped/PR #368 1.289×, with the margin inside the 3.6–5.5% within-process spreads.

**Status:** confirmed (the measurement and the retraction). The decision not to promote was overturned by F068.

### F068 — Is the tail failure systematic or random? <sub>(stage S2)</sub>

**Question.** Is the `rel_max` tail failure found in F067 systematic or random?

**Result.** Random. In the failing regime (Krea2 block 0, N=288, `latent_scale` 0.05, non-causal, BLOCK_N=16), 16 of 16 fresh inputs have `ratio_max` exactly 1.00000 and the fast arm is better on mean error in 15 of 16. Across 504 informative comparisons, `ratio_max` above 1.05 occurs only on the F067 capture (1.1083 at BLOCK_N=16, 1.1909 at BLOCK_N=32); the pooled non-causal test gives W=2, B=3, frac=0.40, p=0.81. The fast path was promoted to the default (`even_n_override=None` now means `EVEN_N=False`; the shipped behaviour is the opt-out).

**Status:** shipped. Overturns F066's rejection and the not-promoted decision of F067.

### F069 — Is the promoted default safe at every tile? <sub>(stage S3)</sub>

**Question.** Is the promoted fast-path default safe at every auto-selected tile?

**Result.** Yes, validated at all four tiles with nothing restricted: 177 cells, 0 NaN, A/A max|d| 0. At the 64×16 tile, 81 cells gave a worst `ratio_max` of 1.1083 on one input (the known F067 capture); at each of the 64×32 and 64×64 tiles (causal), 32 of 32 cells were bit-identical to the shipped path; at the 128×16 tile (N≥8192), 32 cells gave `ratio_max` 1.0000. The worst fast/ship time ratio anywhere is 1.01457 (64×64 tile, N=1536), inside the 13.4% A/A spread and under the 10% cap.

**Status:** confirmed. Extends F068.

## 8. Causal, long-sequence and end-to-end measurements

Measurements that put the kernel in context: long sequences, causal attention, an end-to-end Krea2 forward pass, and an external check of the earlier explanations. They show where the win survives (causal, and against production) and where it does not (non-causal at long N against PR #368). The causal results do not apply to Krea2, which is non-causal.

### F074 — Long-sequence sweep, non-causal

**Question.** Does the win survive at long N (16384, 32768, 47520) at (1,48,N,128), non-causal?

**Result.** Against production (full call), ours/production is 0.8463 / 0.8558 / 0.8648 at N=16384 / 32768 / 47520 (13.5–15.4% faster), and ours/SDPA reaches 0.5226 (1.91× faster) at 47520. Against PR #368 the result flips from 0.9540 at N=8771 to 1.0585 / 1.0719 / 1.0682 at long N: PR #368 is faster. Ours is measured kernel-only and the comparators as full calls, which favours ours.

**Status:** confirmed (wins against production and SDPA at every long N; loses to PR #368 at long N).

### F076 — An end-to-end Krea2 forward with the attention swapped

**Question.** With the attention implementation swapped inside a real Krea2 forward pass, is there an end-to-end regression against production, and what share of the pass is attention?

**Result.** At N=1056, ours/production is 0.9864× of total pass time (1.4% faster) and ours/SDPA 0.8852×. The attention kernel is 0.63% of the pass for ours (2.25% for production, 12.74% for SDPA), while the whole attention module including projections is 28.6% to 37.0%. The N=4128 and N=8681 blocks are void for timing.

**Status:** confirmed (valid at N=1056 only). The project's end-to-end criterion (no regression with the attention swapped inside a real forward pass) is met there.

### F077 — Causal comparison with the current kernel

**Question.** How do ours and PR #368 compare on causal attention at (1,48,8771,128) with the current kernel?

**Result.** Full-call causal ours/PR #368 is 0.6308 raw and 0.6318 bracket-corrected (1.585× faster); kernel-only causal ours/PR #368 is 0.5652 (1.77×) and ours/production 0.2665 (3.75× faster than production), with ours timed kernel-only against full calls of PR #368 and production, which favours ours. The causal mask helps PR #368 by only 1.07× and ours by 1.80×. Krea2 is non-causal, where ours ties PR #368 (1.0010 full call).

**Status:** confirmed (applies to causal models, not Krea2). F078 and F080 qualify how much of it is our strength and how much PR #368's causal weakness.

### F078 — Non-causal attention explained: where the gap is

**Question.** How does non-causal quantized flash attention work, how does ours differ from PR #368, and where is the gap?

**Result.** A source audit, no GPU. Our kernel is fp8 e4m3 for Q, K, V and P, not int8. Under the causal mask the effective rate drops by 1.11× for ours and 1.87× for PR #368, so the file concludes the 1.585× causal win is mostly PR #368's causal-path weakness. The durable position is non-causal: a full-call tie at the production shape (1.0010) and a 6–7% loss at long N (ours kernel-only against PR #368's full call: 1.0585 / 1.0719 / 1.0682).

**Status:** confirmed (the analysis; the causal contradiction stays open). Corrects the record's claim that the kernel is int8.

### F079 — External verification of the non-causal explanation

**Question.** Do external sources (papers, repositories, issues, documentation) confirm or contradict the earlier claims about causal and non-causal attention and the toolchain?

**Result.** Corrections to the project's record: PR #368 is not merged (it is open); a kernel API name was wrong; int8 QK with fp8 PV belongs to SageAttention2, not v1; and the fp8/fp16 WMMA ratio is 2.0× documented, not the 1.81 in the kernel file's header. ComfyUI hardcodes `is_causal=False` for every SageAttention call, and every diffusion DiT checked is full attention, so the causal win is unreachable for this project. PR #368's own description shows an fp8 geometric-mean speedup of 1.44× non-causal against 1.27× causal, so its 1.87× causal collapse is anomalous against the literature.

**Status:** confirmed. Corrects F078 and the 1.81 in F002's framing.

### F080 — Does the causal win survive at video scale?

**Question.** Does the causal win over PR #368 survive at N = 16384, 32768 and 47520 at (1,48,N,128)?

**Result.** Kernel-only causal ours/PR #368 (ours kernel-only against PR #368's full call, which favours ours; below 1 is faster) is 0.6590 at N=32768 and 0.6179 at N=47520, both under the frozen 0.70 threshold, and the falsifier (≥ 0.90 at 47520) was refuted. Under the causal mask ours keeps 94.7 / 94.1% of its non-causal TFLOP/s rate against 58.2 / 54.4% for PR #368. The N=16384 block (ratio 0.6352; rates 92.9% and 55.7%) is void (SDPA spread 37.6%) and should not be quoted.

**Status:** confirmed (scale-durable at N=32768 and 47520). F081 notes the causal win is not sellable as stated: autoregressive video models are block-causal and their speed-ups come from the KV cache.

## 9. `smooth_v`: a lever that kept flipping

`smooth_v` subtracts V's per-channel mean before quantization. Real V carries a large offset, so it looked like a lever, but on real captures over many seeds its effect flipped with block, seed and latent scale. This group records the series, including results that were retracted as seed scatter.

### F085 — Real V carries a large per-channel offset

**Question.** Does real V carry a per-channel DC offset, which would make `smooth_v` a real lever?

**Result.** Yes. All 8 of 8 canonical captures exceed a 0.10 DC ratio, and V's median DC ratio is 0.4946, which is 12.3× the N-matched synthetic control's (corrected from 0.5309 and 11.6×, which had taken one capture's value as the median). The K figure that F082 reported as 0.605 reproduces (0.6052) but is the lowest of the set; the canonical-8 K median is 0.8915 (corrected from 0.9069), so F082 understated the effect.

**Status:** confirmed. The median values were corrected within the file and again in F091.

### F086 — `smooth_v` accuracy in our kernel

**Question.** Does an exact mean add-back (`smooth_v`) in our kernel improve accuracy?

**Result.** The gate passes: with the option off the kernel compiles to the same 835-instruction binary, and the add-back error is at most 0.967 ulp on 8 of 8 cells. F005's ×40 outlier case now gives 0.936× (`smooth_v` better, not 2.8× worse). The frozen rule returned NEGATIVE, because accuracy on the frozen metric `rel_max_err` passes on only 5 of 8 captures. But `smooth_k`, the shipped lever, also scores 5 of 8 on that metric, so the metric fails its own positive control; on output rms, the metric F082 used, `smooth_v` wins 8 of 8 against 7 of 8 for `smooth_k`.

**Status:** open (re-tested in F087 with a corrected metric). The file also reported a kernel abort that F088 explains.

### F087 — `smooth_v` on held-out captures

**Question.** Does `smooth_v` win on held-out captures when scored on output rms and checked against the `smooth_k` control?

**Result.** On 16 held-out captures the control holds (`smooth_k` improves 14 of 16) and `smooth_v` improves 7 of 16, which passes the frozen count but falls below a same-proportion 75% bar. Two predictions were falsified: the `smooth_k` geometric-mean gain is 1.0985 (14 of 16) against 1.0061 for `smooth_v` (7 of 16), and the joint option beats the best single option on only 4 of 16. Decision: do not ship `smooth_v` as the default.

**Status:** refuted for the depth and DC-ratio readings this file reported, which F092 retracted as seed scatter; the production-shape edge for `smooth_v` (2 of 2 at N=8771) stays at n=2.

### F088 — A kernel abort traced to non-contiguous V

**Question.** Why does the `kseq_mean_fp8` kernel hard-abort on one Klein capture?

**Result.** The cause is a non-contiguous V (a reshape produced a view with the wrong `stride(0)`, so reads went out of bounds), not the shape: a same-shape contiguous capture works. On the GPU the non-contiguous cases exit with 3221226505 (0xC0000409, `STATUS_STACK_BUFFER_OVERRUN`); after adding `.contiguous()` all eight case runs exit 0, and the real failing capture gives a maximum difference against torch of 2.384186e-07, bit-identical to the fix control.

**Status:** confirmed (root cause proven on the GPU and fixed; the contiguous path is unchanged).

### F091 — Real K and V offsets at the production length

**Question.** Does the per-channel DC offset of real K and V persist at N=8771?

**Result.** Yes, and it does not shrink with N. Between N=288 and N=8771 the N-matched control falls 6.3× (0.0438 to 0.0070) while real V falls only 1.06× (0.4946 to 0.4652) and real K rises (0.8915 to 0.9209). At N=8771, V is 64.8× (block 0) and 68.1× (block 26) its control (12.3× at N=288), and K is 142.7× (block 0) and 120.5× (block 26). It also corrects F085's median row, which had used the wrong values (V 0.4946 and K 0.8915, not 0.5309 and 0.9069).

**Status:** confirmed. Corrects F085.

### F092 — Is `smooth_v`'s sign a function of block depth?

**Question.** Is `smooth_v`'s sign a function of block depth, and does the DC ratio gate it?

**Result.** The hypothesis is refuted across 28 blocks at N=288. `smooth_v` wins 8 of 9 early blocks, the depth correlation is Spearman −0.219, and the DC ratio is weakly positive rather than anti-predictive (ρ = +0.343). `smooth_k` wins 26 of 26 unseen blocks (geometric mean 1.137) and beats `smooth_v` on 22 of 26. Block 0 flips with the seed: seed 1234 wins (`sv/off` 0.9599), seeds 7001 and 7002 lose (1.0072 and 1.0185), so F087's depth effect was seed scatter.

**Status:** refuted. Overturns the depth and DC-ratio readings in F087.

### F093 — What `smooth_v` costs, against `smooth_k`

**Question.** What does `smooth_v` cost at the production shape (1,48,8771,128), alone and joint with `smooth_k`, compared with `smooth_k`?

**Result.** Composed full call (event timer), as a ratio against the plain call (above 1 is slower): `smooth_k` 1.006–1.017×, `smooth_v` 1.006–1.018×, joint 1.022–1.035× (wall time 1.020–1.021×, 1.019–1.022× and 1.038–1.040×). The clock pair was not clean (drift +6.04%), so these are ratios only, and they are not end to end.

**Status:** confirmed. `smooth_v`'s cost is indistinguishable from `smooth_k`'s, and the joint costs +2.2% to +3.5% of a full call.

### F094 — `smooth_v` on the int8 QK arm

**Question.** Can `smooth_v` be enabled on the int8 QK arm without changing other paths, and does it improve accuracy over int8 with `smooth_k` on real captures?

**Result.** The inertness gates pass: the `smooth_v=False` int8 path and the fp8 path are unchanged, and zero-mean V is bit-identical. As first measured, at N=288 over 28 blocks with one seed (1234), int8 + `smooth_k` + `smooth_v` beat int8 + `smooth_k` on 28 of 28 blocks (geometric-mean gain 1.1636×), and on 2 of 2 at N=8771 (1.2503×). F095 then showed this is a block-axis result at one seed and one latent scale: on the seed axis at the same scale it is 11 of 16 (1.0509), and at two of the three reduced latent scales the option loses 16 of 16 (11 of 16 at the third).

**Status:** refuted (the effect size does not transfer across seeds); the inertness gates stand. Corrected by F095.

### F095 — A multi-seed test of int8 `smooth_v`, and the cost of "everything on"

**Question.** Does `smooth_v` on int8 replicate across seeds and latent scales (64 captures: 4 latent scales × 16 seeds, block 0, N=288), and what does the everything-on configuration cost?

**Result.** At the production latent scale it wins 11 of 16 with geometric-mean gain 1.0509, which is inconclusive (p = 0.2101). At `latent_scale` 0.05 and 0.1 it loses 16 of 16 (gains 0.8341 and 0.8746), and at the smallest scale (0.025) it loses 11 of 16 (0.9620); the pooled geometric-mean gain is 0.9267 (48 of 64 losses). On cost, all six frozen predictions pass: the int8 joint costs 1.0239× (event) and 1.0459× (wall) of a full int8 call.

**Status:** confirmed. Corrects F094's 28/28. Nothing ships.

### F096 — A powered seed test, and the dependence on `latent_scale`

**Question.** With more seeds, is int8 `smooth_v` a real win at the production latent scale and a real loss at reduced scale, and is the effect monotone in latent scale?

**Result.** All seven frozen predictions pass: 21 of 28 wins at the production scale (p = 0.0125), 5 of 28 at `latent_scale` 0.05 (p = 0.0009), and a Spearman value of +0.4257 for the third. The effect is not monotone: the minimum gain is at `latent_scale` 0.25 (0.8658), below the gain at 0.05 (0.9032), which refutes the premise of one frozen prediction.

**Status:** confirmed (a win at the production scale, a loss at reduced scale, non-monotone). It upgrades F095's inconclusive 11 of 16 to 21 of 28.

### Does `smooth_v` beat `smooth_k` at the production length? <sub>(track H8′, stages S5 and S8)</sub>

**Question.** Does the N=288 ordering, `smooth_k` better than `smooth_v`, reverse at the production shape N=8771?

**Result.** A first block with 6 of the 8 planned test blocks (6, 9, 12, 15, 18, 24), one seed per block, ranked `smooth_v` above `smooth_k` on 4 of 6 (the minimum to pass): geometric-mean gains of 1.1732 (`smooth_k`) and 1.0427 (`smooth_v`) at N=288 invert to 1.0456 and 1.2069 at N=8771 on the same 6 blocks. With two more blocks (n=8) the primary prediction passed at 5 of 8, the exact minimum, and the other three predictions passed. A later seed-replication test (F110) did not reproduce this as a seed-robust majority.

**Status:** open. F110 found the ordering is set by block, not by seed, and not a seed-robust majority (6 of 9 against a bar of ≥ 8 of 9).

### F109 — An inventory of the N=8771 captures <sub>(stage S14)</sub>

**Question.** Do the existing captures permit a seed-replicated N=8771 `smooth_v` against `smooth_k` test?

**Result.** No. At N=8771 there is one distinct independent sample (seed 1234) across 10 distinct blocks; the multi-seed families exist at N=288, block 0 only. New captures were required, and no GPU block was run.

**Status:** confirmed.

### F109 — The seed-replicated capture block is void <sub>(stage S15)</sub>

**Question.** Can 3 seeds × 3 blocks of N=8771 captures be produced reproducibly and bit-identically to the earlier seed-1234 captures (gate G-CAP)?

**Result.** Void on the frozen G-CAP. Determinism passes (a duplicate pair is identical by sha256) and completion is 10 of 10, but bit-identity of the seed-1234 capture against the earlier block-sweep capture fails 0 of 3. The capture path is deterministic within a session but not reliably reproducible across sessions (one cross-session match was observed). Stage B was not run and no accuracy number was produced.

**Status:** open (void). The accuracy test was redone on these captures in F110 with a corrected gate.

### F110 — Is the `smooth_v` ordering seed-robust? <sub>(stage S16)</sub>

**Question.** Over 9 attested captures (3 seeds × 3 blocks), is the `smooth_v`-over-`smooth_k` ordering a seed-robust majority (B1: pooled geometric mean above 1.0 and a sign count of at least 8 of 9)?

**Result.** B1 fails: the pooled geometric-mean gain of `smooth_v` over `smooth_k` is 1.115450 (above 1.0), but the sign count is 6 of 9 against a bar of ≥ 8 of 9. The ordering is set by block, not by seed: block 6 favours `smooth_v`, block 12 is marginal and block 18 favours `smooth_k`, and all three seeds agree on each block. The int8 twin's pooled gain is 1.454194 with 9 of 9 wins.

**Status:** refuted (B1 failed). It does not reproduce the N=8771 reversal reported at 5 of 8 blocks (track H8′) as a seed-robust majority, and it corrects F109's stage A gate.

## 10. The `sv` fold and the per-channel V scale

A mechanism for why `smooth_v` reverses at reduced latent scale (it shrinks V's scale, which coarsens the fp8 attention weights), two attempts to fix it, and a per-channel V scale that worked in accuracy tests. Each fix was then priced, none was cheap enough to enable, and the later entries locate where the cost lives.

### F097 — Why `smooth_v` reverses: the `sv` fold into P

**Question.** Why does int8 `smooth_v` lose at reduced latent scale, and would a per-query-row dynamic P scale remove the reversal?

**Result.** A CPU-only fp64 emulation, no GPU. Relative elementwise V quantization error is 2.49–2.63% in both arms at every scale, so V quantization is not the cause. `smooth_v` shrinks the V scale `sv` by 1.28× to 6.54×, which pushes `p·sv` down the e4m3 grid and coarsens P. The emulation matched each capture with Spearman +0.924 (an in-sample retrodiction), and a per-row dynamic scale gave emulated gains of 1.298 / 1.138 / 1.065 / 1.350. As a frozen out-of-sample test on F096's 36 captures, three predictions passed (including gains of 1.1689 / 1.1884 / 1.5969 with 34 of 36 wins) and two were falsified: the effect is not monotone, with its minimum at `latent_scale` 0.25.

**Status:** confirmed (mechanism and fix validated out of sample); the claims that the effect is monotone and lives only in the P term were withdrawn. Nothing shipped.

### F098 — A dynamic per-row P scale: accuracy and cost

**Question.** Does a dynamic per-query-row P scale (`PSC_DYN`, default off) remove the `smooth_v` reversal in the real kernel, and what does it cost?

**Result.** Accuracy on 100 real captures (N=288, block 0): all six frozen accuracy predictions pass, for example `i8_sk_dyn`/`i8_sk` pooled 1.1080 (84 of 100), and the reversal is removed at every scale. The cost prediction was falsified: the composed full-call ratio is 1.1559 against a frozen bar of ≤ 1.05, which the source states as +15.6% of a full call (+16.9% joint), with the kernel growing from 1088 to 1296 instructions (+19.1%). There is no production-shape accuracy claim.

**Status:** closed. The fix is real but costs about 16%, so `PSC_DYN` stays 0 and nothing ships.

### F099 — Folding the row scale into the softmax row max

**Question.** Can F098's per-row P scale be applied in the exponent, folded into the softmax row max, to keep the accuracy and cut the cost?

**Result.** Neither accurate nor cheaper. `i8_sk_fold`/`i8_sk` pools to 1.0042 (33 of 64) and 1.0538 (24 of 36); `sv_fold`/`sv_dyn` is 0.8350 with 0 of 64 wins. The composed cost is 1.1582 against 1.1531 for the dynamic version in the same run, and the static ISA grows by +184 instructions (1088 to 1272) against a frozen expectation of +60 to +150.

**Status:** closed (`PSC_FOLD` stays 0, as does `PSC_DYN`). Corrects F098 §8.4, whose claim that the scaled P stays ≤ 1 is false as written.

### F100 — Per-channel V scale (`VSCALE_CHAN`) fixes the `sv` fold <sub>(stage S6)</sub>

**Question.** Does quantizing V against a per-channel divisor, with `sv` written as 1.0 and the channel scale re-applied in the epilogue, remove the `smooth_v` reversal at N=288?

**Result.** On 100 real captures at N=288 (rms against fp64), the pooled gain over `i8_sk` (above 1 is better) is 1.2666 for `i8_sk_chan` (100 of 100) and 1.2303 for `i8_sk_sv_chan` (95 of 100). The first draft's wording was wrong and was corrected: 1.3122 is the fold gain given `smooth_v`, not the `smooth_v` axis. The `smooth_v` axis, as an error ratio of `smooth_v` on to off (above 1 means `smooth_v` hurts), is 1.0665 under the shipped fold (a 6.65% loss) and 1.0295 under the new fold (still a 2.95% loss). "Flips to a 31% gain" and "the penalty is gone" were retracted; the combined arm's reversal is gone.

**Status:** shipped (behind `VSCALE_CHAN=0`, default off). F101 later reversed the sign of the `smooth_v`-axis loss at N=8771.

### F101 — `VSCALE_CHAN` at the production shape <sub>(stage S7)</sub>

**Question.** Does the per-channel V scale fix hold on 6 N=8771 captures that no int8 arm had touched (blocks 6, 9, 12, 15, 18, 24)?

**Result.** Accuracy only (gain = rms(b)/rms(a); above 1, a is better). The primary prediction passes on all six: `i8_sk_sv_chan` against `i8_sk` is 6 of 6, pooled 1.8095 (bars ≥ 5 of 6 and ≥ 1.10). The `smooth_v` axis under the new fold is 6 of 6 at 1.2480, which reverses F100's N=288 loss; P1 = P2 × P3 exactly (1.4499 × 1.2480 = 1.8095), so the three passes are two facts. One control was falsified (5 of 6, pooled 1.0038 against a bar of 2.0) because its bar was anchored on a block-0-only value; block-matched N=288 blocks give 1.0055.

**Status:** confirmed. It reverses the sign of F100's `smooth_v`-axis loss at N=8771; the N=288 value still stands.

### F102 — The composed cost of `VSCALE_CHAN` <sub>(stage S8)</sub>

**Question.** What does `VSCALE_CHAN` cost in composed (full-call) mode at the production shape, against the frozen bars C1 `sk_chan/sk` ≤ 1.06 and C2 `sv_chan/sv` ≤ 1.08?

**Result.** Both frozen cost predictions were falsified: C1 composed (event) is 1.0728 against ≤ 1.06, and C2 is 1.1112 against ≤ 1.08, with 9 of 9 rounds above the bar for both. The flag stays 0 and nothing is promoted. A flag-on ISA block in the same stage gave 2563 instructions for `sk`, 2665 for `sk_chan` (+102), 2587 for `sv` and 2681 for `sv_chan` (+94).

**Status:** refuted (both cost predictions). The cost was located in later findings (F104 to F108).

### F104 — A launch census, and a fused channel prologue <sub>(stage S9)</sub>

**Question.** Where do `VSCALE_CHAN`'s prologue bytes go, and can a fused prologue be made bit-identical?

**Result.** The census, with no timing, shows `sk_chan` − `sk` and `sv_chan` − `sv` both cost +2 launches, +1 V read and +1 V8 store, 161,667,072 B, while the fused `sv_chan_f` − `sv` is +0 / +0 / +0, 0 B. The fused path is bit-identical on 72 of 72 cases and the flag-off binary is unchanged. It is not claimed to be cheaper.

**Status:** confirmed (the census and the bit-identity gate). F105 refuted the byte attribution as the cost mechanism: `sv_chan_f` − `sv` still costs +10.53% at zero extra bytes.

### F105 — The composed cost of the fused channel prologue <sub>(stage S10)</sub>

**Question.** Does the fused prologue bring composed cost under the bars C1 `sk_chan_f/sk` ≤ 1.06 and C2 `sv_chan_f/sv` ≤ 1.08?

**Result.** C2 is 1.105317 against ≤ 1.08, falsified. C1 is 1.057495 and passes narrowly. The fused prologue does not rescue the per-channel path, and both flags stay 0. A/A spreads were 0.70% and 0.77%.

**Status:** refuted (C2). It overturns F104's byte attribution; F107 and F108 later refuted its quantizer half and confirmed its statistics-kernel half.

### F106 — The `SV_ONE` loop-body specialisation <sub>(stage S11)</sub>

**Question.** Does specialising away the dead `sv` load and fold (`SV_ONE`) bring the per-channel cost under C1 ≤ 1.06 and C2 ≤ 1.08?

**Result.** On the fixed arm C1 is 1.061380 and C2 is 1.106383, both falsified. The census shows the loop body is identical to the per-token arm and the difference is the epilogue re-apply of the channel scale (+128 fp32 multiplies), so the loop-body route is closed. One prediction (C5, 1.002316 and 1.001437) was sign-blind in its frozen form and registers a slowdown.

**Status:** closed (the loop-body route). The epilogue attribution was partly revived by F108 and then not established by F111.

### F107 — Where the per-channel cost lives <sub>(stage S12)</sub>

**Question.** Is the `VSCALE_CHAN` cost in the prologue (the statistics pass and the host sync), rather than in occupancy or the kernel?

**Result.** The occupancy hypothesis (H-OCC) was refuted without the GPU: all 8 arms share 16384 B of shared memory and 4 workgroups per CU, LDS-bound. The timing block was partial (P3 failed). The per-channel prologue is 1.82609× the per-token one, the fused mean-and-amax statistics pass is 2.78474× the mean-only pass (+0.60205 ms), the device-to-host sync makes the full path 11.05% slower than the no-sync one, and the per-channel quantizer itself is neutral (1.00240).

**Status:** refuted (the occupancy hypothesis). F108 withdrew the reading that the prologue dominates: it is 43% of the cost.

### F108 — The same-process split: prologue against kernel <sub>(stage S13)</sub>

**Question.** Is the per-channel cost mostly prologue (frozen prediction R = Δprologue/Δcomposed ≥ 0.80)?

**Result.** The prediction missed: R = 0.4304 against ≥ 0.80, and R ≤ 0.50 means kernel-side. The composed cost is 43.0% prologue and 57.0% attention kernel, the kernel share being an inference by subtraction. It was run as four arms in one process, a disclosed deviation, and the composed ratio sits only marginally beyond the A/A spread.

**Status:** refuted (the prologue-dominance prediction; the 43% and 57% split is the measurement). It withdraws F107's prologue-dominance reading and partly revives F106's epilogue verdict; F111 later found the epilogue attribution not established.

### F111 — The kernel-side epilogue and `SV_ONE` split <sub>(stage S17)</sub>

**Question.** On the kernel itself, does the epilogue re-apply of the channel scale cost more than `SV_ONE` (P1), is `SV_ONE` a null (P2), and is the epilogue difference in [0.8, 1.3] ms (P3)?

**Result.** Additivity is proven (P5) and `SV_ONE` is a null (P2; the pre-registered median-based delta is −0.0838 ms). The epilogue verdict is not established: the pre-registered epilogue difference is +0.6795 ms (paired median +0.3482 ms), but the one-sided sign test gives p = 46/512 = 0.0898, the paired mean is −1.1085 ms, and the A/A-equivalent range of ±2.42 ms exceeds the effect. P3 fails (34.3% below F108's figure), and the sign flipped between the two block attempts.

**Status:** open (the epilogue attribution is unresolved; F108's 57% kernel share remains an inference). The verdict chain for F106's epilogue claim is closed (F106), partly revived (F108), not established (F111).

## 11. The long-N gap and the last Triton stages

At long sequence lengths PR #368 stays ahead non-causally. These stages decompose the gap, test hypotheses for it one at a time, show that the speed ratio depends on a `smooth_k` setting the two kernels did not share, and finally place the gap in the kernel. The Triton work ends here: no flag-gated change closed the gap.

### F112 — The true full-call gap at long N <sub>(stage S18, with supplement F112b)</sub>

**Question.** What is the true full-call, prologue-included, non-causal ratio of ours to PR #368 at N = 8771, 16384, 32768 and 47520 for (1,48,N,128)?

**Result.** With PR #368 as shipped (`smooth_k=True`, the F112b supplement), full-call ours/PR #368 is 1.1005 / 1.1109 / 1.0940 at N = 16384 / 32768 / 47520 (above 1 means ours is slower). That is larger than F074's kernel-only 1.0585 / 1.0719 / 1.0682 by our prologue (full/kernel 1.0391 / 1.0324 / 1.0260). The main run with `smooth_k=False` on both arms gave 0.9449 / 1.0079 / 1.0320 / 1.0161 at the four sizes, so the difference from F074 was the `smooth_k` switch. N=8771 is void (12.88% spread).

**Status:** confirmed (the long-N loss to PR #368 holds for the shipping configuration at 16384, 32768 and 47520).

### F113 — Localising the long-N loss <sub>(stage S19)</sub>

**Question.** Where does the long-N full-call loss to PR #368 come from: bandwidth, re-streaming, rasterisation, occupancy or prologue allocation?

**Result.** The decomposition is exact: full/PR #368 = kernel/PR #368 × full/kernel = 1.0591 × 1.0391, 1.0760 × 1.0324 and 1.0664 × 1.0260 at N = 16384 / 32768 / 47520. Refuted without the GPU: K/V bandwidth arithmetic (identical K/V bytes per MAC), re-streaming (total K/V ratio 0.9946–1.0000), rasterisation and occupancy. The allocation hypothesis was refuted by a pre-registered diagnostic (6 allocations cost 0.4–1.0% of the prologue). What survives is an in-situ interaction, with residuals of 0.17 / 3.2 / 6.2 ms. After re-checking, "linear in N" was retracted (3.249× the time for 2.900× the length) and "1.6–2.4×" was replaced by 1.12 / 2.08 / 2.38×.

**Status:** refuted (the five hypotheses); the in-situ interaction stays open. The first candidate for it was falsified in F114, and F115 attributed it to the quantizer side.

### F114 — A prologue-buffer reuse flag <sub>(stage S20)</sub>

**Question.** Does a default-off prologue-buffer reuse flag remove the in-situ interaction (hypothesis H1′: allocation churn on the DRAM side)?

**Result.** The flag is bit-identical (max abs difference 0.0) and inert (0 new Triton compiles). Reuse/full is 0.99849 at N=32768 and 0.99947 at N=47520, i.e. 0.15% and 0.05% faster. The frozen criterion (≤ 0.995) failed, so H1′ is falsified. The arm meant to test allocation alone was vacuous because of a design defect (it did not allocate per call), so that clause of the frozen rule is unanswered.

**Status:** refuted (H1′); the flag stays default-off.

### F115 — Splitting the in-situ premium by stage <sub>(stage S21, with supplement F115b)</sub>

**Question.** Is the in-situ premium due to a slower quantizer, to a slower attention kernel after it, or to host work?

**Result.** The kernel-after mechanism failed its bar: segment-kernel/our-kernel is 1.0020 at N=16384 and 1.0073 at N=47520 (+0.20% and +0.73%) against a 1.20 bar and an A/A floor of 0.06% and 0.04%, so it is bounded at 0.73% or less. The excess is not host work (a host difference of 0.2237 and 0.1639 ms against a 3.0 ms bar). The premium is GPU-resident and on the quantizer side, but the frozen criteria for naming the quantizer mechanism also failed, so the stage reports the mechanism as unresolved.

**Status:** open (quantizer-side excess indicated; confounds remain).

### F116 — Quantizing Q inside the attention kernel <sub>(stage S22)</sub>

**Question.** Does quantizing Q inside the attention kernel, behind the default-off flag `Q_IN_KERNEL`, speed up the full call without changing accuracy?

**Result.** The flag is additive and accuracy-equivalent (every identity gate true, pooled accuracy gain 0.9999906, 0 NaN), but the speed result is a null. `skipq`/full is 1.00605 at N=8771, 0.99266 at N=16384 and 0.99319 at N=47520 (below 1 is faster). The frozen rule gives NULL because the prediction fails at N=8771, and the block is void under the clock rule (2.1136926 to 2.0373702 GHz, −3.61%). The in-kernel quantizer already existed (`FUSE_Q`), so the baseline in stages 18 and 20 had been the `prologue_fp8` call, not the shipped call.

**Status:** closed (null, and void under the clock rule; the flag stays at 0).

### F117 — The speed ratio is a function of the `smooth_k` setting <sub>(stage S23)</sub>

**Question.** With `smooth_k` matched on both kernels, what is the full-call ratio of ours to PR #368 in each cell, and does a single matched row replace the earlier draft row?

**Result.** The matched pair moves the ratio by 5.63 to 10.50 percentage points across all eight cells (8.08 to 10.50 on the four sound cells), so the ratio depends on the `smooth_k` setting and no single matched row replaces the draft. Non-causal full-call ours/PR #368 (above 1 is slower) at N=16384 / 47520: matched OFF 1.00967 / 1.01600 (parity), matched ON 1.11462 / 1.09803. Causal: matched ON 0.69140 / 0.66537 against OFF 0.77217 / 0.76240. Our kernel moves at most 0.61% with the switch; PR #368's whole call moves 7.2–15.1%. Four of the eight cells are void.

**Status:** confirmed. Replicated by F119.

### F118 — Which `smooth_k` form runs, and what it costs <sub>(stage S24)</sub>

**Question.** Does `smooth_k=True` take the fused or the torch form, what extra work does it add, and how does PR #368 implement it?

**Result.** It takes the fused form in both entry points (24 of 24 cells; the torch `smooth_k_axis` is never executed). ON adds 2 GPU launches and 1 extra read of K (107.8 / 201.3 / 583.9 MB at N=8771 / 16384 / 47520), with no extra V traffic. Direct prologue timing was void under two frozen protocols, so the usable cost is F065's fused figure: +38.1% of the prologue, which is 1.9% of a full call at N=8771. PR #368's ON branch swaps the attention kernel for a raw-Q one.

**Status:** confirmed (form and dispatch are exact; the cost is taken from F065). It narrows F065 and explains the F117 headline through the raw-Q path.

### F119 — A matched `smooth_k`-ON re-time <sub>(stage S25)</sub>

**Question.** Does a matched `smooth_k=ON` re-time reproduce F117's ratios, and is the matched-ON gap on PR #368's side?

**Result.** On three sound cells (16384 causal, 47520 non-causal, 47520 causal) it replicates F117 to 0.05–0.65%. Ours/PR #368 full call (above 1 is slower): matched OFF 0.77722 / 1.01523 / 0.76291, matched ON 0.68958 / 1.09891 / 0.66626. Ours changes by 1.01929 / 1.00429 / 1.00568 between ON and OFF. The equivalence gate passes (0 NaN, output `rel_max` 0.03051) with no kernel edit and no default change. The non-causal matched-ON deficit is on PR #368's side; that the raw-Q swap causes it is an inference.

**Status:** confirmed. Replicates F117.

### A corrected instruction census of the raw-Q kernel <sub>(stage S27)</sub>

**Question.** Which kernel instantiation does PR #368's dispatch actually launch, and what do the instruction counts say about the matched-ON gap at long N?

**Result.** GPU-free. The earlier fork disassembly (the 3 389-instruction loop behind "ours 1.26× better") was of an instantiation that is not launched. For the dispatched one, the fork's loop is 19.7 instructions per column (1 262/64) against 24.8 for ours (397/16), so the fork is 1.26× leaner (reversing F055), and ours spends 1.70× the fork's Q cost per Q row (9.56 against 5.59). The report's first-draft inference that the long-N gap is prologue-side was withdrawn in the same report, and the gap was later placed in the kernel (F125).

**Status:** confirmed. Corrects F054 and F055, which examined the wrong instantiation.

### F120 — Does a wider KV tile help at long N? <sub>(stage S28)</sub>

**Question.** Does widening the KV tile (`BLOCK_N` 32 or 64 against the shipped 16) speed up the long-N full-call path?

**Result.** No. Full-call time against the shipped tile (128×16, `smooth_k` ON) is 1.3808 / 1.3900 for `BLOCK_N`=32 and 1.9474 / 1.9264 for `BLOCK_N`=64 at N=16384 / 47520 (above 1 is slower). Per-column instructions in the KV loop barely change (24.8 / 23.9 / 24.8 for `BLOCK_N` 16 / 32 / 64), but Triton spills 160 registers at `BLOCK_N`=64 (64 at 32) against 0 for the shipped tile, and the spill instructions inside the loop go 0 / 24 / 99 per iteration. The pre-registered prediction failed at both N.

**Status:** refuted (the tile width is not the lever; spilling is the only term that rises with the slowdown, but that it causes it is an inference).

### Measured memory and occupancy model <sub>(stage S30)</sub>

**Question.** What do the planned measurements M1–M4 settle about cache and DRAM bandwidth, the occupancy binder, and whether the loop is memory-bound?

**Result.** M1: L2 delivers 7 005 GB/s, the Infinity Cache 5 825 GB/s and DRAM 626 GB/s. M2: the VGPR file binds at 6 waves/SIMD32 and the effective LDS is 131 072 B per WGP, so F018's "LDS binds at 4" was an artifact of a misreported 65 536 B pool. M3 at N=8771: stripping 11.1% of the instructions cuts the time by 10.5% (strip ratio 0.8947), while removing 97.9% of the K/V footprint moves the time by −0.204%, so the loop is bound by a non-memory resource (issue-bound is an inference, since the strip also removes 4 barriers). M4's cold/warm arm was void, so the cache hit rate stays unmeasured.

**Status:** confirmed for M1–M3; M4 open. Overturns F018's LDS claim and an earlier Infinity Cache bandwidth figure that was 2.3× too low.

### F121 — The lazy softmax rescale behind a uniform branch <sub>(stage S31)</sub>

**Question.** Does skipping the softmax rescale behind a CTA-uniform headroom branch speed up the kernel, given that F042 and F043 said Triton cannot emit that branch?

**Result.** A wash. Full-call lazy/off is in [0.99687, 1.00430] against A/A ratios in [0.99903, 1.00058]. A CTA-wide `tl.max` costs 75 instructions and 4 barriers against the 88-instruction multiply it skips; the not-taken path is 1133 instructions against the shipped 1134 (−1, −0.09%). Accuracy matched the shipped kernel (pooled rms ratio 0.99993 / 0.99991 / 0.99914 for τ = 0.5 / 1 / 2).

**Status:** closed. It overturns F042 and F043's premise that Triton cannot emit the branch, and confirms their economic conclusion that skipping does not pay.

### F122 — The barrier-count ladder <sub>(stage S32)</sub>

**Question.** Is the KV loop barrier-bound or issue-bound?

**Result.** The stage is unresolved by its own frozen rule, which gave H-A at N=8771 non-causal and H-C (unresolved) at the other two cells. Adding 6 barriers per iteration costs +1.787% (N=8771 non-causal), +4.036% (N=8771 causal) and +2.051% (N=47520 non-causal) of a full call, against A/A ratios of 1.00104 / 0.99983 / 1.00116; adding 2 costs +0.487% / +2.446% / +0.562%. The interpolated per-barrier slope is +0.325% / +0.398% / +0.372%, which puts the 4 barriers removed in S30's strip at about 11% / 31% / 13% of its −10.5%: a minority barrier term. Barrier-bound against issue-bound was not separated.

**Status:** open.

### F123 — CU or WGP mode, and the LDS bank count <sub>(stage S33)</sub>

**Question.** Does the kernel launch in CU or WGP mode, and does the 32-bank LLVM feature contradict the ISA's 64 banks?

**Result.** WGP mode: the code object's `COMPUTE_PGM_RSRC1` is `0xE00F001D` with bit 29 set, and flipping the assembler directive changes it to `0xC00F001D`. There is no bank contradiction: 32 banks per compute unit is 64 per WGP (the 32 per compute unit is an inference from the shared feature set), so the premise was wrong.

**Status:** confirmed.

### F124 — gfx1201's `MaxWavesPerEU` <sub>(stage S34)</sub>

**Question.** What is gfx1201's per-GPU `MaxWavesPerEU`, and does it change the finding that the VGPR file binds at 6?

**Result.** `MaxWavesPerEU(GK_GFX1201)` is 16, as is gfx1200's, read from the compiler's tables through the transitive feature closure. The VGPR-binds-at-6 conclusion stands. The installed toolchain's value could not be verified, but 10, 16 and 20 are all at least 6, so the conclusion does not depend on it.

**Status:** confirmed.

### F125 — Kernel against prologue: where the matched-ON gap lives <sub>(stage S35)</sub>

**Question.** Is the matched-ON long-N gap to PR #368 in the kernel or in the prologue?

**Result.** In the kernel. Ours/PR #368 full call is 1.11472 at N=16384 and 1.09831 at N=47520. As shares of the gap (our full-call time minus PR #368's), the kernel term is +1.1001 and the prologue term −0.1401 at N=16384, and +1.2344 and −0.3301 at N=47520. Our prologue is a net advantage, because it is faster than PR #368's with `smooth_k` on: it takes 0.75004 of PR #368's prologue time at N=16384 and 0.33142 at N=47520, where PR #368's prologue costs 3.0× ours. The prediction that the prologue would carry at least 10% of the gap was falsified.

**Status:** confirmed. It sends the next stage (F126) at the kernel.

### F126 — Attacking the kernel with an occupancy flag <sub>(stage S36)</sub>

**Question.** Can a default-off flag (`waves_per_eu`) close the long-N kernel gap?

**Result.** No. `waves_per_eu=7` costs 23 VGPR spills, with a full-call ratio of 1.36924 and a kernel-only ratio of 1.39320 at N=47520 non-causal (above 1 is slower). The instruction census gives `r_instr` = 1.1986: we issue 1.20× more instructions per unit of work than PR #368. No flag-gated Triton change closes the gap.

**Status:** closed. It overturns an earlier hardware-model note that the kernel already sits at its issue floor; the tile route was closed in F120.

## 12. Hand-editing Triton's compiled output

Before writing a kernel by hand, the project tried editing Triton's compiled output (the `llir`, `ttgir` and `amdgcn` stages) and loading the result through `TRITON_KERNEL_OVERRIDE`. These experiments located where the loop's excess lives, showed that the override route is not surgical, and found no edit that cleared its frozen bar.

### F127 — An instruction-level audit of the shipped kernel's KV loop <sub>(track H1)</sub>

**Question.** Where is the shipped default kernel's KV loop, and what does it contain, instruction by instruction?

**Result.** The shipped default is `FUSE_Q=1`, not `FUSE_Q=0`, and its non-causal artifact was verified by hash. The KV body is 427 static and 391 executed slots, with 32 `v_wmma` and 16 barrier instructions (26 listing-wide). The causal `FUSE_Q=1` artifact could not be pinned by hash.

**Status:** confirmed (static). F129 found that the real entry point compiles a slightly different object.

### F128 — Loading a prebuilt code object without compiling <sub>(track H1)</sub>

**Question.** Can an edited kernel ship as a prebuilt gfx1201 code object loaded with zero compilation, and could it fail the way F001's wheel did?

**Result.** By reading the source, yes. A cache-group hit returns before any compile stage runs, through Triton's documented cache-manager hook, and the loader needs only the kernel's metadata and its `.hsaco` code object. The `TRITON_KERNEL_OVERRIDE` route is a development route only: it substitutes after `compile_ir`, so it does not skip compilation.

**Status:** open (read, not executed).

### F129 — The audited object is not the object that runs <sub>(track H2)</sub>

**Question.** Is the object that the shipped entry point compiles the one that the audit and the frozen pre-registration pinned?

**Result.** No. `flash_attn_fp8(FUSE_Q=1)` compiles to 2345 opcodes, a 428-slot KV body and 18 body `s_delay_alu` hints at N=8771 non-causal, against the audited object's 2343, 427 and 17: a difference of 2 slots and 1 hint, with 236 VGPR, 47 SGPR and 0 spills in both. A four-way bisect traced it to the runtime tensor arguments (Triton's per-argument specialisation), not to constexprs.

**Status:** confirmed (measured, GPU-free). Corrects F127's pinned object.

### Where does the roughly 20% loop excess come from? <sub>(track H2a, GPU-free)</sub>

**Question.** Where does the roughly 20% excess of our KV loop against PR #368's come from?

**Result.** A static class-by-class census compares the `FUSE_Q=0` loop that stages 35 and 36 timed (12.406 slots per `wmma`) with PR #368's (10.344). The excess is 2.062 slots per `wmma`, in three classes: fp32 multiplies +2.508, LDS reads +1.094 and barriers +0.703. The shipped `FUSE_Q=1` loop is 13.344 static and 12.219 executed slots per `wmma`; the 12.41 that had been quoted for "ours" belongs to the `FUSE_Q=0` build, not the shipped kernel.

**Status:** confirmed (static census). The attribution of the LDS-read excess to P was later re-attributed to V's staging by F133.

### Deleting the `s_delay_alu` hints in the KV loop <sub>(track H2b)</sub>

**Question.** Does deleting the `s_delay_alu` hints in the KV body (edit E2, applied with `TRITON_KERNEL_OVERRIDE` at the `amdgcn` stage) make the kernel at least 1.5% faster?

**Result.** No. E2 is bit-identical but 3.09% to 3.75% slower at every cell, against a same-process A/A of 0.007% to 0.161%. Edit/shipped ratios are 1.035445 (N=8771 non-causal), 1.037462 (its repeat), 1.030879 (N=8771 causal) and 1.035631 (N=47520 non-causal), full call, one process per cell.

**Status:** refuted (the frozen bar was at least 1.5% faster).

### F130 — A wave-ballot lazy rescale <sub>(track H3)</sub>

**Question.** Can a per-wave ballot predicate replace the CTA-wide `tl.max` in the lazy softmax rescale of F121, at lower cost?

**Result.** The ballot lowers to one instruction and costs +5 instructions in total, against +82 for F121's reduction. The KV body goes from 496 to 435 slots (−61, −12.30%) with 236 VGPR, 47 SGPR and 0 spills, and the offline pipeline reproduces both pinned code objects byte for byte. But the candidate is invalid (F131), and it was never executed: no timing, no accuracy.

**Status:** refuted (the candidate was retracted; the ballot's cost claim stands).

### F131 — A wave-uniform predicate is not a CTA-uniform predicate <sub>(track H3)</sub>

**Question.** Is H3's per-wave ballot branch legal and exact?

**Result.** Two measured defects. The branch body contains two workgroup-barrier pairs but is entered per wave, so a workgroup barrier reached by only some waves does not complete: a hang that no accuracy gate can catch (F121's branch is fed by an LDS-broadcast scalar and is CTA-uniform, so its barrier is legal). And the accumulator has its warp bits along N while the row vectors `m_i` and `l_i` have both along M, so the wave that advances `m_i[r]` is not the wave that scales `acc[r,:]`. Any per-wave gate in this kernel must check, before building, for a workgroup barrier in the branch body and for one wave partition shared by every tensor it updates.

**Status:** confirmed (a reusable negative result; static, GPU-free). It retracts F130's candidate.

### F132 — A GPU-stage pre-registration for the ballot candidate

**Question.** How should the H3 candidate be tested on the GPU (an override at the `llir` stage, gates G0 to G8)?

**Result.** The pre-registration was withdrawn and never run: freezing gates for an invalid object would have turned a broken candidate into a runnable experiment. The gates stay in the file as a record, with a new liveness gate (a hard wall-clock cap and a post-call sync that must return) that would have caught F131's first defect. Its own gate G0 was also wrong, since the listings differ in 1 216 distinct lines rather than only at the predicate site.

**Status:** closed (withdrawn).

### F133 — Giving both dots the same wave layout <sub>(track H4)</sub>

**Question.** Does moving the PV dot's accumulator from the `#mma1` layout to the QK dot's `#mma` layout remove LDS round trips and speed up the KV loop?

**Result.** GPU-free. It compiles for gfx1201 with `vgpr_count` unchanged at 236 and 0 spills, and it deletes the two `128x1xf32` layout conversions (the alpha hop and the epilogue `acc*(448/l_i)`) and the P staging. But the KV body grows from 427 to 518 slots (+21.3%) and LDS reads from 43 to 72, while barrier pairs fall from 4 to 2. The V operand's per-wave read doubles (`ds_load_u8` 32 to 64), because under `#mma` each wave reads the whole tile. So the mismatch's real cost is the two hops, not the fp8 P staging, and the LDS-read excess attributed to P in H2a (and in the rationale of edit E3) is V's staging.

**Status:** refuted (a static regression of +21.3%). Corrects H2a's attribution.

### A frozen GPU stage for edits E1 and H4 <sub>(track H5)</sub>

**Question.** Do E1 (an `llir` override) and H4 (a `ttgir` override) reach their frozen speed predictions?

**Result.** Not run as written. The frozen predictions were H4 ≥ 1.05 (a falsification arm) and E1 between 0.965 and 0.990, with N=16384 pre-registered void and a liveness gate (run by a barrier-liveness checker whose controls validated: the shipped and H4 listings pass, the H3 one fails) before any launch. E1 was later run on its own under the H2 pre-registration (F134) and H4 was refuted statically (F133).

**Status:** open (never run as a combined stage; its H4 arm was refuted statically in F133 and its E1 arm was run separately in F134).

### F134 — E1: hoisting `qs*qk_scale` out of the KV loop <sub>(track H2d)</sub>

**Question.** Does an `llir`-stage hoist of `qs*qk_scale` out of the KV loop reach the frozen promotion bar of a ratio of 0.985 or lower at N=8771 non-causal?

**Result.** It removes exactly the predicted arithmetic (`fmul` 461 to 447 in the `llir`, −14, nothing else) and is a real, repeatable gain, but below the bar. E1/shipped is 0.986376 at N=8771 non-causal (A/A 1.000078), 0.987009 at N=8771 causal and 0.989889 at N=47520 non-causal, i.e. −1.36% / −1.30% / −1.01%, full call, same-process A/A, one process per cell. The fp64 accuracy check is bit-equal on 24 of 24 cells (8 seeds × 3 cells).

**Status:** refuted (rejected by the frozen bar; the pre-registered gate G0 also fails, so it is rejected on every reading).

### F135 — The `llir` override route is not surgical <sub>(track H2d)</sub>

**Question.** Does editing the `llir` stage change only the instructions the edit targets?

**Result.** No. The override replaces the result of the `llir` stage after it ran, so the backend re-runs instruction selection, register allocation and VOPD pairing on the substituted text. The edit removes 14 `fmul`, yet the KV body loses 9 / 10 / 6 slots at the three cells while 9–14 other opcodes move. A gate requiring that the diff contains only the predicted opcode changes cannot be met by an `llir`-stage rewrite; only a pure deletion, like E2's, leaves the rest of the schedule alone.

**Status:** confirmed (measured on 3 cells; the general statement is an inference from two).

### F136 — The conversion factor from instruction count to time does not transfer

**Question.** Does the project's `0.946% time per 1% instructions` conversion factor predict the gain from a body-only arithmetic removal?

**Result.** No. That factor comes from a 130-of-1171-instruction deletion on the `FUSE_Q=0` listing. E1 removes 0.17% of the real listing (2345 to 2341) yet buys 1.36% of wall time, 8× what the factor predicts, while against body slots (−2.10%) the factor over-predicts (1.99% against 1.36% measured). A per-class or per-slot model is needed, because the whole-listing count is the wrong normaliser.

**Status:** open (labelled inference).

## 13. SK1: a hand-written HIP kernel

A hand-written HIP kernel, SK1, designed from what the Triton work had learned. Variants were built one change at a time and judged against thresholds frozen before the run, with bit-identity gates and ratio-only timing. Two variants were wins: `sk1_t1` (one softmax rescale per 64 keys) and `sk1_t4a1` (mask and skip work behind one branch). Most of the rest were refuted.

### A design for a hand-written HIP kernel <sub>(track H6)</sub>

**Question.** What should a hand-written HIP attention kernel for gfx1201 look like, given what the Triton work showed?

**Result.** Design only. fp8 e4m3 Q and K with per-token scales, fp8 V, `smooth_k`, a causal split loop and ragged N with no per-iteration mask; a declared 161-VGPR budget and 10 build-time gates; and an estimated 5.8–5.9 `slots/wmma` (370 to 378 slots per 64 `wmma`) against 13.34 for the shipped Triton loop and 10.34 for PR #368. The estimates are labelled estimates, and nothing was built or run. The toolchain recipe (offline compile, `.hsaco`, `ctypes` loader) had been verified offline only.

**Status:** closed (design only). It was built as SK1 in H7, where the loop came out at 12.609 `slots/wmma`, far above the 5.8–5.9 estimate.

### SK1 built, accuracy-checked and timed <sub>(track H7)</sub>

**Question.** Does the hand-written SK1 kernel pass its gates, match the shipped kernel's accuracy, and beat the shipped kernel and PR #368?

**Result.** It builds as recorded (217 VGPR, 0 spills, 38 912 B of LDS) and passes its gates, and its error against an fp64 reference, as a ratio to the shipped Triton kernel's, is 0.9941 to 1.0017 over 18 of 18 cells with 0 NaN/Inf. Full-call SK1/shipped (below 1 is faster) is 0.8547 / 0.8042 / 0.7874 causal and 1.0494 / 1.0155 / 1.0024 non-causal at N=8771 / 16384 / 47520: 1.17× / 1.24× / 1.27× faster causally (1.28–1.37× faster than PR #368) and 4.9% / 1.6% / 0.2% slower non-causally. The kernel alone is faster at every N; the non-causal deficit is the prologue, padding and V transpose. Its KV loop is 12.609 slots per `wmma`, leaner than the shipped kernel's static 13.344 but fatter than PR #368's 10.344, so the census and the wall clock disagree.

**Status:** confirmed (the causal win; non-causal is a wash). Contrast F020's first HIP kernel, which was 11.3× slower than PR #368.

### SK1 with a fused prologue, the 16-cell scoreboard and a loop census <sub>(track H8)</sub>

**Question.** Does fusing the quantizer, transpose and pad prologue close SK1's non-causal deficit, and what does the KV-loop census say?

**Result.** A fused quantizer writes all four SK1 inputs in one pass and is bit-identical to H7's path on 10 of 10 gate cells and on all 16 timed cells. It passes its frozen prologue test (new/old prologue time 0.4049 at N=8771 and 0.4136 at N=47520 against a bar of 0.75; the magnitude is not claimed), which closes the non-causal deficit: full-call SK1/shipped is 0.9680 (H7: 1.0494) at N=8771 non-causal and 0.7385 (H7: 0.8547) causal. One of eight predictions was falsified (P5: that the pad's cost would show up as the aligned N=8768 being cheaper than N=8771; it was relatively more expensive, 0.9993 against 0.9680), and the loop census left the issue-bound against matrix-bound question open.

**Status:** confirmed (the fused prologue); nothing shipped.

### Variant sweep: one rescale per 64 keys <sub>(track H9; pre-registration F137)</sub>

**Question.** Do four pre-registered SK1 variants beat `sk1` by the frozen thresholds (kernel-only/`sk1_kernel` ≤ 0.98 at both N and both causal flags with `smooth_k` off, and full-call/`sk1_full` ≤ 0.99 at both production-shape non-causal cells)?

**Result.** `sk1_t1`, which does one softmax rescale per 64 keys and re-orders the QK, P and PV phases, is a real lever: kernel-only/`sk1_kernel` is 0.9179–0.9306 and the full call 5.1–7.8% faster at all 8 cells, so the frozen rule gives CANDIDATE. At the cell that motivated the track (N=47520 non-causal, `smooth_k` on), its kernel alone is 3.5% faster than PR #368's full call and its full call 1.05% faster. `sk1b` (0.9865–0.9970) and `sk1_c` (0.9897–0.9931) are non-levers and `sk1_2q` fails its offline gate (256 VGPRs, 71 and 99 spills). The KV loop shrinks from 807/64 = 12.609 to 757/64 = 11.828 slots per `wmma`, but the measured gain exceeds the census-predicted bound, which is refuted; the evidence is inconsistent with a matrix-bound loop (an inference).

**Status:** confirmed (`sk1_t1` is a candidate; not promoted at this point). Refutes the upper bound given for it in H8's census.

### Gates and a 16-cell scoreboard for the one-rescale variant <sub>(track H10a; pre-registration F138)</sub>

**Question.** Does `sk1_t1` pass the owed accuracy, edge-shape and 16-cell timing gates, and does the frozen decision rule make it a candidate?

**Result.** The frozen rule fires: `sk1_t1` is a candidate. Against `sk1` (below 1 is faster), kernel-only is at most 0.98 at all 6 `smooth_k`-off verdict cells (0.9191 to 0.9318) and full call at most 0.99 at N=8771 and N=47520 non-causal (0.9246 and 0.9206). fp64 accuracy passes 8 of 8 cells (`R_vs_sk1` 0.999675–1.000651, 0 NaN/Inf) and 112 of 112 edge cells, with 13 of 13 attempted refusals raising. At the cell that motivated H9, `sk1_t1`/PR #368 is 0.9650 kernel-only and 0.9885 full call; N=16384 is void.

**Status:** confirmed (a candidate; nothing promoted at this stage).

### Is the KV loop limited by the latency of its LDS reads? <sub>(track H11a; pre-registrations F139)</sub>

**Question.** Is the `sk1_t1` KV loop issue-bound, matrix-bound, or limited by the latency of its LDS reads?

**Result.** The primary ladder, which strips one pipe's instructions at a time, failed its own null gate: only `strip_vmul` was admissible (ρ 0.6702 and 0.6697, so the loop is not matrix-bound), and stripping LDS reads or gathers made the output non-finite. A finiteness-preserving repair (each LDS read replaced by writes of 0, net +48 slots) gave `LDS_LATENCY_LEVER`: `lds_swap` gains 13.61% and 14.52% at N=8771 and N=47520 non-causal even after paying the extra issue slots, an upper bound on what any latency-hiding scheme could recover. The frozen rule separates issue from non-issue, not LDS latency from LDS-pipe occupancy.

**Status:** confirmed, narrowed by H12.

### Making the K/V fragments arrive earlier by rescheduling <sub>(track H12; pre-registration F140)</sub>

**Question.** Can the KV-loop LDS fragment reads of `sk1_t1` be made to arrive cheaper by pure source reordering (three single-variable variants)?

**Result.** All three variants are NON_LEVER. Kernel/`sk1_t1` (below 1 is faster) at N=8771 and N=47520 non-causal is 0.9943 and 0.9938 for `t2c` (the per-row alpha broadcast moved earlier, fragment reads unmoved; the compiler re-allocated the whole loop), 1.0456 and 1.0487 for `t2b` (partial hoist) and 1.0793 and 1.0789 for `t2d` (half the V reads moved about 650 slots from their consumer): the further a read is hoisted, the slower the kernel, against frozen predictions of 0.965 to 1.000. Giving half the reads that slack recovered none of the 13.61% and 14.52% that deleting them gained in H11a, so the scheduling mechanism is refuted; whether the cost is LDS latency or LDS-pipe occupancy is still not separated.

**Status:** closed (the "issue the reads early" family). Narrows H11a's lever to LDS latency or occupancy without a scheduling fix.

### A wave-uniform skip of the softmax rescale <sub>(track H13; pre-registrations F141 and F142)</sub>

**Question.** Can a wave-uniform branch that skips the softmax rescale when `alpha == 1` speed up `sk1_t1` while staying bit-identical?

**Result.** The first build (`sk1_t3`) failed bit-identity (12 of 30 variant-cell pairs differ) because the branch let the compiler split a fused multiply-add; a repair (`sk1_t3b`) restores it on 30 of 30 pairs. A CPU census gives skip rates of 0.6914 at N=8771 non-causal and 0.9065 at N=47520 non-causal. Timed against `sk1_t1` (below 1 is faster, `smooth_k` off), `sk1_t3b` is 1.0296 at N=8771 and 1.0235 at N=47520 non-causal (1.0490 at the adversarial cell), so it is a NON_LEVER: the skip loses even at N=47520, where its census skip rate is 0.9065.

**Status:** closed (the repaired variant is slower; the rescale-skip route is closed for SK1).

### Fragments straight from global memory <sub>(track H14; pre-registration F143)</sub>

**Question.** Does deleting the LDS tile staging and the per-iteration barrier from `sk1_t1`, so that fragments are read straight from global memory, make the kernel faster?

**Result.** No. `sk1_g` is a NON_LEVER and its waves-per-EU follow-up `sk1_g1` is not admissible; the winner is none. Kernel-only `sk1_g`/`sk1_t1` (above 1 is slower) is 1.4588 at N=8771 non-causal and 1.4391 at N=47520, i.e. 44–52% slower, against a pre-registered range of [1.03, 1.12]: the predicted sign was right, the size under-predicted by 4–5×. `sk1_g1` is 1.4663 and 1.5067, and its lower VGPR counts (192 and 184) bought nothing.

**Status:** refuted (the direct-global route is closed).

### Roofline microbenchmarks: where the remaining time goes <sub>(track H15; pre-registration F144)</sub>

**Question.** Where does `sk1_t1`'s remaining time go: the WMMA pipe, the LDS pipe, or the instruction stream?

**Result.** Register-resident microbenchmarks give a compliant fp8 WMMA peak of 359.1 TFLOP/s against 199.6 for fp16 (ratio 1.799), `v_wmma` dependent latency of 27.0 ticks, and marginal issue costs of 8 (fp8) and 16 (fp16) ticks, so the instruction-rate ratio is 2×; the LDS read roof is 14.58 TB/s. `sk1_t1` runs at 0.407 and 0.409 of the fp8 peak at N=8771 and 47520. The ranking is WMMA work at 40.7% (irreducible), then the 488 extra instructions per 64 `v_wmma`, 36.1% of the kernel time at N=8771, the largest recoverable item; LDS at 47% of its roof and WMMA latency are not limiters. F002's 195.6 TFLOP/s loop rate was therefore an achieved rate, not a ceiling.

**Status:** confirmed.

### The instruction stream: three variants and two probes <sub>(track H16; pre-registration F145)</sub>

**Question.** Do three single-variable edits of `sk1_t1`, and two probes, cut H15's 488 extra instructions per 64 `v_wmma` enough to be at least 2% faster?

**Result.** One lever: `sk1_t4a1`, which puts the mask-and-skip select group behind one CTA-uniform `if (need_mask)`, bit-identical to `sk1_t1` on all 32 accuracy cells. Kernel-only `sk1_t4a1`/`sk1_t1` (below 1 is faster) is 0.9316 at N=8771 and 0.9284 at N=47520 non-causal, and 0.9253 to 0.9325 over the 12 non-void cells of the 16-cell re-run; at N=47520 non-causal with `smooth_k` on, full-call `sk1_t4a1`/PR #368 is 0.9216. The non-levers, kernel/`sk1_t1` at N=8771 and 47520, are `t4a2` (predicate deletion) 0.9985 and 0.9987, `t4b` (fold `c_q` into the exponent) 0.9868 and 0.9866, `t4c` (strength-reduced staging addresses) 1.0060 and 1.0060, and probe A (reverse `start_m` causal) 0.9849 and 0.9912. The gain is not separated from the loop reschedule that also lowered VGPRs from 240 to 224.

**Status:** confirmed (`sk1_t4a1` is a lever); shipped in H18 as the packaged kernel.

### F147 — The KV-loop instruction budget <sub>(track H19, step 1; GPU-free)</sub>

**Question.** Which lines of the KV loop account for the slots per 64 `v_wmma`, and where is the recoverable gap?

**Result.** Static slots per 64 `v_wmma`: `sk1_t1` 754 non-causal (11.781 per `wmma`) and 757 causal; the packaged base `sk1_t4a1` 776 (12.125); PR #368 1324 per 128 `v_wmma` (10.344). The base's largest lines are `WAIT` 108 (13.9%), `ADDR` 59 (7.6%) and `SCHED` 53 (6.8%), and the recoverable gap is the WAIT/SCHED line, 161 slots against a floor of 8. `need_mask` is true on only 1 of 138 key tiles at N=8771 and never executes at N=47520, so the static census overstates the base.

**Status:** confirmed.

### F148 — Three variants aimed at the wait/schedule line <sub>(track H19, step 2)</sub>

**Question.** Does any of up to three single-variable reorderings of `sk1_t4a1` reach kernel/base ≤ 0.98 at both N=8771 and N=47520 non-causal?

**Result.** No lever; the winner is none. (a) Dual-issue pairing: `sk1_t5a` and `sk1_t5d` are void, with no slot drop (776), and the residue is at most about 62 slots (8%). (b) A packed-fp16 softmax tail is refuted by the ISA (gfx1201 has no packed `exp2` and no `V_PK_SUB_F16`), so it was not built. (c) Load batching: `t5b`, `t5c`, `t5e` and `t5f` all reach 765 slots (11.953 per `wmma`, −1.4% static) and are bit-identical at all 8 cells. Timed kernel/base at N=8771 and N=47520 non-causal, `smooth_k` off: `t5b` 1.0016 and 1.0081, `t5e` 0.9979 and 1.0010, `t5f` 1.0020 and 1.0076, all NON_LEVER; `t5c` was not timed (`t5f` subsumes it).

**Status:** closed (the packaged object is unchanged).

### F149 — A design study for overlapping the matrix and softmax phases <sub>(track H20; GPU-free)</sub>

**Question.** Can a second-generation kernel (SK2) overlap the WMMA and softmax phases profitably on gfx1201, and which design is worth building?

**Result.** Design only. The premise that the matrix pipe needs to be kept busier is already refuted by measurement: `sk1_t4a1` runs it at 43.7% of its measured peak with dependent latency hidden, and 107 of the 161 `WAIT`/`SCHED` slots are ALU dependency-counter scaffolding, not memory waits. FlashAttention-3's overlap schemes do not port (gfx1201 lacks async WGMMA, TMA, `setmaxnreg` and sub-workgroup named barriers). Of three RDNA4-native structures costed against the real register and LDS budgets, S1a is refused on registers (264 VGPR), S2 on LDS (4 buffers need 67 584 B against a 65 536 B cap) and S3 three ways; only S1b survives, with a predicted net gain of +2.2% to +13.6% (point estimate +3.4%, below the 10% bar) that rests on assumed clock readings and an assumed overlap fraction.

**Status:** open (design only; nothing built or promoted).

## 14. Packaging, shipping and end to end

Packaging SK1 behind a flag, promoting it to default-on with automatic fallback, a real fp8 overflow found when a render went black and its fix, end-to-end tests inside Krea2, and widening the set of inputs the kernel accepts.

### Packaging `sk1_t1` behind an off-by-default flag <sub>(track H10b)</sub>

**Question.** Can the prebuilt `sk1_t1` code object ship as package data behind an off-by-default flag, with a wheel that is built and install-checked?

**Result.** Yes. The gfx1201-only code object ships in a new `sk1_backend` subpackage with a `ctypes` loader that has no hard-coded install path. With the flag off, output equals the shipped default (0 differing elements on all 3 smoke shapes); with it on, output equals the raw SK1 launch, with `R_vs_off` 0.99891 / 1.12783 / 1.02711 against a bar of 1.25, and NHD layout, head_dim 64 and `return_lse` fall back bit-identically. A wheel was built and its install-target check passes; the checking found and fixed one real defect (the opt-in defaulted `smooth_k=False` while the shipped HND path defaults to True).

**Status:** shipped (behind an off-by-default flag). Superseded by H18, which made the flag default-on and packaged `sk1_t4a1`.

### Promotion: `sk1_t4a1` packaged, default on with fallback <sub>(track H18; pre-registration F146)</sub>

**Question.** Do the nine frozen predictions pass, so that `sk1_t4a1` can be packaged and the SK1 backend made default-on on gfx1201 with automatic fallback elsewhere?

**Result.** All nine frozen predictions pass and the decision rule fires the promotion. Gates: 112 of 112 edge cells bit-identical to `sk1_t1`; 13 of 13 attempted refusals raise; a simulated non-gfx1201 architecture and a simulated load failure fall back bit-identically with one log line and no raise. Default path over the flag-off path (full call at (1,48,8771,128); below 1 is faster) is 0.824345 non-causal and 0.509416 causal, with A/A twins within 0.48%; the frozen prediction of 0.90–0.97 was wrong. Routing: with `SAGEATTN_SK1_BACKEND` unset the backend is on and non-strict (silent fallback, at most one stderr line per process); `=0` turns it off and it is never loaded; `=1` makes it strict and it raises if it cannot load. The packaged object is `sk1_t4a1.gfx1201.hsaco`.

**Status:** shipped (default on for gfx1201 with silent fallback elsewhere). Supersedes H10b's off-by-default state.

### F150 — An fp8 overflow in the shipped default, and its fix <sub>(track H21)</sub>

**Question.** Why did a Krea2 render (fp16, 8 steps) go black from step 8, and can the cause be fixed without changing any result that was already correct?

**Result.** The SK1 prebuilt path, default-on since H18, folded the softmax weight and V's scale into one unclamped fp8 conversion, `e4m3(p · max(amax_v, 1))` with `p ≤ 1`, so any key token with |V| above 448 overflowed and only the SK1 path went non-finite. On the black-render configuration through `sageattn(...)` the old object returned 7 721 600 non-finite elements; the repaired kernel (`sk1_t4a1s`) returns 0, is bit-identical to the old one on all 181 test cells, and is finite on all 288 outlier-sweep cells (the old kernel is non-finite on 99.986% of elements at the worst cell). The cost by event timer is 1.00401 / 1.00518 / 1.00032 against a 1.01 bar, passing on one disclosed stricter re-run. No captured workload exercised the bug (the largest |V| in 61 captures is 201.25), which is why every earlier test missed it.

**Status:** shipped. Corrects the object promoted in H18. The work included an audit of every fp8 conversion on the SK1 path (Q, K, V and P).

### F151 — End-to-end Krea2: the first attention-backend comparison <sub>(track H22)</sub>

**Question.** End to end in Krea2, how do ComfyUI's default SDPA, SageAttention 1.x int8 (`old`) and the H21 fp8 build (`new`) compare in speed, accuracy and image quality?

**Result.** Measured in bf16 with Krea2 Turbo (fp8-scaled) at 1 MP and 2 MP, 8 steps, 3 prompts × 3 seeds. Time per step relative to SDPA is 0.9691× for SageAttention 1.x (`old`) and 0.9906× for the H21 build (`new`) at 1 MP, and 0.9031× and 0.9101× at 2 MP; latent error against SDPA is 0.382 (`old`) and 0.186 (`new`) at 1 MP and 0.394 and 0.225 at 2 MP. ComfyUI's `smooth_k` gate was on for only about 4% of Krea2's attention calls (96 of 2304 at 1 MP, 81 of 2304 at 2 MP). However, `new` did not run SK1: the model computed in bf16, SK1 accepts only fp16, and it ran the native gfx12 fallback on every call.

**Status:** refuted for the `new` arm, whose numbers are fallback numbers (see H22b). The SDPA and `old` numbers stand.

### F152 — Krea2 end to end again, in fp16, with SK1 proven to run <sub>(track H22b)</sub>

**Question.** With the dtype corrected so that SK1 actually runs, how do SDPA, `old` and `new` compare end to end in fp16?

**Result.** H22's `new` arm never ran SK1, and its claim that empty error logs proved otherwise is retracted. In fp16 with a community Krea2 int8 fine-tune, time per step relative to SDPA is 0.9562× (`old`) and 0.9348× (`new`) at 1 MP and 0.8972× and 0.8554× at 2 MP, so `new` is the fastest arm: 6.5% and 14.5% faster than SDPA and 2.2% and 4.7% faster than `old`. SK1 served 4 368 of 4 368 self-attention calls with 0 non-finite values in 27 648 attention outputs, and `new` is closer to SDPA than `old` in all 4 run × resolution combinations; the strict gate failed on 624 contiguity fallbacks per slot (the text-fusion calls, 12.5% of calls). Cross-process bit-identity holds only if the text conditioning is identical: a 3.0e-6 difference in the text-encoder output moves the final latent by 0.04–0.58 relative RMS (median 0.18).

**Status:** confirmed. Overturns H22's `new`-arm results; H23 removed the contiguity fallbacks.

### F153 — Widening SK1's envelope to strided and NHD inputs <sub>(track H23)</sub>

**Question.** Can SK1's envelope be widened to strided HND and NHD inputs, bit for bit, without changing anything it already serves?

**Result.** All frozen gates pass: bit-identity on 184 of 184 cells, 181 of 181 on the unmodified H21 test and 288 of 288 outlier-sweep cells with 0 non-finite. The timing gate (widened/H21 ratio on inputs already served, bar 1.005) failed on its first run (1.0153 / 1.0076 / 1.0004) and passed on one disclosed stricter re-run (1.0011 / 1.0030 / 1.0003). The widened kernel takes 0.584 to 0.858 of the native fallback's time at NHD production shapes (N ≥ 8771) and is 1.13× to 5.1× slower at the tiny text-fusion shapes (at most 0.17 ms per call). In Krea2 end to end SK1 now serves 4 992 of 4 992 attention calls (it was 4 368), and median step time changes by +0.41% at 1 MP and −0.02% at 2 MP.

**Status:** shipped (the widened kernel is built into the wheel and stays default-on).
