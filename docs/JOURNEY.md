# How this was built

This is the story of the project, roughly in the order it happened: what was tried, what the
measurements said, and why each turn was taken. Every claim here points to an entry in
[FINDINGS.md](FINDINGS.md), which has the numbers. [METHOD.md](METHOD.md) describes how those
numbers were taken.

---

## 0. The starting point

The goal was simple to state: fast, accurate quantized attention on an AMD Radeon RX 9070 XT
(`gfx1201`, RDNA4) under Windows, for diffusion models in ComfyUI.

SageAttention already had an RDNA4 port in waiting: PR #368 on the upstream repo, a HIP port of
SageAttention 2 to gfx12. It became the baseline. The aim was not just to run SageAttention on this
card, but to **beat PR #368 on its own hardware** without giving up accuracy.

Three facts about the chip shaped everything after:

- Its matrix instructions (WMMA) run fp8 at twice the fp16 rate: 2048 vs 1024 FLOP per clock per
  CU. Measured with a register-resident loop, that is 359 vs 200 TFLOP/s.
- A wave32 matrix instruction works on a 16×16×16 tile, and the operand layouts are fixed by the
  hardware. They are not what most CUDA-derived code assumes.
- The register file decides occupancy: 1536 VGPRs per SIMD in steps of 24. A kernel that keeps
  under 240 VGPRs per wave gets 6 waves per SIMD. One register more drops it to 5.

## 1. A Triton kernel, and the first lessons

The first kernel was written in Triton: an fp8 FlashAttention-style forward pass with per-token
scales for Q, K and V. It used the hardware fp8 convert instruction (`v_cvt_pk_fp8_f32`) through
inline assembly, after a compiler abort had first been misread as "this chip has no fp8 convert"
(see METHOD.md, mistake 5).

Early head-to-heads against PR #368 were mixed. One shape won: N=8192 causal, by 1.05× (F010).
Every other shape lost, by up to 1.6×. The work that followed was mostly diagnosis.

**What worked.** Splitting the causal loop into a mask-free main part and a masked diagonal
(`SPLIT_LOOP`) was the first lever that turned into wall time: a 1.29–1.38× causal speed-up after
re-verification, and a causal win over PR #368 at every tested shape (F025, F029, F030, F038). A
fast path for sequence lengths that are a multiple of the tile removed per-iteration masking from
non-causal attention. It was adopted as the default only after an accuracy check across 504
comparisons showed it was not less accurate (F068).

**What it found that was wrong.** The shipped causal path read past the end of K for some
(sequence length, tile) pairs and produced NaN, including at the production length 8768 (F062).
The loop bound was not clamped. It was fixed, and the fix became part of every later test.

**Where it stopped.** At the real production shape (1×48×8771×128, non-causal) the Triton kernel
was still 1.22× slower than PR #368 (F041). The first explanation was software pipelining:
Triton emitted no asynchronous global-to-LDS copies on this chip. That explanation was wrong.
Reading PR #368's source and its compiled code object showed a kernel with no pipelining at all
(F053–F055). It wins on loop structure instead:
- it walks the keys 64 at a time, where ours walked 16 (138 loop iterations against 549 at
  N=8771);
- it issues four times as many matrix instructions per iteration;
- it loads each K fragment once for two query groups.

A tile sweep confirmed that Triton could not follow it there: wider Triton tiles were slower, not
faster (F052).

## 2. Accuracy, measured on real activations

Speed only matters if the output is right, so the next question was how much error each scheme
adds. It was measured on real Q/K/V tensors captured from inside Krea2 and Flux2-Klein, against an
exact reference (F058).

The answer was more interesting than "better" or "worse":

- PR #368 computes Q·K in int8, with one K scale per 64-token block. On well-behaved layers that
  is the more precise scheme, by a wide margin.
- On a layer with a few extreme keys (Krea2's first block is one), a single outlier sets the scale
  for its whole block, and the error explodes: up to 55× worse than per-token fp8, and up to 200×
  once `smooth_k` was added to per-token fp8.
- Per-token fp8 scaling never has a bad regime. Its error is flat across blocks, models, timesteps
  and resolutions.

One correction came much later. PR #368's kernel quantizes Q to int8 as well, and the comparison
above had left that out. With it fixed, PR #368's lead on calm layers shrank from up to 11× to
3–6× (F154).

Two follow-ups settled the defaults. Subtracting the per-head key mean before quantizing
(`smooth_k`) is exact in real arithmetic and improved the per-token kernel on 10 of 10 captures, by
3.2× on average (F063). Per-channel V scaling also improved accuracy, but it cost 7–11 % in speed
and was closed (F100–F102).

## 3. Editing the compiler's output

Before writing a kernel from scratch, I tried the cheaper route: take Triton's compiled assembly
and edit it by hand. Delete scheduling hints, skip the softmax rescale when no maximum changed, keep
the accumulators in a different layout.

Every edit was either slower or unsafe. Deleting scheduling hints cost 3–4 %. A wave-level skip of
the rescale would have placed a barrier inside a branch that only some waves take, which can hang
the GPU. A layout change added 21 % more instructions. The edits that did help stayed below the
bar. That closed the route (tracks H1–H5 in FINDINGS.md).

## 4. Writing the kernel by hand

The hand-written kernel, called SK1 here, is a single HIP source compiled to a code object for
`gfx1201`. It is loaded at runtime through the HIP driver API, so the package needs no compiler on
the user's machine. Its design came straight out of section 1:

- **64 keys per loop iteration,** as two 32-key sub-tiles, with 8 waves of 16 queries each
  (128 queries per workgroup). K and V tiles are double-buffered in LDS with one barrier per
  iteration.
- **The attention weights never leave registers.** The scores are computed transposed (Kᵀ·Q
  instead of Q·Kᵀ). In the hardware's operand layouts, the result fragment of that product is then
  exactly the input fragment the P·V product needs. No shuffle through shared memory is required.
  This only works because the layouts were measured on the device first. A small probe kernel
  confirmed them bit for bit, and it also showed that the B operand is K-major, contrary to the
  first design sketch.
- **V is stored transposed** (head-dim-major), so each lane's eight fp8 values for a fragment are
  contiguous in memory and in LDS.
- **One prologue pass** quantizes K and V, writes the per-token scales, pads to the tile and
  transposes V, all in a single kernel. That made the prologue 2.47× faster than the first version.

The first full scoreboard already won every causal shape by a wide margin. It still lost two
non-causal cells, both with `smooth_k` on (N=8768 and N=47520).

## 5. Making it faster, one measured variant at a time

From there every change was a single-variable variant of the current kernel. Each had a written
prediction and an offline gate: registers, spills, LDS, barrier safety and instruction counts, all
before any GPU run. Then came a bit-identity or accuracy check, then same-process interleaved
timing.

Two variants won:

- **One softmax rescale per 64 keys instead of per 32** (`sk1_t1`): 7–8 % faster. It turned both
  losing cells into wins at the production shapes.
- **Mask and skip work behind a single uniform branch** (`sk1_t4a1`): about 7 % faster, bit for bit
  identical. That is the kernel the package ships.

Most variants lost, and each loss narrowed the search:

- the K and V reads were already paired by the compiler;
- issuing reads earlier made the kernel slower;
- skipping the rescale lost even where it skipped 91 % of the time;
- reading fragments straight from global memory was 1.44–1.52× slower;
- higher occupancy did not help;
- reordering the source changed nothing measurable.

A roofline measured on this card put the kernel at about 0.41–0.44 of the fp8 peak. An instruction
budget showed why. The arithmetic is already at its minimum; the gap is wait and scheduling
instructions the compiler places itself. Source-level tuning is therefore exhausted. What remains
is structural: overlapping one tile's matrix work with another tile's softmax. A design study
predicts +3–8 % for that, to be confirmed by a cheap probe before anyone builds it.

## 6. Shipping it

The kernel first shipped behind a flag that was off by default, inside a normal `sageattention`
wheel. It now runs by default where it applies: gfx1201, fp16, head_dim 128, no mask. It falls
back silently to PR #368's kernel everywhere else, including other GPUs, a missing runtime and
unsupported shapes. Every launch passes a contract check on shapes, dtypes and strides first,
because the kernel is launched through raw `ctypes` and nothing downstream bounds-checks.

## 7. The bug a real render found

The first real render with the new default came out black from step 8 onward. Every earlier test
had passed.

The cause was in how the attention weights are quantized. Each weight is multiplied by its key's
V scale and converted to fp8 in one step. That product can exceed fp8's maximum of 448 when a
single token has |V| > 448 and receives a large weight. The conversion then produces NaN, and NaN
spreads through the image. No earlier test had a V that large. The real model did, at its last
step.

The fix scales each head's V down just enough to keep that product in range, and undoes the scale
in the epilogue. When no value exceeds 448, the output is bit for bit identical to before. The
cost is 0.03–0.52 %. Since then, every conversion in the kernel is swept past its format's limit as
a standing test (METHOD.md §2.4).

## 8. Testing where it matters

Kernel benchmarks do not show what a user sees, so the last step was a render test: Krea2 in
ComfyUI's own attention path, three prompts × three seeds, 1 MP and 2 MP. It had three lessons of
its own:

- **The allocator.** With about 13 of 16 GB in use, PyTorch's allocator thrashed and doubled the
  step time. `expandable_segments:True` fixed it, and every earlier large-model timing taken
  without it had to be discarded.
- **Prove which kernel ran.** The first run loaded the model in bf16. The new kernel only accepts
  fp16, so it quietly fell back on every call, and the "new" numbers were the fallback's numbers.
  They were withdrawn. The rerun counts which kernel served each call and fails the run if any call
  falls back.
- **8-step samplers are chaotic.** A 3·10⁻⁶ change in the text conditioning alone moves the final
  latent by 4–58 %. Single-image comparisons mostly measure that divergence. Accuracy is better
  judged per call, against an exact reference.

In the clean run the package was the fastest backend at both resolutions: 5.4 % and 13.7 % faster
per step than PyTorch SDPA, and 2.4 % and 5.0 % faster than SageAttention 1.x. Its images were
also closer to full-precision attention than SageAttention 1.x's. The final change widened the
kernel's input envelope to strided and NHD inputs, which is what video models such as Wan send.

## 9. int8, bf16 and new defaults

Two things had been waiting since the kernel first shipped.

The first was int8 Q·K. On this chip the int8 and fp8 matrix instructions run at the same rate, so
int8 costs nothing in the matrix work, and with one scale per token it is more precise than fp8.
Across ten real captures it beat per-token fp8 on nine and PR #368 on all ten (F154). Inside a
real render it served every call, its per-call error was 0.82× the fp8 kernel's, and the step time
did not change (F157). The kernel alone is about 2.5 % slower. Its prologue quantizes only K and V,
which pays that back at image sizes but not quite at video lengths: at 47k tokens the call is 1.7 %
slower. The precision was worth that, so int8 became the default for fp16, with the fp8 kernel one
environment variable away (F159).

The second was bf16. Many models run in bf16, and until now every one of their calls fell back to
PR #368. The bf16 kernels come from the same source as the fp16 ones, templated on the dtype, and
the fp16 objects stayed byte for byte as they were. They run 1.3× to 2.2× faster than PR #368's
bf16 path (F158). They were held back once: on the most extreme outlier inputs, some cells came
out non-finite. A closer look showed those were inputs where exact fp32 attention overflows too.
The rule was restated before the rerun as "no worse than exact fp32 attention", the kernel passed
it, and bf16 went on by default (F159).

Memory showed up again. PR #368's bf16 path measured almost twice as slow in one run as in another,
with the same code. The difference was how full the GPU's memory was during the run: under
pressure its allocator thrashes, while the hand-written kernel's time did not move. A baseline has
to be measured in the same memory state as whatever it is compared with.

Not everything worked:

- **head_dim 64.** A port of the D=128 kernel built clean and was bit-exact across its variants,
  but even the variants that reached higher occupancy ran 1.55–2.7× slower than PR #368's D=64
  kernel. It is not used (F156).
- **The overlap rewrite.** Three cheap probes were meant to decide whether overlapping one tile's
  matrix work with another tile's softmax was worth building. The main probe could not be built as
  a fair comparison, because the compiler's own wait instructions change with the schedule. The
  other two showed that barriers are cheap here (0.6–0.9 %) and that `s_setprio` costs time
  instead of saving it. The rewrite was not started (F155).

## 10. What is next

- **Video models end to end.** The kernels now accept what video models send (strided and NHD
  layouts, bf16), but every end-to-end test so far used image models.
- **Per-call overhead.** At the tiny text-fusion shapes, launch and quantization cost more than the
  attention itself (F153). It is at most 0.17 ms a call, but there are many such calls.
- **head_dim 64** needs its own design rather than a port of the D=128 kernel.
