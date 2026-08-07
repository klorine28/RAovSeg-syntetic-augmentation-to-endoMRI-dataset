# The λ Ablation Collapse — What Happened, How We Found It, and What It Means

A data-integrity investigation that started as "why do exp2, exp2_lam05,
and exp2_lam50 produce identical outputs" and ended with a single-line
code bug that voids the entire PatchGAN adversarial signal in **every**
1c and Phase 2 training run.

Companion to [OVARY_INTENSITY_ISSUE.md](OVARY_INTENSITY_ISSUE.md).
Discovered while regenerating the ovary-intensity mechanism figures with
n = 8 subjects and noticing that the exp2 and exp2_lam05 curves matched
to three decimal places across every per-subject row.

> **Reader beware:** §5 documents two hypotheses that turned out to be
> wrong. They're preserved as a record of what we believed at the time
> and how the diagnostic path corrected them. The actual root cause is
> in §6, confirmed empirically in §7, fixed in §8.

---

## 0. TL;DR

- **Root cause**: a single misplaced `.detach()` on `eps_pred` at
  [train.py line ~464](src/Generator/train.py) severed the computation
  graph between the generator's ε-prediction and the discriminator's
  output. As a result, `loss_g_adv.backward()` propagated **exactly zero**
  gradient into the generator's parameters, regardless of λ.
- **Empirically confirmed** by the `GRAD_DIAG` diagnostic in
  [scripts/tackle_lambda_collapse.sh](scripts/tackle_lambda_collapse.sh):
  `|grad_lam·L_adv| = 0.000000e+00` at every logged step past λ warmup,
  while `|grad_L_diff|` was healthy (~3e-2 to 7e-2).
- **The λ ablation is void** because λ never entered the generator's
  gradient. Multiplying an always-zero gradient by any scalar still gives
  zero — `exp2`, `exp2_lam05`, `exp2_lam50` produce identical weights and
  byte-identical synth volumes for this reason.
- **The scope is bigger than just Phase 2.** Any 1c or Phase 2 training
  run with PatchGAN enabled was affected. The discriminator itself trained
  fine on its own path; its output just never influenced the generator.
- **Fix**: change `eps_pred.detach().float()` → `eps_pred.float()` at the
  x0_hat estimation line. The D update below already applies its own
  `x0_hat.detach()` to sever the graph for D's backward, so the two
  update directions stay correctly isolated once the source `.detach()`
  is removed.
- **Silver lining**: the numerical DSC results reported in Phase 2
  (0.020, catastrophic) and Phase 1c still stand as measurements of what
  those checkpoints produce. What no longer stands is any interpretation
  that attributes an effect to PatchGAN adversarial pressure — because
  there wasn't any.

---

## 1. What we were trying to do — the intended experiment

Phase 2 tests whether cross-domain synthesis (`D1` for the generator's
image domain, `D2` for the discriminator's real reference) can produce
synthetic MRIs that improve the downstream RAovSeg DSC beyond the Phase 1
baseline. See [project_phase2_design](.claude/…/project_phase2_design.md).

The three planned λ variants tested a single knob — the peak weight of
the adversarial term in the generator's total loss:

```
loss_g_total = L_diffusion + λ_current * L_adversarial
```

with a warmup/ramp schedule `λ_current = ramp(step, warmup_end=10000,
ramp_end=30000, peak=λ_peak)`.

| Variant | `output_dir` | `lambda_peak` | Intended purpose |
|---|---|---|---|
| `exp2`       | `runs/exp2_d1_gen_d2_disc` | 0.01 | Baseline (inherits from 1c_spade) |
| `exp2_lam05` | `runs/exp2_lam05`          | 0.05 | 5× baseline — mild adversarial pressure |
| `exp2_lam50` | `runs/exp2_lam50`          | 0.5  | 50× baseline — strong adversarial pressure |

Config files: [exp2.yaml](src/Generator/exp2.yaml),
[exp2_lam05.yaml](src/Generator/exp2_lam05.yaml),
[exp2_lam50.yaml](src/Generator/exp2_lam50.yaml).

All three configs also set `resume: true` and share the same random seed
(42). All three train against the same data splits.

---

## 2. First hint — the plots were identical

The trigger was regenerating the ovary-intensity histograms on HPC with
n = 8 real train subjects and n = 8 synth subjects per variant. The
output table:

```
=== ovary voxel intensity summary (post RAovSeg normalize) ===
variant                          n_vox   mean  median   p10   p90  in_window%
exp2 (P2) (pooled)               83399  0.337  0.329  0.169 0.521    23.1%
exp2_lam05 (P2) (pooled)         83399  0.337  0.329  0.169 0.521    23.1%
```

The two "pooled" rows matched to three decimal places on every metric.
The per-subject rows matched, too — for every one of the 8 subject
volumes, `n_vox`, `mean`, `median`, `p10`, `p90`, and `in_window%` were
identical between exp2 and exp2_lam05. Not statistical coincidence:
identical to the digits printed.

The exp2_lam50 numbers were not printed in that run (we hadn't included
it in `--synth-dirs`), but subsequent tests would show it matches too.

**Suspicion at this point:** either the two directories point at the
same files (symlink, copy, or same-inode), or the assembly job for
lam05 accidentally used exp2's checkpoint.

---

## 3. Evidence chain — how we narrowed it down

Every step below was run in an interactive HPC session on Stanage
(`srun --pty --time=01:00:00 --mem=16G --cpus-per-task=4 bash`), on
node001, with the `synth_mri` conda environment active.

### 3.1 Are the synth volumes on disk actually independent files?

```bash
stat -c 'Inode=%i  Links=%h  Size=%s  %n' \
    synth_volumes/exp2/D2-900/D2-900_T2FS.nii.gz \
    synth_volumes/exp2_lam05/D2-900/D2-900_T2FS.nii.gz \
    synth_volumes/exp2_lam50/D2-900/D2-900_T2FS.nii.gz
```

Result:

```
Inode=144117936485935475  Links=1  Size=70042443  ...exp2/...
Inode=144117887949406727  Links=1  Size=70042443  ...exp2_lam05/...
Inode=144117983898269288  Links=1  Size=70042443  ...exp2_lam50/...
```

Three distinct inodes, `Links=1` each. Not hardlinks. Not symlinks.
Three genuinely separate files on disk.

### 3.2 Are the bytes truly identical?

```bash
cmp exp2/D2-900/D2-900_T2FS.nii.gz exp2_lam05/D2-900/D2-900_T2FS.nii.gz
cmp exp2/D2-900/D2-900_T2FS.nii.gz exp2_lam50/D2-900/D2-900_T2FS.nii.gz
cmp exp2_lam05/D2-900/D2-900_T2FS.nii.gz exp2_lam50/D2-900/D2-900_T2FS.nii.gz
cmp exp2/D2-915/D2-915_T2FS.nii.gz exp2_lam05/D2-915/D2-915_T2FS.nii.gz
```

All four `cmp` calls returned success (byte-level identical). We tested
D2-915 as well as D2-900 to rule out a subject-specific coincidence.

Also confirmed at the whole-set level:

```bash
(cd exp2       && find . -name '*_T2FS.nii.gz' -exec md5sum {} \; | sort) > /tmp/exp2_hashes.txt
(cd exp2_lam05 && find . -name '*_T2FS.nii.gz' -exec md5sum {} \; | sort) > /tmp/lam05_hashes.txt
(cd exp2_lam50 && find . -name '*_T2FS.nii.gz' -exec md5sum {} \; | sort) > /tmp/lam50_hashes.txt
diff /tmp/exp2_hashes.txt /tmp/lam05_hashes.txt   # empty
diff /tmp/exp2_hashes.txt /tmp/lam50_hashes.txt   # empty
```

All 32 synth volumes are byte-identical across all three variants.

### 3.3 Did the three assembly jobs actually run?

We were briefly worried that one of them had errored and a placeholder
was manually copied in. The Jul 8 assembly log for lam50 did show a
crash:

```
File "src/Generator/assemble_synthetic_volumes.py", line 380, in main
  ckpt = torch.load(args.ckpt, ...)
FileNotFoundError: '/mnt/parscratch/.../runs/exp2_lam50/ckpt/step_100000.pt'
```

But that was a Jul 8 job that ran *before* the lam50 training finished
(Jul 10 00:14). A **later** successful lam50 assembly ran on Jul 11:

```
logs/synth_lam50_10800960.err   601 bytes    (deprecation warnings only)
logs/synth_lam50_10800960.out  9413 bytes    (full run to completion)
```

The successful log printed:

```
[assemble] loaded ckpt step 100000, weights=EMA
[assemble] guidance=2.0, steps=100, iscs_alpha=0.8
[assemble] fixes — body_mask=True, histogram_match=True,
           resample_to_source=True, ovary_target_intensity=0.26
[assemble] [1/32] D1-001 → D2-900
[assemble] [2/32] D1-002 → D2-901
...
```

The exp2 and lam05 assembly logs printed the same header, with the same
`guidance=2.0, steps=100, iscs_alpha=0.8, ovary_target_intensity=0.26`.
All three ran with the same settings.

**Conclusion at this stage:** the three assembly jobs each ran to
completion. They each printed `loaded ckpt step 100000, weights=EMA`.
So the source of identicality is upstream of assembly.

### 3.4 Do the three checkpoint files differ?

```bash
md5sum runs/exp2_d1_gen_d2_disc/ckpt/step_100000.pt \
       runs/exp2_lam05/ckpt/step_100000.pt \
       runs/exp2_lam50/ckpt/step_100000.pt
```

Result:

```
effab06b69ac8282f004b8cf76e7111b  runs/exp2_d1_gen_d2_disc/ckpt/step_100000.pt
0f8d1890ef1f39dbfbe138dcce953b88  runs/exp2_lam05/ckpt/step_100000.pt
4e22d87ea7019b672c64e2924ade019e  runs/exp2_lam50/ckpt/step_100000.pt
```

Three different file-level md5 hashes. That would suggest the
checkpoints are genuinely distinct. But the sampled output being
byte-identical requires the *weights* to be identical. So we compared
tensor content directly:

### 3.5 Are the weights inside the checkpoints identical?

```python
import torch, hashlib

paths = {
    "exp2_d1_gen_d2_disc": "runs/exp2_d1_gen_d2_disc/ckpt/step_100000.pt",
    "exp2_lam05":          "runs/exp2_lam05/ckpt/step_100000.pt",
    "exp2_lam50":          "runs/exp2_lam50/ckpt/step_100000.pt",
}
ckpts = {n: torch.load(p, map_location="cpu", weights_only=False)
         for n, p in paths.items()}

def hash_state(sd):
    h = hashlib.md5()
    for k in sorted(sd.keys()):
        h.update(k.encode())
        h.update(sd[k].detach().cpu().numpy().tobytes())
    return h.hexdigest()

for n, c in ckpts.items():
    print(n, "EMA tensor hash:",   hash_state(c["ema"]))
    print(n, "MODEL tensor hash:", hash_state(c["model"]))
```

Result:

```
exp2_d1_gen_d2_disc EMA tensor hash:   6ed0dedb493744ad6e66c52214abb760
exp2_d1_gen_d2_disc MODEL tensor hash: ad7bb7aac21efee20f0efb101d035831
exp2_lam05          EMA tensor hash:   6ed0dedb493744ad6e66c52214abb760  ← same
exp2_lam05          MODEL tensor hash: ad7bb7aac21efee20f0efb101d035831  ← same
exp2_lam50          EMA tensor hash:   6ed0dedb493744ad6e66c52214abb760  ← same
exp2_lam50          MODEL tensor hash: ad7bb7aac21efee20f0efb101d035831  ← same
```

Every state-dict tensor is bit-identical across all three variants.
Sampling three random parameters confirmed layer-level equality:

```
unet.up_resblocks.1.1.norm1.mlp_shared.0.bias: same=True  |Δ|.max=0.000e+00
unet.up_resblocks.2.0.norm1.mlp_shared.0.bias: same=True  |Δ|.max=0.000e+00
unet.down_resblocks.0.1.conv1.weight:          same=True  |Δ|.max=0.000e+00
```

**Conclusion:** the three `.pt` files carry the same model weights.
The file-md5 differences come from the *non-weight* fields of the
checkpoint dict (`optim`, `step`, `cfg`, pickled RNG snapshots) —
things that don't affect inference.

### 3.6 So did the training runs actually train differently?

We looked at the tail of the training logs — the last 20 lines of steps
before `=== finished ===`, well after the warmup + ramp phase (both λ
schedules complete their ramp by step 30 000):

**`exp2_lam05_10701270.out` (peak=0.05, finished Jul 8 01:34):**

```
[step  99050/100000] L_diff=0.0010 L_adv=12.2009 L_D=0.0983 λ=0.0500 D_acc(r/f)=0.91/1.00 (2.65 it/s)
[step  99100/100000] L_diff=0.0047 L_adv=8.2781  L_D=0.0564 λ=0.0500 D_acc(r/f)=0.98/0.98 (2.65 it/s)
[step  99150/100000] L_diff=0.0059 L_adv=14.1021 L_D=0.0415 λ=0.0500 D_acc(r/f)=0.98/0.99 (2.65 it/s)
...
[step  99950/100000] L_diff=0.0009 L_adv=10.2901 L_D=0.0189 λ=0.0500 D_acc(r/f)=1.00/1.00 (2.65 it/s)
=== finished: Wed  8 Jul 01:34:06 BST 2026 ===
```

**`exp2_lam50_*.out` (peak=0.5, finished Jul 10 00:14):**

```
[step  99050/100000] L_diff=0.0010 L_adv=12.2009 L_D=0.0983 λ=0.5000 D_acc(r/f)=0.91/1.00 (2.61 it/s)
[step  99100/100000] L_diff=0.0047 L_adv=8.2781  L_D=0.0564 λ=0.5000 D_acc(r/f)=0.98/0.98 (2.61 it/s)
[step  99150/100000] L_diff=0.0059 L_adv=14.1021 L_D=0.0415 λ=0.5000 D_acc(r/f)=0.98/0.99 (2.61 it/s)
...
[step  99950/100000] L_diff=0.0009 L_adv=10.2901 L_D=0.0189 λ=0.5000 D_acc(r/f)=1.00/1.00 (2.61 it/s)
=== finished: Fri 10 Jul 00:14:34 BST 2026 ===
```

Note carefully: for every step from 99050 through 99950 (a random 900-
step window near the end):

- `L_diff` — same to the fourth decimal
- `L_adv` — same to the fourth decimal
- `L_D` — same to the fourth decimal
- `D_acc` — same to the second decimal on both real and fake
- Only `λ` (the printed schedule value) and `it/s` (trivial per-node
  timing) differ.

Two runs on different days, on different GPUs, with SBATCH scripts that
explicitly configure different `lambda_peak` values, produced the same
loss values at every step. That's not statistical noise — after
~90 000 gradient steps, even tiny stochastic differences would have
diverged the loss curves completely.

---

## 4. The training-loop code isn't obviously wrong (first read)

Our first look at [train.py:472](src/Generator/train.py#L472) focused on
the loss composition:

```python
loss_g_total = loss_diff + lam * loss_g_adv
optim.zero_grad(set_to_none=True)
loss_g_total.backward()
```

Structurally, this is correct. `lam` is computed from the config's
`lambda_peak`, and it multiplies `loss_g_adv` before backprop. Setting
`lambda_peak = 0.05` versus `0.5` *should* give the generator a 10×
larger adversarial gradient contribution — **if the gradient path from
`loss_g_adv` back to model parameters is intact**.

We didn't check that assumption at first. The bug wasn't a missing
multiplication; it was a broken graph upstream. §6 explains.

---

## 5. Two candidate root causes — BOTH WRONG (kept for record)

Before we ran the direct gradient diagnostic, we proposed two hypotheses
that would explain the observed byte-identity. Both turned out to be
wrong. They're kept below to show the reasoning trail.

### 5.1 (WRONG) Saturated D → adversarial gradient underflowed in AMP

At the tail of training, the log showed:

```
L_adv=12.2009  D_acc(r/f)=0.91/1.00
```

We argued the discriminator was near-perfect, so its gradient to the
generator was tiny (~10⁻⁶ via the sigmoid derivative), and under AMP
this would round to zero regardless of λ.

**Why this was wrong:** the gradient wasn't underflowing to zero — it
was *exactly* zero because the computation graph was severed at
`eps_pred.detach()` (§6). The sigmoid-saturation math is correct in
isolation, but it's a red herring here.

### 5.2 (WRONG) Deterministic training + shared resume + fixed seed

We argued that all three configs used seed 42 and `resume: true`, so
if the adversarial contribution was already ~zero (hypothesis 5.1),
deterministic re-execution would produce byte-identical trajectories.

**Why this was wrong:** determinism didn't cause the collapse; the code
bug did. The seed + resume aspects are true statements about the
configs, but they weren't the collapse's cause.

### 5.3 What we should have done earlier

Instead of theorising about AMP + determinism, we should have measured
the actual gradient contribution directly. Adding the `GRAD_DIAG` env
guard (§8) would have taken 30 minutes and given a definitive answer.
Lesson: **measure the invariant you're theorising about**, don't
speculate about it.

---

## 6. The actual root cause — a misplaced `.detach()`

In [train.py:464](src/Generator/train.py#L464), the single-step x0
estimate for the PatchGAN block was created with:

```python
# BUG
x0_hat = estimate_x0_from_eps(x_t, eps_pred.detach().float(),
                              train_sched, t)
```

The `.detach()` on `eps_pred` breaks the computation graph at creation
time. Every downstream tensor that flows from `x0_hat` — including
`d_fake_logits_for_g = discriminator(x0_hat, lbl_d_fake)` and
`loss_g_adv = generator_adv_loss(d_fake_logits_for_g)` — is functionally
a leaf tensor as far as `model.parameters()` are concerned.

When the training loop later computes:

```python
loss_g_total = loss_diff + lam * loss_g_adv
loss_g_total.backward()
```

only the `loss_diff` term produces non-zero gradients into
`model.parameters()`. The `lam * loss_g_adv` term produces exactly zero
gradient there, because the graph doesn't connect.

The comment three lines below the buggy line even acknowledged the
issue:

```python
# --- G adversarial loss: D should call our fakes 'real' ---
# x0_hat WITHOUT detach so gradient flows back into G.
d_fake_logits_for_g = discriminator(x0_hat, lbl_d_fake)
```

but the "without detach" claim was false — `x0_hat` was already
detached at creation. The comment was aspirational, not descriptive.

### Why the D update was fine

The D update path uses:

```python
d_fake_logits = discriminator(x0_hat.detach(), lbl_d_fake)
loss_d = discriminator_loss(d_real_logits, d_fake_logits)
loss_d.backward()
optim_d.step()
```

The `x0_hat.detach()` here is redundant given the bug (x0_hat was
already detached), but not harmful. D's own parameters get updated
normally from `loss_d.backward()`, which flows through
`discriminator(...)` — a valid graph.

So the discriminator trained. It learned to distinguish real from
generated. Its accuracy climbed toward `0.91/1.00` as we saw. But its
learning never influenced G, because the graph pointing to G was
severed at creation.

---

## 7. GRAD_DIAG confirms — `|grad_lam·L_adv|` is exactly zero

We ran a short (15 000-step) diagnostic training with a permanent
`GRAD_DIAG=1`-guarded logging block added to `train.py`
(scripts/tackle_lambda_collapse.sh manages this). Every log step past
λ warmup, the block computes `|∇L_diff|` and `|∇(λ·L_adv)|` separately
using `torch.autograd.grad`. Sample of what we saw:

```
[GRAD_DIAG] step=10025 lam=6.2500e-05 |grad_L_diff|=3.430474e-02 |grad_lam_L_adv|=0.000000e+00 ratio=0.000000e+00
[GRAD_DIAG] step=10050 lam=1.2500e-04 |grad_L_diff|=7.646020e-02 |grad_lam_L_adv|=0.000000e+00 ratio=0.000000e+00
[GRAD_DIAG] step=10075 lam=1.8750e-04 |grad_L_diff|=2.938816e-02 |grad_lam_L_adv|=0.000000e+00 ratio=0.000000e+00
[GRAD_DIAG] step=10100 lam=2.5000e-04 |grad_L_diff|=3.399197e-02 |grad_lam_L_adv|=0.000000e+00 ratio=0.000000e+00
[GRAD_DIAG] step=10125 lam=3.1250e-04 |grad_L_diff|=4.178875e-02 |grad_lam_L_adv|=0.000000e+00 ratio=0.000000e+00
[GRAD_DIAG] step=10150 lam=3.7500e-04 |grad_L_diff|=3.628077e-02 |grad_lam_L_adv|=0.000000e+00 ratio=0.000000e+00
[GRAD_DIAG] step=10175 lam=4.3750e-04 |grad_L_diff|=4.131101e-02 |grad_lam_L_adv|=0.000000e+00 ratio=0.000000e+00
[GRAD_DIAG] step=10200 lam=5.0000e-04 |grad_L_diff|=5.399683e-02 |grad_lam_L_adv|=0.000000e+00 ratio=0.000000e+00
...
```

Every single row shows the same signature: `|grad_lam_L_adv| =
0.000000e+00`. Not a small number that would round to zero in fp16 —
a literal zero returned by `torch.autograd.grad(lam * loss_g_adv, ...)`.
That output means there is no derivative path from `loss_g_adv` to
`model.parameters()`.

`L_adv` itself was healthy in the same log range (~1.4 to ~3.3),
confirming the discriminator was producing meaningful outputs. Only the
gradient chain to G was broken.

This ruled out both hypotheses in §5 and pointed directly at a graph
issue upstream of `loss_g_adv`. Grepping for `.detach()` in the
adversarial block found the single culprit.

---

## 8. The fix

Change one line in [train.py:464](src/Generator/train.py#L464):

```diff
-            x0_hat = estimate_x0_from_eps(x_t, eps_pred.detach().float(),
-                                          train_sched, t)
+            x0_hat = estimate_x0_from_eps(x_t, eps_pred.float(),
+                                          train_sched, t)
```

That preserves the graph from `eps_pred` (which is a function of
`model.parameters()`) forward to `x0_hat` → `d_fake_logits_for_g` →
`loss_g_adv`. The D-update block below still gets the correct isolation
via its own `x0_hat.detach()` before feeding into the D forward pass —
that line was already present, we just weren't relying on it.

**Status:** fix pushed to HPC. All future training runs against
`src/Generator/train.py` on HPC will have the adversarial gradient
correctly flowing to G.

### How to verify the fix works

Re-run the diagnostic:

```bash
sbatch scripts/_diag_lambda_grads.sbatch
```

Once it lands past step 10025 (past λ warmup, into ramp), expect
`|grad_lam_L_adv|` to be **non-zero and rising with λ**. Ratio should
climb from ~0 at step 10025 (λ ≈ 6e-5) to a meaningful value (10⁻³ to
10⁻¹ ballpark) as λ ramps toward its peak.

If any `[GRAD_DIAG]` line past step 10025 still shows
`|grad_lam_L_adv| = 0.000000e+00`, the fix didn't stick — investigate.

---

## 9. Affected experiments — full scope

The bug affects **every training run that used the PatchGAN block**
(`discriminator: patchgan` in the YAML, i.e., any 1c or Phase 2
config). The `if lam > 0.0:` gate on line 456 opens once λ warmup ends;
past that point, the buggy `.detach()` fires every step through the
end of training.

| Run | Config | Nominal λ_peak | Adversarial gradient into G | Effective equivalent |
|---|---|---|---|---|
| `exp1c_concat` | [exp1c_concat.yaml](src/Generator/exp1c_concat.yaml) | 0.01 | **zero** | Concat DDPM (= 1a with a useless D) |
| `exp1c_spade`  | [exp1c_spade.yaml](src/Generator/exp1c_spade.yaml)   | 0.01 | **zero** | SPADE DDPM (= 1b with a useless D) |
| `exp2` (= `exp2_d1_gen_d2_disc`) | [exp2.yaml](src/Generator/exp2.yaml)             | 0.01 | **zero** | Cross-domain DDPM only |
| `exp2_lam05`   | [exp2_lam05.yaml](src/Generator/exp2_lam05.yaml)   | 0.05 | **zero** | Same as exp2 above (identical weights) |
| `exp2_lam50`   | [exp2_lam50.yaml](src/Generator/exp2_lam50.yaml)   | 0.5  | **zero** | Same as exp2 above (identical weights) |

**Not affected**: `exp1a` and `exp1b` — no discriminator, no PatchGAN
block ever ran.

### Downstream artefacts derived from affected checkpoints

These don't need re-training, but any claim citing adversarial pressure
needs revision:

- **Assembled synth volumes**: `exp1c_concat/`, `exp1c_spade/`,
  `exp1c_spade_t022/`, `exp1c_spade_t028/`, `exp2/`, `exp2_lam05/`,
  `exp2_lam50/`. All 32 subjects per variant. The volumes are what those
  (silently non-adversarial) generators produced; plots and distributions
  computed from them remain valid measurements.
- **RAovSeg DSC evaluations**: all `raov_aug_*` seed replicates for
  concat/spade/t022/t028, all `raov_*_pathC_*` runs for exp2/lam05/lam50.
  DSC numbers are correct measurements; interpretations attributing
  differences to PatchGAN are void.
- **Tier 1 sweep**: uses the `exp1c_spade` checkpoint as its base
  generator. Tier 1 varies only assembly-time knobs, so trials still
  produce valid downstream DSC — but the base generator is effectively
  `1b`, not "SPADE + PatchGAN".
- **Mechanism figures**: `figures/fig_mech_ovary_hist.png`,
  `figures_v3_*/`. These are marginal intensity distributions of the
  synth output. They're valid representations of what those generators
  produced; they just can't be used to argue about adversarial effects.

### Documents that need corrections after the retrain

- `TIER1_TUNING_AND_EXPLAINABILITY.md` §7-onward attributes texture
  improvements to PatchGAN in 1c. That interpretation is void until
  1c is retrained.
- [docs_archive/RESULTS_2x2.md](docs_archive/RESULTS_2x2.md) and
  [docs_archive/EXP1C_SUMMARY.md](docs_archive/EXP1C_SUMMARY.md):
  1c-vs-1a/1b comparisons are not testing what the ablation title
  claims. They're comparing "with useless D" to "no D", so any observed
  difference is from seed/data-order variation, not adversarial signal.
- `project_phase2_result` memory entry: DSC = 0.020 stands as a fact.
  The framing that "PatchGAN cross-domain synthesis catastrophically
  collapses" needs the caveat that it collapses even *before* PatchGAN
  gets a chance to intervene — the collapse is in the DDPM+cross-domain
  pairing, not in the adversarial dynamics.

---

## 10. Implications for the dissertation

### 10.1 What's void

| Artefact | Status | Why |
|---|---|---|
| `synth_volumes/exp2_lam05/` (32 volumes) | Identical to exp2 | Same weights, same seed, same assembly |
| `synth_volumes/exp2_lam50/` (32 volumes) | Identical to exp2 | Same weights, same seed, same assembly |
| `lam05_dsc_summary.json`         | Duplicate of exp2 DSC | Segmenter fed identical inputs |
| `lam50_dsc_summary.json`         | Duplicate of exp2 DSC | Same |
| `lam05_pathC_dsc_summary.json`   | Duplicate                | Same |
| `lam50_pathC_dsc_summary.json`   | Duplicate                | Same |
| `exp2_pathC_dsc_summary.json`    | Correct measurement, but no meaningful comparison partners | The other two collapse into it |
| Commit `50b9c56` ("Phase 2 pathC results + repo organisation") | Numerically correct but the λ interpretation is wrong | Same three numbers reported as different |
| Commit `ec158b6` ("Phase 2 pathC (lam05, lam50) scripts and metrics") | Scripts fine, metrics collapse | Same |
| Any table/figure caption mentioning "1c PatchGAN texture improvement" | Interpretation void | PatchGAN never fed gradient to G |

Anything reporting "λ = 0.05 vs λ = 0.5 vs baseline λ = 0.01" as three
independent conditions in a table is currently reporting one number
three times.

### 10.2 What's not void

- **Real-D2 mechanism figures** (baseline ovary intensity distributions,
  intensity-dial sweeps). These don't depend on how the generator was
  trained — they measure post-hoc distributions of samples that exist on
  disk. Still valid.
- **The Path B ovary intensity rescale** and its effect on
  `in_window %`. This is an assembly-time post-processing step,
  completely independent of the training bug.
- **The `1c_spade_t022` and `1c_spade_t028` synth volumes** — the
  intensity-dial variants of the affected `1c_spade` checkpoint. They
  demonstrate the Path B mechanism working; that demonstration is
  independent of whether PatchGAN was actually training.

### 10.3 Suggested framing after the retrain

Instead of "λ = 0.01 / 0.05 / 0.5 gave DSC 0.020, 0.020, 0.020"
(which invites the reader to ask "why identical?"), frame as:

> *"An earlier version of our training pipeline contained a bug that
> severed the discriminator's gradient path to the generator (a
> misplaced `.detach()` on the ε-prediction fed into the single-step
> x̂₀ estimate). All 1c and Phase 2 experiments trained with an
> effectively idle PatchGAN — its parameters updated correctly, but
> its output never influenced the generator. This was discovered when
> the three λ variants for the cross-domain Phase 2 ablation produced
> byte-identical outputs; a direct gradient measurement (§8) confirmed
> `|∇(λ·L_adv)| = 0` at every training step. The bug was fixed and
> the affected experiments re-run; results reported here are from the
> corrected runs."*

That's a defensible one-paragraph statement. The old results can be
preserved as an appendix labelled "measurements from the pre-fix
pipeline" if useful for reproducibility.

### 10.4 What to do with the redundant / superseded files

**Recommended:** keep them on HPC for reproducibility but rename to make
the situation explicit. Something like:

```bash
cd /mnt/parscratch/users/$USER/synth_mri/synth_volumes/
# Rename the identical lam05/lam50 dirs — they're not really variants
mv exp2_lam05 exp2_lam05_PREFIX_BUG
mv exp2_lam50 exp2_lam50_PREFIX_BUG
cat > exp2_lam05_PREFIX_BUG/README.txt <<'EOF'
This variant was intended to test λ_peak = 0.05, but the training
pipeline had a graph-severance bug (eps_pred.detach() at train.py:464)
that produced zero adversarial gradient into G regardless of λ. Weights
are identical to exp2_d1_gen_d2_disc. See:
  EndometriosisDataset/LAMBDA_ABLATION_COLLAPSE.md
EOF
cp exp2_lam05_PREFIX_BUG/README.txt exp2_lam50_PREFIX_BUG/README.txt
```

Similarly for 1c_concat and 1c_spade after their retrains land — move
the old checkpoints and synth into `_PREFIX_BUG` suffixed dirs so
future-you can distinguish pre-fix from post-fix artefacts.

---

## 11. Retraining plan

The bug affects 5 training runs (§9). The minimum retrain set depends
on what you want the dissertation to claim.

### 11.1 Minimal set — just the Phase 2 λ ablation

3 training runs, ~36 h each, submittable in parallel:

- `exp2_fixed` (λ=0.01, seed 42)
- `exp2_lam05_fixed` (λ=0.05, seed 43 — different from exp2 to break determinism as a belt-and-suspenders measure)
- `exp2_lam50_fixed` (λ=0.5, seed 44)

Once complete, verify all three checkpoints diverge at the tensor
level:

```bash
python - <<'PY'
import torch, hashlib
def h(sd):
    hh = hashlib.md5()
    for k in sorted(sd.keys()):
        hh.update(k.encode()); hh.update(sd[k].detach().cpu().numpy().tobytes())
    return hh.hexdigest()
for p in [
    "runs/exp2_fixed/ckpt/step_100000.pt",
    "runs/exp2_lam05_fixed/ckpt/step_100000.pt",
    "runs/exp2_lam50_fixed/ckpt/step_100000.pt",
]:
    c = torch.load(p, map_location="cpu", weights_only=False)
    print(p, "EMA:", h(c["ema"]))
PY
```

Three distinct hashes → the fix worked and the ablation is real. Then
re-assemble volumes and re-run DSC eval.

### 11.2 Extended set — Phase 1c too

Add `exp1c_concat_fixed` and `exp1c_spade_fixed`. 5 training runs total.
This is what you'd want if the dissertation claims anything about
1c-vs-1a or 1c-vs-1b — those claims need Phase 1c to have actually
had adversarial gradient during training.

### 11.3 Full set — Phase 1c + Phase 2 + new Tier 1

Add a fresh Tier 1 sweep against the newly-adversarial `1c_spade_fixed`
checkpoint. The 20 currently-queued Tier 1 jobs are still valid
measurements against the pre-fix `1c_spade`, but if the fix makes 1c
meaningfully different from 1b, the tier-1 search space to explore may
also differ.

### 11.4 Recommended: start with 11.1

Even without 1c retrains, a working Phase 2 ablation would answer the
research question the doc set out to answer ("does λ matter for
cross-domain synthesis?"). If Phase 2 remains catastrophic at every
λ, then the story becomes "we fixed the bug that had been masking the
ablation and confirmed the collapse is not λ-tunable — the failure
mode is deeper." That's a publishable negative result. If Phase 2
recovers at some λ, that becomes the headline of a much better story.

Only expand to 11.2/11.3 if Phase 2 recovery makes 1c-vs-1a/1b
comparisons load-bearing.

---

## 12. Discovery path — reproducing the diagnosis

For future-you or a reader who wants to verify: the whole trail runs
in ~2 h on an HPC interactive session, most of which is the diag
training. The key commands, in order:

```bash
# 1. Cheap sanity check: any subject, byte-level
cd /mnt/parscratch/users/$USER/synth_mri/synth_volumes/
cmp exp2/D2-900/D2-900_T2FS.nii.gz exp2_lam05/D2-900/D2-900_T2FS.nii.gz
cmp exp2/D2-900/D2-900_T2FS.nii.gz exp2_lam50/D2-900/D2-900_T2FS.nii.gz

# 2. Whole-set check
for v in exp2 exp2_lam05 exp2_lam50; do
    (cd $v && find . -name '*_T2FS.nii.gz' -exec md5sum {} \; | sort)
done > /tmp/all_hashes.txt
# Should be three columns of identical hashes if collapse

# 3. Confirm at the checkpoint tensor level
python - <<'PY'
import torch, hashlib
paths = [
    "/mnt/parscratch/users/ijp25lg/synth_mri/runs/exp2_d1_gen_d2_disc/ckpt/step_100000.pt",
    "/mnt/parscratch/users/ijp25lg/synth_mri/runs/exp2_lam05/ckpt/step_100000.pt",
    "/mnt/parscratch/users/ijp25lg/synth_mri/runs/exp2_lam50/ckpt/step_100000.pt",
]
def h(sd):
    hh = hashlib.md5()
    for k in sorted(sd.keys()):
        hh.update(k.encode())
        hh.update(sd[k].detach().cpu().numpy().tobytes())
    return hh.hexdigest()
for p in paths:
    c = torch.load(p, map_location="cpu", weights_only=False)
    print(p, "EMA:", h(c["ema"]))
PY

# 4. Direct gradient measurement — the definitive check
./scripts/tackle_lambda_collapse.sh diag --submit
# Wait for job to finish (~2 h queued + ~80 min run)
grep '\[GRAD_DIAG\]' logs/diag_lambda_grads_*.out | tail -40
# ratio should be zero on unfixed train.py, non-zero on fixed
```

If all four checks agree — byte-identical files, byte-identical
weights, byte-identical loss trajectories, and `|grad_lam·L_adv| = 0` —
the bug is present. The fix in §8 addresses it directly.
