# Exp 1b — 2D SPADE-Conditioned DDPM: Implementation Notes

A record of what was built for Exp 1b, the bugs encountered, design
decisions, and the rationale behind each. Companion to `EXP1A_NOTES.md`.
The forward-looking *planning* notes were originally in this file; they
have been kept as Section 12 at the bottom for reference and to track
which planned decisions were followed.

---

## 1. What Exp 1b is

A 2D conditional DDPM that **replaces** concat conditioning with SPADE
(Spatially-Adaptive Normalization, Park et al. 2019). SPADE injects label
information into the decoder at every level via spatially-varying
per-channel γ and β modulation, instead of being concatenated to the noisy
image at the input.

**Pure SPADE**: model input is the 1-channel noisy image only. The label
enters through SPADE modules in the bottleneck and decoder ResBlocks.

Final architecture:

| Component | Value |
|---|---|
| U-Net | Custom `DiffusionUNetSPADE` (we hand-built it; see Section 4) |
| Levels (resolutions) | 4 — 512², 256², 128², 64² |
| Channel widths | 64, 128, 256, 256 |
| ResBlocks per level | 2 in encoder, 2 in decoder |
| Self-attention | Deepest level only (64²); same memory budget as 1a |
| Norm — encoder | Standard affine GroupNorm(32) |
| Norm — bottleneck + decoder | **SPADE**, GroupNorm(32 affine=False) + label-conditioned γ/β |
| Total params | **23.9 M** (1a was 25.3 M — comparable backbone capacity) |
| Conditioning | Pure SPADE — `in_channels: 1`, label routes via SPADE only |

Everything else carried over from 1a unchanged for ablation fairness:

| | |
|---|---|
| Optimiser | AdamW, lr 1e-4 |
| Batch size | 4 |
| Total steps | 80,000 |
| Mixed precision | bf16 autocast |
| CFG dropout / guidance | 0.1 / 3.0 |
| EMA decay | 0.9999 |
| Diffusion timesteps / schedule | 1000 / linear_beta |
| Sampling | DDIM, 50 inference steps |
| Hardware | 1× A100 80 GB on Sheffield Stanage |
| Wall clock | ~10-11 h |

---

## 2. The ablation contract

> 1a vs 1b answers: **does spatially-adaptive normalisation produce better
> label-image alignment than channel concatenation, holding everything else
> equal?**

For this to be a fair comparison, every choice that isn't "concat vs
SPADE" must match between 1a and 1b. The list of things we kept locked:

- 6-channel one-hot label (outside_body, uterus, ov_L, ov_R, em, body_other)
- Same train/test split (32 train ∪ 8 sacred test)
- Same hyperparameters, same total step budget
- Same CFG, EMA, fixed_labels resampling
- Same body silhouette preprocessing (until v2 — see Section 9)

---

## 3. Files created for 1b

| File | Role |
|---|---|
| `src/Generator/spade.py` | The SPADE module: GroupNorm + label-conditioned γ/β |
| `src/Generator/unet_spade.py` | Custom 4-level SPADE-conditioned U-Net (~330 lines) |
| `src/Generator/model.py` (additions) | `SPADEConditionedDDPM` wrapper, `_BaseConditionedDDPM` shared sampler, `build_model_from_cfg()` dispatcher, `build_unet_spade()` factory |
| `src/Generator/exp1b.yaml` | Config — mirrors `exp1a.yaml` with `model.type: "spade"`, `in_channels: 1`, `spade_hidden: 64` |
| `scripts/train_exp1b.sh` | SLURM submit — clone of `train_exp1a.sh` with new job name + config path |

`train.py`, `inference_validate.py`, and `smoke_test.py` were updated to
use the model dispatcher `build_model_from_cfg(cfg)` so they work for both
1a and 1b without duplication.

---

## 4. Why we hand-built the U-Net instead of subclassing MONAI's

MONAI Generative's `DiffusionModelUNet` (used in 1a) buries its
GroupNorm modules inside ResBlocks whose `forward()` methods don't take
a label argument. Injecting SPADE would require either monkey-patching
ResBlock forwards (fragile, hard to read) or fork-and-modify (high
maintenance). We opted to write a clean custom U-Net that:

- Mirrors 1a's architecture closely (same levels, channels, attention at
  the deepest level only)
- Exposes SPADE injection as a constructor flag per ResBlock
- Keeps the encoder using standard GroupNorm; bottleneck + decoder use SPADE
- Sinusoidal time embedding, GroupNorm(32), SiLU activations — all the
  diffusion U-Net defaults

The custom U-Net has 23.9 M params; MONAI's UNet for 1a has 25.3 M. Close
enough that capacity is not a confounder for the ablation.

---

## 5. Bugs found and resolutions (in the order we hit them)

### 5.1 `RuntimeError: view size is not compatible with input tensor's size and stride` (smoke test)

**Location**: `unet_spade.py:155` — `SelfAttention2D.forward`, the post-einsum
reshape from `(B, heads, head_dim, N)` back to `(B, C, H, W)`.

**Cause**: `torch.einsum` can return a non-contiguous tensor depending on
the input strides. `.view()` requires contiguous memory; `.reshape()`
handles both.

**Fix**: replaced `out.view(b, c, h, w)` with `out.reshape(b, c, h, w)`.

### 5.2 `RuntimeError: tensor a (384) must match the size of tensor b (256) at non-singleton dimension 1` (smoke test, after 5.1)

**Location**: First SPADE ResBlock at decoder level 1 (the level where
the channel width steps down from 256 → 128).

**Cause**: I sized the first ResBlock's `norm1` as
`SPADE(target_ch + skip_ch, ...)` — assuming the incoming features had
the current level's channel count. But the features entering decoder
level `k` carry the *previous* level's channel count (after upsampling),
which only equals `target_ch[k]` when consecutive widths match. At the
boundary levels (`128→256` step-down), the math broke:

| Level | Previous out | Skip ch | Actual in_ch | Bug computed | OK? |
|---|---|---|---|---|---|
| lvl 3 | 256 (bottleneck) | 256 | 512 | 512 | ✓ |
| lvl 2 | 256 | 256 | 512 | 512 | ✓ |
| lvl 1 | 256 | 128 | **384** | 256 | ✗ |
| lvl 0 | 128 | 64 | **192** | 128 | ✗ |

**Fix**: track `prev_ch` explicitly across the decoder construction loop
so the first ResBlock at each level sizes its `norm1` correctly. See
`unet_spade.py:260-281`.

### 5.3 SPADE produces visibly slow convergence (first full training run, 40k steps)

**Symptom**: at step 40,000 of the first 1b run (with bugs 5.1 and 5.2
already fixed), sample grids showed:
- Fuzzy body silhouettes
- Within-body content was mostly uniform grey noise
- Step 40k looked barely different from step 5k

For comparison, 1a at the same step had clear body silhouettes, organ-
shaped masses, and visible texture differentiation.

**Cause**: SPADE's γ and β heads (the conv layers that produce the
spatial modulation) had default PyTorch Kaiming initialisation. At step
0 this means:

- γ has random non-zero values → feature scaling `* (1 + γ)` is chaotic
  from the first forward pass
- β has random non-zero values → additive offsets are random noise

This is fine for SPADE in a GAN context (the original Park et al.
setting) where adversarial training stabilises quickly, but in a
diffusion training loop the gradient signal through SPADE's small MLP
takes many thousands of steps to find a stable point.

**Fix**: zero-init both γ and β heads. SPADE then starts as identity-like
modulation (`out = GroupNorm(x) · 1 + 0 = GroupNorm(x)`) and the model
gradually learns to use the modulation as training progresses. This is
standard practice for modulation networks in diffusion (DiT's AdaLN-Zero,
Imagen, the Semantic Diffusion Model). Applied in `spade.py` after the
γ and β head definitions.

We also applied the related **output-conv zero-init** in `unet_spade.py`:
the final 1×1 conv that predicts noise starts at zero, so the model
literally begins by predicting "no noise" — the trivial baseline.
Non-zero predictions then have to come from learned signal, which routes
gradients more efficiently through SPADE.

### 5.4 (Considered but skipped) Time projection zero-init

We considered also zero-initialising the per-ResBlock time projection
(`self.time_proj = nn.Linear(time_emb_dim, out_ch)` in
`unet_spade.py:99`). This is the third "AdaLN-Zero-style" fix and is
standard in DiT. We deferred it: with γ, β, and `out_conv` zero-init,
the second 1b run converged at a normal rate, so the additional time-
proj zero-init wasn't necessary. Left as a possible future tweak.

---

## 6. Results: broken vs fixed comparison

### Run A — Broken (no zero-init fixes), abandoned at step 40k

`runs/exp1b_bad_init/samples/step_040000.png`

- Body silhouettes fuzzy and indistinct
- Within-body content mostly uniform grey
- Minimal evolution from step 5k to step 40k
- Same fixed_labels (the visualisation choice) as Run B → direct comparison
  of training dynamics, not of conditioning

Run preserved on the HPC for the dissertation methodology section ("we
observed slow convergence in the first 1b run; identifying SPADE γ/β
initialisation as the cause and zero-initialising them gave the results
in [Run B]").

### Run B — Fixed (γ + β + out_conv zero-init), completed 80k

`runs/exp1b/samples/step_080000_final.png`

- **Body silhouette crisp and well-bounded**
- **Outside-body region uniformly dark**
- Within-body content shows organ-shaped masses, contrast variation
- Recognisable anatomy by step 80k
- Sample-to-sample texture variation higher than 1a, lower than Run A —
  likely because SPADE is a stronger conditioning mechanism (different
  labels produce more differentiated outputs)

Run B is the canonical 1b result for the 1a-vs-1b ablation.

---

## 7. Final hyperparameter summary (Run B / canonical 1b)

```yaml
data:
  num_label_channels: 6           # outside_body, uterus, L-ov, R-ov, em, body_other
  ovary_oversample_weight: 3.0
  image_size: 512

model:
  type: "spade"                   # routes through build_unet_spade
  in_channels: 1                  # PURE SPADE — no concat at input
  out_channels: 1                 # noise prediction on image
  num_channels: [64, 128, 256, 256]
  attention_levels: [false, false, false, true]  # 64² only (memory)
  num_res_blocks: 2
  spade_hidden: 64                # MLP hidden width inside SPADE

diffusion:
  num_train_timesteps: 1000
  beta_schedule: "linear_beta"
  beta_start: 0.0001
  beta_end: 0.02
  prediction_type: "epsilon"

training:
  batch_size: 4
  total_steps: 80000
  lr: 1.0e-4
  weight_decay: 0.0
  grad_clip: 1.0
  amp: true                       # bf16 autocast
  cfg_dropout_prob: 0.1
  ema_decay: 0.9999

sampling:
  num_inference_steps: 50         # DDIM
  num_samples_per_grid: 4
  guidance_scale: 3.0
```

---

## 8. SPADE module internals (for the methodology section)

```python
class SPADE(nn.Module):
    def __init__(self, norm_channels, label_channels, hidden=64, num_groups=32):
        super().__init__()
        self.param_free_norm = nn.GroupNorm(num_groups, norm_channels, affine=False)
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_channels, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.mlp_gamma = nn.Conv2d(hidden, norm_channels, kernel_size=3, padding=1)
        self.mlp_beta  = nn.Conv2d(hidden, norm_channels, kernel_size=3, padding=1)
        # Identity-start init (γ=β=0 at step 0):
        nn.init.zeros_(self.mlp_gamma.weight); nn.init.zeros_(self.mlp_gamma.bias)
        nn.init.zeros_(self.mlp_beta.weight);  nn.init.zeros_(self.mlp_beta.bias)

    def forward(self, x, label):
        normalised = self.param_free_norm(x)
        label_r = F.interpolate(label, size=x.shape[-2:], mode="nearest")
        actv = self.mlp_shared(label_r)
        gamma = self.mlp_gamma(actv); beta = self.mlp_beta(actv)
        return normalised * (1.0 + gamma) + beta
```

Key choices:
- `affine=False` on GroupNorm — SPADE provides the spatially-varying γ
  and β instead of the usual learned per-channel scalars
- `nearest`-neighbour label downsampling — preserves the categorical
  structure of one-hot labels (bilinear would blur class boundaries)
- 3×3 convs inside the SPADE MLP — small spatial receptive field is
  enough for local modulation
- Identity-start zero-init — see Section 5.3

---

## 9. Open issue at end of Run B and the v2 path

### Observation

In Run B's samples, the body silhouette is well-defined and centred
content is realistic, but **the body itself occupies only ~70-80% of
the 512×512 frame**. Air margins around the body are visible. Same
issue exists in 1a's CFG+EMA+6chan result — the cause is preprocessing,
not the conditioning mechanism.

### Cause

`preprocess_for_generator.py` resampled to 512×512 at 0.35 mm in-plane,
centred on the **image** centre. For most pelvic MRI:
- Native FOV: ~300 mm
- 512 × 0.35 = 179.2 mm FOV after resample
- Body width at pelvis: ~150-200 mm
- Image centre ≠ body centre (rater frames vary), so the body ends up
  offset to one side with asymmetric air margins

### Fix (v2 — body-centered resampling)

Replaced the image-centered `_resample_image_to_target()` with a body-
centered `_resample_image_body_centered()`:

1. Compute body silhouette at native resolution
2. Find body bounding box (x0, x1, y0, y1)
3. Pad to a square (longest side) + 5% margin
4. Compute per-subject output spacing so body+margin = 512 px exactly
5. Resample with output centred on body bbox centroid

The label resampling code (`_resample_label_to_image`) already used the
resampled image as its reference, so it automatically picks up the new
body-centered frame.

Effect on the model:
- Body fills ~90% of the 512×512 output frame
- Per-subject in-plane spacing varies (~0.3-0.6 mm/px depending on
  body size); recorded in the per-subject summary JSON
- Outside_body channel becomes a thin frame margin instead of dominating
- Organs are ~30-50% larger in pixel terms — model has more pixels to
  spend on the anatomy of interest

This change is invisible to the model architecture, dataset code, and
configs. It's a purely preprocessing-side improvement that requires
re-preprocessing all subjects and re-training both 1a and 1b for the
ablation to remain on equal footing.

### Run names after v2

| HPC dir | What it is |
|---|---|
| `runs/exp1a_v1_imgcentered/` | 1a CFG+EMA+6chan, image-centered FOV (kept for reference) |
| `runs/exp1b_v1_imgcentered/` | 1b SPADE Run B, image-centered FOV (kept for reference) |
| `runs/exp1a/` | 1a CFG+EMA+6chan, **body-centered** (v2 — re-run after the preprocessing change) |
| `runs/exp1b/` | 1b SPADE, **body-centered** (v2 — re-run after the preprocessing change) |

Run A (the broken init run) was archived as `runs/exp1b_bad_init`.

---

## 10. Known limitations of 1b (not 1b's job to fix)

- **No quantitative metrics yet** — FID, boundary DSC, NN-LPIPS are
  Phase 1 endpoints, planned for after 1c
- **No augmentation** — same scope as 1a; small data ceiling
- **The L/R ovary distinction is mostly synthetic** in the training
  data (92.5 % midline_fallback in preprocessing); same effect on 1b as
  on 1a, so the ablation stays fair
- **Encoder sees the noisy image with no label info** — this is inherent
  to pure SPADE. The encoder has to learn good features blind, then the
  label modulates the decoder. SPADE's hypothesis is that label-aware
  *decoding* outweighs label-blind *encoding*; this ablation tests that

---

## 11. Things 1c will inherit from 1b

When we add the PatchGAN discriminator (Exp 1c):
- Same SPADE U-Net as the generator
- Same 6-channel labels (or v2 body-centered, whichever is current)
- Same data, split, hyperparameters except for the new adversarial loss
  term
- Discriminator likely trained on (real_image, label) vs (synth_image,
  label) pairs to keep adversarial signal label-aware

Memory budget for 1c: adds the discriminator's parameters and gradients
(~5-10 M params plus its activations). Should still fit at batch 4; if
not, drop to batch 2.

---

## 12. (Archived) Original planning notes

The first version of this file was a forward-looking checklist drafted
before any 1b code was written. The decisions we made and where to look
for them now:

| Original planning question | Decision taken |
|---|---|
| Pure SPADE or hybrid? | Pure SPADE (Section 1, Section 7's `in_channels: 1`) |
| SPADE at which levels? | Decoder + bottleneck, encoder uses standard GroupNorm (Section 1's norm table) |
| Backbone parity with 1a? | Same depth, widths, attention positions; 23.9 M vs 25.3 M params (Section 1) |
| How does SPADE interact with CFG? | Standard CFG dropout zeros entire label; SPADE collapses to its learned identity-baseline; no change to CFG code (Section 7) |
| Subclass MONAI or fork? | Hand-built custom U-Net (Section 4) |
| Memory forecast at batch 4 | Confirmed fit during smoke test; no OOM in either run (Section 5 / 6) |
