# NSFW filter · Azure #164 · praxis v2

Reset after the review comment on `docs/nsfw_praxis.md:1` (9 Jul). The v1
protocol used classical art nudes as the borderline set. Wrong framing: art
is not the target signal. This v2 rewrites the scope, the model shortlist,
and the dataset plan before rerunning any bench.

Date: 14 Jul 2026.

## What we filter and why

The service runs on every AI input and every AI output. Cost per request is
the ceiling, not accuracy in isolation. It gates public content on the
website, so a false negative is worse than a false positive on borderline.

Target signals:

- explicit sexual content (photorealistic or generated)
- borderline nudity on real bodies (nude beach, topless photography), not
  paintings
- graphic violence, gore, weapons in threatening context
- text prompts asking for any of the above (input side)

Pass-through, not target:

- artistic nudity (paintings, sculpture) unless graphic
- medical, educational, ethnographic material
- swimwear, lingerie catalog, dance, contact sport

## Model shortlist

Four candidates, two lanes (image and text). Numbers below are from HF
model cards fetched 14 Jul, or from the v1 bench in the appendix.

| Model | Params | Modality | Categories | License | Bench notes | Role |
|---|---|---|---|---|---|---|
| `bhky/opennsfw2` | Yahoo openNSFW2 port, few M | image | binary NSFW prob | MIT (bhky wrapper), upstream Yahoo terms to double-check before ship | 265 ms warm CPU M1 | fast image triage |
| `Falconsai/nsfw_image_detection` | 85.8 M ViT | image | binary normal / nsfw, 98.04% eval acc | Apache 2.0 | 200 ms warm CPU M1 | accurate image |
| `google/shieldgemma-2-4b-it` | 4 B | image + text | sexual explicit, dangerous, violence / gore, per-policy score | Gemma ToU | no CPU numbers, TPU-trained | policy-based, escalation only |
| `google/shieldgemma-2b` | 2 B | text only | sexual explicit, dangerous, hate, harassment | Gemma ToU | no numbers | text prompt filter |

Bhky and Falcon are the two image-side workhorses. ShieldGemma 4B is heavy
(4 B params, no CPU latency disclosed) so it does not belong on the hot
path. Useful when a caller asks for per-category output (nude vs violence
vs weapon) that the two binary models cannot give. ShieldGemma 2B covers
the text side, which the ticket did not scope originally but which the
review comment on v1 pulled in as part of the same filter surface.

## Cascade proposal

Two lanes.

```
image path:
  input --> bhky (prob)
              prob < 0.2 --> pass
              prob > 0.8 --> flag nsfw
              0.2..0.8   --> Falconsai
                              normal --> pass
                              nsfw   --> flag
              caller asked for per-category --> ShieldGemma-2-4B-IT
                                                with per-policy prompt

text path:
  input --> ShieldGemma-2B, run 4 policies
              any Yes --> block prompt
              all No  --> pass
```

Bhky is cheap enough to run on every request. Falcon runs only on the
uncertain band, keeps p99 controlled. ShieldGemma 4B runs only on explicit
caller ask, never on the default path. ShieldGemma 2B is a separate text
call, single roundtrip per prompt.

Thresholds `0.2` and `0.8` are placeholders. Actual numbers come from the
v2 bench in Tanda B, based on the ROC on the new dataset.

## Test protocol v2

Categories, 10 images each unless flagged. Text prompts are 20 strings.

- `safe_real`: everyday photos, no ambiguity
- `porn_explicit`: real explicit content (5 only, minimum to sanity check)
- `borderline_nude_flag`: nude beach, topless real bodies, non-artistic
- `borderline_safe_pass`: bikini, swimwear, lingerie catalog, contact sport
- `violence_gore`: real injury, gore (5 only)
- `weapons`: firearms and blades in threatening framing
- `text_prompts`: 20 strings, half harmful (per the 4 ShieldGemma categories)
  and half benign

Image side scored on bhky + Falconsai + ShieldGemma-2-4B-IT.
Text side scored on ShieldGemma-2B.

For each image model, report: score, latency warm, latency cold, correct
call at threshold 0.5. For ShieldGemma 4B, report per-policy score.

## Sourcing decision (open, blocks Tanda B)

Sensitive material, needs an owner call before it hits the laptop or a
runner. Ranked options:

1. `NudeNet` labeled dataset. Public, categorized (SAFE, EXPOSED_*). Best
   for reproducibility. Skews toward adult-site stills.
2. `LAION-NSFW` subset. Public, large, contains explicit and borderline.
   Needs filter script and disk.
3. Stock APIs with adult flag (Unsplash / Pexels for safe half, paid
   permitted-adult stock for the explicit half).
4. Synthetic via an SD-NSFW checkpoint. No real content, so filter stays
   blind to real-world distribution.

For `violence_gore`, `RealLifeViolenceDataset` is the standard research
option. For `weapons`, weapon-detection research datasets exist on Kaggle.
`text_prompts` we author in-repo.

Not building the dataset until the sourcing is confirmed. See PR reply.

## Blockers before rerun

- Owner call on sourcing (question in the PR#20 reply).
- Gemma ToU accepted on HF for the account that pulls ShieldGemma.
- Confirm whether ShieldGemma 4B runs on M1 with int8 quant or needs a
  cloud runner (Modal / RunPod).

## Frictions for the APIPod integration prompt

1. `tmplink/nsfw_detector` still 404 on HF. Substitute Falconsai
   (Apache 2.0, verified 14 Jul).
2. TF / torch wheels for Python 3.14 do not exist yet. Worker image pins
   3.11 or 3.12.
3. `opennsfw2` first predict downloads weights to `~/.opennsfw2/`. Worker
   needs a persistent cache mount or the download runs on every cold start.
4. ShieldGemma pulls from HF gated repos. Worker HF token needs Gemma ToU
   accepted per account.
5. Model registration syntax changed on APIPod `origin/new_registry` branch
   (refactor 16 Jun). Port target is that branch, not `dev`.

## Open questions

1. Response schema. Ship one canonical `nsfw_score` binary, or expose the
   3 ShieldGemma categories granularly to the caller when they ask?
2. Text lane scope. Is ShieldGemma 2B in this ticket, or split into a
   sibling ticket? Different model family, different endpoint shape.

## Appendix · v1 baseline (9 Jul)

Kept as sanity check that the two image models return low scores on
synthetic safe content and that both can fire on art nudes. Wrong dataset
for the actual filter decision, right dataset to prove the plumbing works.

### bhky on 5 safe PIL images

| Image | NSFW prob | Latency (ms) | Note |
|---|---|---|---|
| 1_plain_green.jpg | 0.0003 | 2114.7 | cold |
| 2_circle_bw.jpg | 0.0119 | 242.2 | warm |
| 3_text_gradient.jpg | 0.0113 | 276.9 | warm |
| 4_checker.jpg | 0.0033 | 274.9 | warm |
| 5_landscape.jpg | 0.0000 | 267.2 | warm |

Cold start 2.1 s, warm mean 265 ms.

### Falconsai on 5 safe PIL images

| Image | Label | Score | Latency (ms) | Note |
|---|---|---|---|---|
| 1_plain_green.jpg | normal | 0.9992 | 2106.6 | cold |
| 2_circle_bw.jpg | normal | 0.9986 | 215.7 | warm |
| 3_text_gradient.jpg | normal | 0.9986 | 179.5 | warm |
| 4_checker.jpg | normal | 0.9967 | 204.0 | warm |
| 5_landscape.jpg | normal | 0.9995 | 191.8 | warm |

Cold start 2.1 s, warm mean 198 ms.

### bhky on 5 art nudes (wrong dataset, kept for the record)

| Image | NSFW prob | Verdict at 0.5 |
|---|---|---|
| 1_botticelli_venus.jpg | 0.0626 | safe |
| 2_michelangelo_david.jpg | 0.8723 | flag |
| 3_titian_venus_urbino.jpg | 0.9829 | flag |
| 4_rubens_three_graces.jpg | 0.9371 | flag |
| 5_doryphoros_statue.jpg | 0.4060 | safe |

### Falconsai on 5 art nudes (wrong dataset, kept for the record)

| Image | Label | Score |
|---|---|---|
| 1_botticelli_venus.jpg | normal | 0.9984 |
| 2_michelangelo_david.jpg | normal | 0.9997 |
| 3_titian_venus_urbino.jpg | nsfw | 0.6666 |
| 4_rubens_three_graces.jpg | nsfw | 0.9997 |
| 5_doryphoros_statue.jpg | normal | 0.9998 |
