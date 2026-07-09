# NSFW praxis · Azure #164 local test

Date: 9 Jul 2026. Python 3.11 venv.

## Why this service

Socaity is a MaaS provider, users deploy models and generate media
through us. A NSFW filter flags nudity, gore, or other adult content
before outputs land in someone else's app. Ticket Azure #164 wraps two
candidate models into a callable APIPod service. This doc is the
praxis before the service PR: install them, run them, prove they
score sanely on safe content.

## Ticket ask

`Azure #164 NSFW service kickoff` says: praxis test `bhky/opennsfw2` +
`tmplink/nsfw_detector` before APIPod integration.

## Two model paths

1. **`bhky/opennsfw2`** · Yahoo's OpenNSFW2 port. Available as a PyPI
   package `opennsfw2` (v0.14.0). Under the hood: TensorFlow / Keras
   SavedModel. Loads once, caches to disk. Returns a single float in
   [0, 1] = NSFW probability.
2. **`tmplink/nsfw_detector`** · 404 on HuggingFace when queried
   9 Jul 2026 (`hf.co/tmplink/nsfw_detector` returns
   `RepositoryNotFoundError`). The name may be a typo or the repo was
   privated/removed. Substitute with a live, actively-maintained
   equivalent: **`Falconsai/nsfw_image_detection`** (ViT base, image
   classification pipeline, ~85M params, MIT).

## Test images

5 synthetic safe images generated with PIL, 224x224 JPEGs, no
real-world content that could be false-positive-y. Files are in
`docs/example_images/nsfw_praxis/safe/`:

1. `1_plain_green.jpg`, flat brand-color rectangle.
2. `2_circle_bw.jpg`, black circle on white.
3. `3_text_gradient.jpg`, "SOCAITY" on a light gradient.
4. `4_checker.jpg`, 7x7 checkerboard.
5. `5_landscape.jpg`, blue-to-green landscape gradient.

All five must score as safe (low NSFW probability) or the model is
broken for baseline content.

## Setup (Python 3.11 venv)

Python 3.14 does not have TF/torch wheels yet (9 Jul 2026), so the
venv must be 3.11 or 3.12. Reproduced on 3.11.14.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install opennsfw2 tf_keras transformers torch pillow
```

## bhky/opennsfw2 results

Run at 13:08 CET on M-series MBP. The model returns a single float in
[0, 1] = probability the image is NSFW. Safe content should score
near 0. Standard threshold is 0.5 (above = flag as NSFW).

| Image | NSFW prob | Latency (ms) | Note |
|---|---|---|---|
| 1_plain_green.jpg | 0.0003 | 2114.7 | cold start |
| 2_circle_bw.jpg | 0.0119 | 242.2 | warm |
| 3_text_gradient.jpg | 0.0113 | 276.9 | warm |
| 4_checker.jpg | 0.0033 | 274.9 | warm |
| 5_landscape.jpg | 0.0000 | 267.2 | warm |

All 5 safe test images score well below any reasonable NSFW threshold
(highest 0.0119, threshold typically 0.5). Cold start 2.1s, warm mean
265ms.

## Falconsai/nsfw_image_detection results (substitute for tmplink)

Run at 13:10 CET on M-series MBP. The model returns a label ({normal,
nsfw}) with a softmax score. Safe content should get `normal` with
high confidence.

| Image | Top label | Score | Latency (ms) | Note |
|---|---|---|---|---|
| 1_plain_green.jpg | normal | 0.9992 | 2106.6 | cold start |
| 2_circle_bw.jpg | normal | 0.9986 | 215.7 | warm |
| 3_text_gradient.jpg | normal | 0.9986 | 179.5 | warm |
| 4_checker.jpg | normal | 0.9967 | 204.0 | warm |
| 5_landscape.jpg | normal | 0.9995 | 191.8 | warm |

Model returns two labels: `normal` (safe) and `nsfw`. All 5 test
images labelled `normal` with 99.67% or higher confidence. Cold start
2.1s, warm mean 198ms.

## Borderline test · classical art nudes

The safe-image test above only proves the models don't false-alarm.
It doesn't prove they catch actual nudity. Running the exact same
protocol on 5 classical art nudes from Wikimedia Commons (public
domain) so we know the models fire when they should. Not explicit
content, but recognisably nude subjects.

The 5 images: Botticelli's Birth of Venus (painting), Michelangelo's
David (sculpture), Titian's Venus of Urbino (painting), Rubens's
Three Graces (painting), Doryphoros (classical Greek sculpture,
Roman copy). Files in `docs/example_images/nsfw_praxis/borderline/`.

### bhky/opennsfw2 · borderline

| Image | NSFW prob | Verdict (≥0.5) |
|---|---|---|
| 1_botticelli_venus.jpg | 0.0626 | safe |
| 2_michelangelo_david.jpg | 0.8723 | **flag** |
| 3_titian_venus_urbino.jpg | 0.9829 | **flag** |
| 4_rubens_three_graces.jpg | 0.9371 | **flag** |
| 5_doryphoros_statue.jpg | 0.4060 | safe |

Flags 3 of 5. Misses Botticelli (soft rendering, model reads it as
mostly safe) and Doryphoros (just below threshold at 0.41). Aggressive
overall.

### Falconsai/nsfw_image_detection · borderline

| Image | Top label | Score |
|---|---|---|
| 1_botticelli_venus.jpg | normal | 0.9984 |
| 2_michelangelo_david.jpg | normal | 0.9997 |
| 3_titian_venus_urbino.jpg | nsfw | 0.6666 |
| 4_rubens_three_graces.jpg | nsfw | 0.9997 |
| 5_doryphoros_statue.jpg | normal | 0.9998 |

Flags 2 of 5. Reads sculptures (David, Doryphoros) and the softer
Botticelli as normal. Flags the more explicit paintings (Titian,
Rubens). More permissive with sculpture and art, more targeted at
explicit content.

### What this contrast says

Both models can detect nudity, they don't just always answer "safe".
The two disagree in a useful way: `bhky/opennsfw2` is aggressive
(older CNN, few false negatives, expect false positives on art
content); `Falconsai/nsfw_image_detection` is nuanced (modern ViT,
distinguishes classical art from explicit content). For a MaaS
filter service, Falconsai is the better default. bhky is useful as a
stricter secondary mode if the caller wants zero tolerance.

## Gate for APIPod integration (Jul 2)

Move to APIPod integration only if both:

1. All 5 safe test images score as safe on both models.
2. First-image latency < 3s (cold start), subsequent < 300ms
   (warm), on this M-series MBP.

**Both gates pass.** OK to schedule the APIPod integration in Sprint
Jul 2 (calendar cell Wed 22 in the sprint plan).

## Frictions to note in the APIPod integration prompt

1. `tmplink/nsfw_detector` no longer resolves. If the intent was the
   name shown in the ticket, either update the ticket or substitute
   `Falconsai/nsfw_image_detection`. The Falconsai model is more
   modern (ViT vs old CNN), maintained, and MIT.
2. TF/torch wheels for Python 3.14 do not exist yet. APIPod worker
   image should pin Python 3.11 or 3.12.
3. `opennsfw2` bundles a TF SavedModel; first predict downloads
   weights to `~/.opennsfw2/`. APIPod worker needs write access to a
   persistent cache dir, or the download runs every cold start.

## Open questions

1. Should Socaity offer both models as separate services (bhky's
   OpenNSFW2 legacy compat + Falconsai's ViT modern) or one canonical?
2. Response schema: single float (bhky) vs multi-label with softmax
   (Falconsai). If we ship one canonical `nsfw_score` on APIPod, we
   pick a schema and adapt the underlying model to it.
