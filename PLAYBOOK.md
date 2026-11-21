# USA Shorts — daily production playbook

Each run makes the next 2 Shorts from the content calendar (Claude Sheet "USA Shorts — 30-Day Content Calendar").

## Setup in a fresh session
1. `pip install --break-system-packages kokoro-onnx soundfile shapely pyshp scipy pillow numpy`
2. Kokoro voice model (GitHub release):
   `curl -L -o /tmp/tts/kokoro.onnx https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx`
   `curl -L -o /tmp/tts/voices.bin https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin`
3. Country shapes: `pip download geopandas==0.14.4 --no-deps -d /tmp/gp && cd /tmp/gp && unzip -o geopandas*.whl 'geopandas/datasets/*'`
4. Voice: `am_michael` (US English). Format 1080x1920, 30 fps, 35–45 s.

## Rules for every video (monetization safety)
- Research every fact with web search first; only claims you can source. Avoid politics and real people's photos.
- Each video must differ: new map region, different scene order, at least one visual device not used in the previous 3 videos
  (spotlight, region fill, dashed route, counter, chart, icon grid, before/after split, timeline, scale comparison).
- Hook in the first 1.5 s: surprising claim + big text + impact SFX. Payoff before 40 s. End with a loop-friendly line.
- Title < 60 chars, "Why…?" style. Description: 2 lines + 3 hashtags (#shorts #geography #usa) + sources.

## Steps
1. Read the calendar, take the first 2 rows with Status = Idea.
2. Research, write scene list + VO lines (spoken numbers spelled out).
3. Write `videos/<slug>.py` using engine.py (see example_*.py), generate VO (vo.py), render, check stills at the busiest moments.
4. Send the 2 MP4s + title/description/tags to the user; set calendar Status = Made.
