Place any available projector checkpoints here to switch the web app from approximate raw-cosine mode
to the exact projected-cosine feature path used by `main.py`.

Expected filenames:

- `text_projector.pt`
- `deja_projector.pt`
- `unifont_projector.pt`
- `gentium_projector.pt`
- `libre_projector.pt`
- `exo2_projector.pt`
- `doulos_projector.pt`
- `cousine_projector.pt`

If a projector file is missing, the web app still runs, but it falls back to direct cosine similarity for
that source and labels the request as `raw_cosine_fallback`.
