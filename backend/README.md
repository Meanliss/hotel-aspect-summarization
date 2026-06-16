# Analyze Backend (to be implemented)

The web app's **Explore** tab is fully static and needs no backend. The
**Analyze** tab (paste reviews → run the model → see the aspect tree) needs a
small Python service that wraps the existing pipeline and returns the JSON shape
defined in [`../web/API_CONTRACT.md`](../web/API_CONTRACT.md).

This directory is a placeholder: the contract and the frontend are ready, the
service itself is intentionally **not** implemented yet (it requires a GPU/CPU
host that can load PyTorch + the SemAE checkpoint + transformers — which Vercel
cannot do).

## Recommended stack

- **FastAPI** + **uvicorn** for the HTTP layer.
- Reuse the existing modules directly (no rewrite):
  - `src/aspect_inference.py` — SemAE evidence ranking + aspect matching.
  - `src/sentiment_classifier.py` — BERT/DeBERTa aspect-based sentiment.
  - `scripts/synthesize_aspect_summaries.py` — abstractive pos/neg summaries.
  - `scripts/export_web_data.py` (`build_export`) — final JSON shaping.

## Suggested steps

1. `POST /analyze` receives `{ reviews, entity_name, options }`.
2. Wrap `reviews` into a one-entity SPACE-format dict (entity_id `"adhoc"`).
3. Load the SemAE checkpoint once at startup (module-level singleton), plus the
   shared sentiment classifier via `sentiment_classifier.get_classifier(...)`.
4. Run inference → threshold evidence → synthesis (`--split_sentiment`,
   `--hierarchical`) in-process, writing to a temp run dir.
5. Call the single-entity export shaping and return it as JSON.

## Deployment

Host where a model can run (GPU droplet, a VM, HF Spaces, Modal, RunPod, etc.).
Then point the frontend at it:

```
# web/.env (or Vercel project env)
NEXT_PUBLIC_API_URL=https://your-backend.example.com
```

Enable CORS for the Vercel domain in the FastAPI app.
