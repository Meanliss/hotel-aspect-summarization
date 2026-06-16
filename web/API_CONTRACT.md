# Analyze API Contract

The **Explore** tab reads static JSON from `public/data/` and needs no backend.
The **Analyze** tab (paste reviews → run model → see the same aspect tree) calls a
Python backend that you stand up separately. This file is the contract that
backend must satisfy. The TypeScript types live in [`lib/types.ts`](lib/types.ts).

## Endpoint

```
POST {NEXT_PUBLIC_API_URL}/analyze
Content-Type: application/json
```

### Request body (`AnalyzeRequest`)

```json
{
  "reviews": ["The room was spotless.", "Breakfast was cold."],
  "entity_name": "My Hotel",
  "options": {
    "sentiment_backend": "bert",
    "split_sentiment": true,
    "max_tokens": 120
  }
}
```

- `reviews`: array of review sentences/strings (required, non-empty).
- `entity_name`: display name (optional).
- `options`: optional knobs that mirror the pipeline CLI flags.

### Response body (`AnalyzeResponse` = one `Entity`)

The response is exactly one `Entity` object — the **same shape** the Explore tab
renders, so the frontend reuses `<AspectTree>` directly:

```json
{
  "entity_id": "adhoc",
  "entity_name": "My Hotel",
  "split": "adhoc",
  "overall_summary": "….",
  "parents": [
    {
      "code": "FACILITY",
      "summary": "….",
      "children": [
        {
          "code": "FAC_ROOM",
          "scale": "Room, Bed & Sleep Quality",
          "description": "….",
          "summaries": { "positive": "….", "negative": "….", "neutral": "" },
          "evidence": {
            "positive": [
              { "sentence": "The room was spotless.", "score": -0.001, "rank": 1, "review_id": null }
            ],
            "negative": [],
            "neutral": []
          }
        }
      ]
    }
  ]
}
```

Field semantics match `scripts/export_web_data.py`:

- `parents[]` ordered by taxonomy group (FACILITY, AMENITY, SERVICE, EXPERIENCE,
  BRANDING, LOYALTY).
- Each `child.summaries.positive` / `.negative` is the abstractive summary for
  that polarity (empty string if none). `.neutral` holds a combined/legacy
  summary when sentiment was not split.
- Each `child.evidence.{positive,negative,neutral}[]` are the source sentences,
  sorted best-first by SemAE `score`.

## Errors

Non-2xx responses should return a plain-text or JSON error body; the frontend
shows the status line and body text.

## How the backend maps to the pipeline

A reference implementation would, per request:

1. Build an in-memory SPACE-format entity from `reviews`.
2. Run `src/aspect_inference.py` logic with `--sentiment_backend bert`
   (uses `src/sentiment_classifier.py`) to rank evidence + label sentiment.
3. Run `scripts/synthesize_aspect_summaries.py` logic with `--split_sentiment`
   + `--hierarchical` to produce per-polarity child summaries, parent summaries,
   and the overall summary.
4. Shape the result with the same logic as `scripts/export_web_data.py`
   (`build_export`) but for a single entity, and return that object.

See [`../backend/README.md`](../backend/README.md) for the build steps.
