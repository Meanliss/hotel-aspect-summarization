# Method 1: UASum

UASum is a controlled ABSA-to-summary pipeline for hotel reviews. The design goal
is not only to produce readable summaries, but also to keep a traceable path from
each generated claim back to evidence units in the original reviews.

## Motivation

Hotel reviews are noisy and often contain multiple opinions in one sentence. A
single review can praise the room, complain about Wi-Fi, and mention staff
service at the same time. Direct free-form summarization hides where mistakes
come from. UASum separates the task into auditable stages.

## Stages

| Stage | Output | Purpose |
|---|---|---|
| Raw review ingestion | review records | Preserve hotel and review identifiers. |
| Opinion unit segmentation | opinion units | Split mixed reviews into smaller units. |
| ABSA extraction | aspect, sentiment, evidence | Structure opinions for downstream grouping. |
| Guardrail and normalization | valid taxonomy labels | Prevent labels outside the schema. |
| Cluster assignment | cluster code and descriptor | Group evidence by business meaning. |
| Evidence rollup | evidence buckets | Aggregate by hotel, aspect, sentiment, cluster. |
| Controlled generation | aspect summaries | Generate from evidence buckets, not raw text. |
| Audit and metrics | trace files and scores | Check consistency and failure modes. |

## Taxonomy

The main hotel taxonomy contains six aspects:

- `facility`
- `amenity`
- `service`
- `experience`
- `branding`
- `loyalty`

Sentiment labels:

- `positive`
- `negative`
- `neutral`

The final full run uses 51 evidence clusters across these aspects.

## Important implementation detail

For final audit, use `sentence_absa_processing_trace.csv`, not the older
`segmentation_trace.csv`. The sentence-level ABSA trace is the canonical source
because it contains the final aspect, sentiment, cluster, evidence, and source
pointers used by the summary layer.
