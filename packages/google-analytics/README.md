# Google Analytics Process for Loom

A [Loom](https://github.com/sfw/loom) process package for auditing and optimizing Google Analytics 4 implementations.

This package uses process contract v2 (`schema_version: 2`) with explicit
verification policy/remediation, evidence contracts, and prompt constraints.

## What it does

This process guides Loom through a structured GA4 audit:

1. **Tracking Audit** - Event taxonomy, custom dimensions, data streams, enhanced measurement
2. **Data Quality** - Sampling, consent mode impact, cross-domain tracking, session stitching
3. **Funnel Analysis** - Conversion funnels with drop-off rates, segmented by channel and device
4. **Audience Segmentation** - Behavioral segments with GA4 audience definitions
5. **Attribution Analysis** - Model comparison (last-click, data-driven, position-based)
6. **Recommendations** - Prioritized fixes with implementation steps and impact estimates

In addition to exported-file analysis, the package now includes a live API tool
(`ga_live_api`) to pull metadata and report rows directly from GA4.

## Installation

```bash
loom install https://github.com/sfw/loom-process-google-analytics
```

Or install from a local directory:

```bash
loom install /path/to/loom-process-google-analytics
```

## Usage

```bash
loom cowork --process google-analytics
```

Then force process orchestration from the chat input:

```text
/run Audit our GA4 implementation for our B2B SaaS free trial funnel. We think we're losing visibility on the trial-to-paid conversion step.
```

`/run` executes the active process in-session via `delegate_task` (no `loom serve` required).
For non-interactive execution, use:

```bash
loom run "Audit our GA4 implementation for our B2B SaaS free trial funnel" --workspace /tmp/ga-audit --process google-analytics
```

## Auth setup for live GA API retrieval

Create or edit an auth profile for provider `google_analytics`:

```toml
[auth.profiles.ga_prod]
provider = "google_analytics"
mode = "api_key"
secret_ref = "env://GA_API_KEY"

[auth.profiles.ga_prod.env]
GA_PROPERTY_ID = "123456789"
```

Then select it for runs:

```bash
loom auth select google_analytics ga_prod
```

Or set it per run:

```bash
loom run "Audit GA4 with live API pull" \
  --process google-analytics \
  --auth-profile google_analytics=ga_prod
```

The bundled `ga_live_api` tool supports:

- `run_report`
- `run_realtime_report`
- `get_metadata`

## Deliverables

The process produces:

| Phase | Files |
|-------|-------|
| Tracking Audit | `tracking-audit.md`, `event-inventory.csv` |
| Data Quality | `data-quality.md`, `quality-metrics.csv` |
| Funnel Analysis | `funnel-analysis.md`, `funnel-metrics.csv` |
| Audience Segmentation | `audience-segments.md`, `segment-comparison.csv` |
| Attribution Analysis | `attribution-analysis.md`, `attribution-comparison.csv` |
| Recommendations | `recommendations.md`, `recommendation-matrix.csv`, `implementation-roadmap.md` |

## Verification

Built-in verification checks:

- **No placeholders** (regex, error) - Catches `[TBD]`, `[TODO]`, `[INSERT]`, `XX%`
- **Metrics quantified** (LLM, warning) - Every recommendation must include a quantified impact
- **GA4-specific** (LLM, warning) - Implementation steps must reference actual GA4/GTM features
- **Funnel math** (LLM, error) - Step-by-step conversion rates must be arithmetically consistent

## Providing input data

Place any of these in your workspace before running (optional if using live API retrieval):

- **GA4 exports** (`.csv`) - Event data, audience exports, conversion reports
- **GTM container** (`.json`) - Exported GTM container for tag/trigger inventory
- **Existing documentation** (`.md`) - Analytics briefs, measurement plans, prior audits

## Dependencies

- `pyyaml` (installed automatically by `loom install`)
- `httpx>=0.27` (used by `ga_live_api` for GA Data API requests)

## License

MIT
