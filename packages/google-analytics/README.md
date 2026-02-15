# Google Analytics Process for Loom

A [Loom](https://github.com/sfw/loom) process package for auditing and optimizing Google Analytics 4 implementations.

## What it does

This process guides Loom through a structured GA4 audit:

1. **Tracking Audit** - Event taxonomy, custom dimensions, data streams, enhanced measurement
2. **Data Quality** - Sampling, consent mode impact, cross-domain tracking, session stitching
3. **Funnel Analysis** - Conversion funnels with drop-off rates, segmented by channel and device
4. **Audience Segmentation** - Behavioral segments with GA4 audience definitions
5. **Attribution Analysis** - Model comparison (last-click, data-driven, position-based)
6. **Recommendations** - Prioritized fixes with implementation steps and impact estimates

## Installation

```bash
loom install github.com/sfw/loom-process-google-analytics
```

Or install from a local directory:

```bash
loom install /path/to/loom-process-google-analytics
```

## Usage

```bash
loom cowork --process google-analytics
```

Then describe your analytics goal:

> "Audit our GA4 implementation for our B2B SaaS free trial funnel. We think we're losing visibility on the trial-to-paid conversion step."

Loom will decompose this into phased subtasks following the process definition, using the calculator for conversion rate math and spreadsheets for data tables.

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

Place any of these in your workspace before running:

- **GA4 exports** (`.csv`) - Event data, audience exports, conversion reports
- **GTM container** (`.json`) - Exported GTM container for tag/trigger inventory
- **Existing documentation** (`.md`) - Analytics briefs, measurement plans, prior audits

## Dependencies

- `pyyaml` (installed automatically by `loom install`)

## License

MIT
