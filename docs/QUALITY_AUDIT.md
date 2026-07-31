# Annotation-quality audit

## Correctness unit

The primary unit is an image. An audited image is correct only when:

1. every true defect is represented;
2. every final box is spatially acceptable under the written protocol;
3. every box has the correct expert taxonomy label; and
4. no spurious box remains.

An image with one wrong or missing instance is incorrect. Do not count individual boxes as independent trials for the headline claim.

## Independence and holdout

Use an expert or adjudicated expert pair who did not create the reviewed annotation. Draw a probability sample from the finalized reviewer-validated dataset. Do not use the audit sample to tune detector threshold, representation settings, K, taxonomy guidance, or reviewer training.

## Acceptance rule

For a simple random sample of independent images, compute the one-sided 95% exact Clopper–Pearson lower confidence bound. Accept “greater than 95%” only when the lower bound is strictly greater than `0.95`.

```bash
python scripts/binomial_audit.py --audited 400 --incorrect 12
```

Small technically passing samples are not adequate for a heterogeneous industrial corpus. Use approximately 400–1,000 globally sampled images for the headline claim, plus oversampled diagnostics for rare classes and critical domains.

## Complex sample designs

Do not pool disproportionate stratum samples without weights. When images are correlated by batch, line, lot, or acquisition session, use a cluster-aware bootstrap or design-based interval. The simple exact-binomial script is only valid for a simple random sample of independent images.

## Required report

Record the sampling frame and seed, inclusion probabilities, audit protocol, auditors, adjudication method, audited/incorrect counts, confidence method, global lower bound, stratum diagnostics, error taxonomy, unresolved cases, and the exact dataset/run versions audited.
