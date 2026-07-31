# Dataiku native-labeling workflow

## Create the labeling task

1. Run **Cluster defect instances** and verify `LATEST.json`, the output dataset row count, and the recipe log.
2. Create an object-detection Labeling task from the output review dataset.
3. Associate the original input image managed folder.
4. Select `image_path` as the image path column.
5. Import `prelabels_json` as existing object-detection labels.
6. Add `primary_cluster_id`, `instances_json`, `num_defects`, `review_order`, `image_status`, and `run_id` as context columns.
7. Define the real defect taxonomy and retain `UNVALIDATED_DEFECT` only as a machine-proposal placeholder.
8. Require annotators to replace the placeholder or delete the box.
9. Configure reviewers and consume only reviewer-validated output as ground truth.

## Reviewer rules

- Resize loose boxes.
- Delete false proposals.
- Add every missed defect, including secondary defects.
- Assign classes independently to each box.
- Do not infer a class from `cluster_id`; it is only a run-local grouping.
- Escalate ambiguous, occluded, or taxonomy-conflicting examples.

## Multi-defect images

The output contains every detector box for an image. Each instance has its own ID, score, cluster assignment, centroid similarity, rank, and warnings in `instances_json`. The image receives one `review_order` based on the first representative instance selected by the cross-cluster interleave.

## Error and no-detection rows

- `NO_DETECTION` rows have `[]` for both JSON columns and should be sampled in recall audits.
- `ERROR` rows are review-inert and must be remediated before any dataset-completeness claim.
