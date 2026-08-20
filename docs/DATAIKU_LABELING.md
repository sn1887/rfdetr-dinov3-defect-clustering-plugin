# Dataiku native-labeling workflow

## Create the labeling task

1. Run **Cluster defect instances** and verify `LATEST.json`, the output dataset row count, and the recipe log.
2. Create an object-detection Labeling task from the output review dataset.
3. Associate the original input image managed folder.
4. Select `image_path` as the image path column.
5. Import `detection_bbox` as existing object-detection labels.
6. Add `primary_cluster_id`, `detection_score`, `detection_bbox_cluster`, `instances_json`, `num_defects`, `review_order`, `image_status`, and `run_id` as context columns.
7. Define the real defect taxonomy used by reviewers.
8. Require annotators to validate, rename, or delete each proposed box.
9. Configure reviewers and consume only reviewer-validated output as ground truth.

## Reviewer rules

- Resize loose boxes.
- Delete false proposals.
- Add every missed defect, including secondary defects.
- Assign classes independently to each box.
- Do not infer a class from `cluster_id`; it is only a run-local grouping.
- Escalate ambiguous, occluded, or taxonomy-conflicting examples.

## Multi-defect images

The output contains every detector box for an image. Detector confidences are stored in `detection_score`, cluster-labeled boxes are stored in `detection_bbox_cluster`, and clustering context remains in `instances_json`. The image receives one cluster-local `review_order` within `primary_cluster_id`.

## Error and no-detection rows

- `NO_DETECTION` rows have `[]` for detection box columns. In image-level clustering mode they may still have a `primary_cluster_id` and image-level context in `instances_json`.
- `ERROR` rows are review-inert and must be remediated before any dataset-completeness claim.
