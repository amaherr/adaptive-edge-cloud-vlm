# Physical VQA dataset freeze v1

The first real task campaign uses 100 VQA v2 validation questions and their 100 COCO val2014 images. The request manifest is [`data/manifests/physical_vqa_v1.jsonl`](../data/manifests/physical_vqa_v1.jsonl), SHA-256 `4e62826280e49359e8cf98d85154000c1583d63474be83a2e9a429da9bde72c9`. Each row freezes the question, ten human answers, image ID/path, and SHA-256 of the image bytes. Selection uses `random.Random(42).sample` over sorted paired question IDs. The dataset revision is `vqa-v2-val2014-physical-v1`; the model contract remains `e7ef01806bfa51ee6cedb15a3331598bc4713ff8a7538fb041fd3d0aaf294aa1`, with a 32-new-token generation cap.

The original [VQA v2 download page](https://visualqa.org/download) supplies the validation questions and annotations; its [terms](https://visualqa.org/terms.html) identify the annotation license. The questions archive SHA-256 is `e71f6c5c3e97a51d050f28243e262b28cd0c48d11a6b4632d769d30d3f93222a`, and the extracted JSON SHA-256 is `f34d9c9909dea76700d361baddd63dd47b4494945097459b542d9fe840244811`. The annotations archive SHA-256 is `0caae7c8d1dafd852727f5ac046bc1efca9b72026bd6ffa34fc489f3a7b3291e`, and the extracted JSON SHA-256 is `0564650e712b34c246e4ad142894692bf2e45dee8e9cc6ebb2cedbe85731c4bf`. The 100 selected images were fetched individually from the [official COCO validation bucket](https://images.cocodataset.org/zips/val2014.zip) through its TLS-valid Amazon S3 endpoint. Source archives and images remain in ignored `data/raw/`; the repository does not redistribute them.

The prospective split is [`configs/evaluation/physical_vqa_split_v1.json`](../configs/evaluation/physical_vqa_split_v1.json), SHA-256 `608e06c06e3916b79e33eda400f324a6bc9d2db73ccc6549fa1214277945e3cc`. Seed 1707 assigns 60 request IDs to training, 20 to validation, and 20 to final evaluation, with no overlap. Training network blocks are N1, N3, N5, N6, and N8; validation blocks are N2 and N7; held-out evaluation blocks are N4 and N9. N4 is an unseen intermediate bandwidth, and N9 has an unseen added RTT. The 60-second SLA, bootstrap seed 1707, 2,000 resamples, and 95% confidence level are declared before profiling. Final evaluation requests and blocks must remain untouched during calibration, threshold choice, and predictor development.

[`configs/experiments/physical_vqa_v1.yaml`](../configs/experiments/physical_vqa_v1.yaml) points at the frozen manifest and revision. It requests one warmup and three measured repetitions per policy and network block, as required by the prospective protocol.

The manifest stores paths relative to its own directory, so the same tracked file can load on cloud and edge. The edge must fetch the same ignored source images before running it. After pulling this commit, use the official VQA URLs and the committed downloader:

```bash
mkdir -p data/raw/vqa
curl -fL -o data/raw/vqa/v2_Questions_Val_mscoco.zip https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip
python3 -m zipfile -e data/raw/vqa/v2_Questions_Val_mscoco.zip data/raw/vqa
python3 scripts/fetch_vqa_subset_images.py --questions data/raw/vqa/v2_OpenEnded_mscoco_val2014_questions.json --images-dir data/raw/coco/val2014 --count 100 --seed 42
.venv-model/bin/python -c 'from adaptive_vlm.data.manifest import load_frozen_manifest; m=load_frozen_manifest("data/manifests/physical_vqa_v1.jsonl", expected_revision="vqa-v2-val2014-physical-v1"); print(len(m.items), m.sha256)'
```

The last command verifies all 100 image hashes. It must print `100` and the frozen manifest SHA above. The original annotation JSON is also required on the host that runs the official VQA evaluator; fetch it from the same official page and verify its hash above. The importer now records relative image paths to support this cross-host contract. Do not regenerate or resample the manifest or split after profiling begins. Official VQA accuracy requires the separate official evaluator; the in-run consensus score is provisional.

P6 is frozen on the cloud checkout. Edge materialization and hash verification are the first steps before the physical profiling campaign.
