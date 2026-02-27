Master Degree Thesis

## Run solution
- Ensure having the following installed:
    - [Docker](https://www.docker.com/)
    - [VS Code](https://code.visualstudio.com/)
        - [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension

- It is sufficient to use _Dev Containers_ extension's 'Dev Containers: Rebuild and Reopen in Container' command.

## Project setup
- [uv](https://docs.astral.sh/uv/) for package management

## Shared resources
- [Google Drive](https://drive.google.com/drive/folders/1zr0JyMETI44G2ikBb93U3lLXUdiOqqnZ?usp=sharing) for data (_gest.csv_) and fine-tuned models.

<!-- TODO: Mention about settings required to be made for using Gmail provider

Steps
1.	Create / select a project → APIs & Services ▸ Enabled APIs & services ▸ + ENABLE APIs → turn on Gmail API.
2.	OAuth consent screen → External → add your own Google account as a Test user.
3.	Create credentials → OAuth client ID ▸ Desktop app.
•	Download the client_secret_<…>.json file; place it next to your code (or mount it into Docker) and keep it private.
4.	First run locally: the script below pops a browser asking you to log in and grant the single scope https://www.googleapis.com/auth/gmail.send.
Remarks: For 4 run 'Init gcloud CLI' task
Google returns a refresh token which ends up in token.json; afterwards the script renews access tokens silently.

Resources:
1. (Create Gmail API App in the Google Developer Console)](https://www.youtube.com/watch?app=desktop&v=1Ua0Eplg75M)
2. ChatGPT
 -->

## [Narrative Similarity Task](https://narrative-similarity-task.github.io/) (Track A)
The Track A pipeline is available under `gest.service.evaluation.narrative_similarity.track_a`.

Optional metrics:
 - `bleurt` requires `bleurt-pytorch` and an available checkpoint.

End-to-end flow:

0) (Optional) Generate LLM outputs for Track A triples  
Store outputs in `data/Narrative Similarity Task/development` and `data/Narrative Similarity Task/test` before LLM-aware evaluation/hybrid runs.
```bash
python -m gest.service.evaluation.narrative_similarity.generate_track_a_llm_outputs \
  --model-name Qwen/Qwen3-32B \
  --dataset-path "miscellaneous/datasets/Narrative Similarity Task/development/dev_track_a.jsonl" \
  --output-path "data/Narrative Similarity Task/development/dev_track_a_Qwen3-32B.jsonl"
```

```bash
python -m gest.service.evaluation.narrative_similarity.generate_track_a_llm_outputs \
  --model-name Qwen/Qwen3-32B \
  --dataset-path "miscellaneous/datasets/Narrative Similarity Task/test/test_track_a.jsonl" \
  --output-path "data/Narrative Similarity Task/test/test_track_a_Qwen3-32B.jsonl"
```

1) Evaluate all graph/text metrics on dev (and optionally LLM outputs)
```bash
python -m gest.service.evaluation.narrative_similarity.track_a evaluate-all \
  --dataset dev \
  --text-metrics sbert_cosine,bleurt,bleu,rouge_l \
  --output-dir "results/Narrative Similarity Task/metrics"
```

Optional: include LLM output files in the same `metrics_summary.csv`:
```bash
python -m gest.service.evaluation.narrative_similarity.track_a evaluate-all \
  --dataset dev \
  --text-metrics sbert_cosine,bleurt,bleu,rouge_l \
  --llm-output-dir "data/Narrative Similarity Task/development" \
  --llm-output-glob "dev_track_a_*.jsonl" \
  --llm-modes chosen,score \
  --output-dir "results/Narrative Similarity Task/metrics"
```
This also writes `results/Narrative Similarity Task/metrics/llm_parse_summary.csv`.

2) (Optional) Plot score histograms from `scores.csv`
```bash
python -m gest.service.evaluation.narrative_similarity.plot_all_metric_histograms \
  --scores "results/Narrative Similarity Task/metrics/scores.csv" \
  --output-dir "results/Narrative Similarity Task/metrics/histograms"
```

3) Direct submission (single metric)
```bash
python -m gest.service.evaluation.narrative_similarity.track_a submit \
  --dataset test \
  --metric gest_spectral_w2v_google_temporal \
  --output "results/Narrative Similarity Task/submissions/track_a.jsonl"
```

```bash
python -m gest.service.evaluation.narrative_similarity.submit \
  --input "results/Narrative Similarity Task/submissions/track_a.jsonl" \
  --output "results/Narrative Similarity Task/submissions/submission.zip"
```

4) Combined-score submission (2 metrics, learned alpha/beta/bias)
```bash
python -m gest.service.evaluation.narrative_similarity.track_a search \
  --train sample,synthetic \
  --dev dev \
  --metric-a gest_spectral_w2v_google_temporal \
  --metric-b sbert_cosine \
  --method logreg \
  --save-weights "results/Narrative Similarity Task/submissions/weights_combo.json"
```

```bash
python -m gest.service.evaluation.narrative_similarity.track_a submit \
  --dataset test \
  --weights-file "results/Narrative Similarity Task/submissions/weights_combo.json" \
  --output "results/Narrative Similarity Task/submissions/track_a.jsonl"
```

```bash
python -m gest.service.evaluation.narrative_similarity.submit \
  --input "results/Narrative Similarity Task/submissions/track_a.jsonl" \
  --output "results/Narrative Similarity Task/submissions/submission.zip"
```

5) Local evaluation helpers
```bash
python -m gest.service.evaluation.narrative_similarity.track_a evaluate \
  --dataset dev \
  --weights-file "results/Narrative Similarity Task/submissions/weights_combo.json"
```

```bash
python -m gest.service.evaluation.narrative_similarity.track_a evaluate \
  --dataset test \
  --weights-file "results/Narrative Similarity Task/submissions/weights_combo.json" \
  --labels-file "miscellaneous/datasets/Narrative Similarity Task/test/labels/test_track_a_labels.jsonl"
```
