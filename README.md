# DSA4213 Assignment 

**Language models, frozen embeddings and sentiment representations on IMDb**

Name: Dai Chenze  Student ID: A0327453A


## 1. Quick start

Use Python 3.12 and run the following commands from the project directory.

**Windows PowerShell** (activation is not required):

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
# Download GloVe and extract glove.6B.100d.txt to data/glove.6B/ (Section 3).
.\.venv\Scripts\python.exe -m notebook Assignment.ipynb
```

**macOS / Linux**:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
# Download GloVe and extract glove.6B.100d.txt to data/glove.6B/ (Section 3).
.venv/bin/python -m notebook Assignment.ipynb
```

In the browser, select the Python kernel from this environment, restart the kernel, and choose **Run All** to reproduce the experiments.

**Notebook startup dependencies:** `notebook` provides the browser interface and `python -m notebook` launcher; `ipykernel` executes Python cells. Both are included in `requirements.txt`, along with the experiment dependencies. The Notebook interface uses the supported range `notebook>=7,<8`; this is a startup dependency, not a recorded experiment package version. Internet access is needed to install dependencies and download IMDb/GloVe. Alternatively, open the notebook in VS Code with its Python and Jupyter extensions and select the `.venv` interpreter.

> **Warning:** Run All retrains every model and **overwrites `results/`**. Back up the submitted results first. No API key is needed.

Environment used: Python 3.12.13, PyTorch 2.14.0+cu130, gensim 4.4.0, scikit-learn (see `requirements.txt`), NVIDIA RTX 3060 Laptop GPU, Windows. A full run took about 87 minutes. CPU-only execution is supported but was not timed (expect it to be much slower). Install the PyTorch build matching your CUDA version if the pinned one is unavailable.

## 2. Files

| Path | Role |
|---|---|
| `Assignment.ipynb` | Single executable source: all experiments, result checks, provenance |
| `scripts/build_report.py` | Formats the PDF from saved results only (no training) |
| `results/*_results.json`, `*_history.json` | Part I/II metrics, generation samples, learning curves, timings |
| `results/part3/` | Part III classifier results, features, selected errors |
| `results/supplementary/` | Extension results (generation, embedding diagnostics, error analysis, environment); the name is historical, no separate run step |
| `results/run_provenance.json` | Run time, notebook hash, data and output hashes |
| `data/split_config.json`, `data/reproduction_manifest.json`, `data/current_run_manifest.json` | Split config and SHA-256 fingerprints for checking a regenerated split |
| `PACKAGE_MANIFEST.json` | SHA-256 of submitted source and numeric evidence |

## 3. Data

- **IMDb:** loaded by the notebook with `datasets.load_dataset('stanfordnlp/imdb')`. The dataset revision was not pinned; use the SHA-256 hashes in `data/reproduction_manifest.json` to verify the regenerated split. Check the dataset's usage terms.
- **GloVe 6B 100d:** download `glove.6B.zip` from https://nlp.stanford.edu/projects/glove/ and extract `glove.6B.100d.txt` to `data/glove.6B/`. Do not use another release. The notebook checks the file before training.
- **GitHub repository contents:** notebook, PDF builder, report, README, requirements, numeric results, curves, saved predictions, review splits, token-ID files, vocabulary, trained Skip-gram vectors, and classifier features/models. Downloaded GloVe files and neural model checkpoints (`.pt`) are excluded by `.gitignore`. Download GloVe 6B 100d as described above; running the notebook regenerates checkpoints. To submit a ZIP with the same contents, export the current Git-tracked files.
- **Full local working folder:** also retains those data, vectors, checkpoints and features. Only the 100d GloVe file is used. External vectors and checkpoints are excluded from `PACKAGE_MANIFEST.json`. The existing parent-directory ZIP predates the latest edits and must be repackaged before submission.

## 4. Preprocessing

1. Deduplicate the official test pool first (lowercase, collapse whitespace, keep first occurrence). Remove training reviews matching any official-test text; keep the first occurrence of other duplicates.
2. Seed 42: stratified 6,000-review training-pool sample → 5,000 train / 1,000 validation; stratified 1,000 reviews from the official test pool.
3. Tokenize: unescape HTML, remove line breaks, lowercase, regex for words (with internal apostrophes), numbers, punctuation.
4. Vocabulary from train only: max 20,000 entries including PAD/UNK/BOS/EOS, min count 2, ties broken alphabetically. Unseen words → UNK.
5. Each review: input = BOS + tokens, target = tokens + EOS. PAD and BOS are never targets. Neural chunks ≤ 128 positions; recurrent states and Transformer positions reset between chunks.

## 5. Configurations

| Item | Setting |
|---|---|
| Trigram | add-0.1 unigram, interpolated bigram/trigram MLE; 400-pair validation grid selects (w2, w3) = (0.6, 0.25) |
| Neural (all) | 100-d embeddings, untied output layer, Adam lr 0.001, batch 16, grad clip 1.0, max 20 epochs, patience 3, tolerance 0.0001, best-validation checkpoint restored |
| RNN / LSTM | 1 layer, hidden 256 |
| Transformer | 2 causal layers, 4 heads, d=100, FFN 400, learned positions, GELU, pre-LN, dropout 0.1; extra run with 40-epoch cap |
| Skip-gram (own) | gensim, dim 100, window 5, 5 negatives, min count 2, subsampling 0.001, 5 epochs, 1 worker, seed 42 |
| Public embeddings | GloVe 6B 100d; missing and UNK/BOS/EOS rows ~ N(0,1), PAD = 0; tables fully frozen |
| Sentiment classifiers | StandardScaler (train only) + logistic regression (L-BFGS, max 2,000 iterations), C ∈ {0.001, 0.01, 0.1, 1, 10} by validation macro-F1; TF-IDF baseline: 1–2 grams, min_df 2, ≤ 50,000 features, sublinear TF |
| Generation | 5 prompts × 3 samples × 6 models; temperature 1; full sampling vs top-k 40; max 40 new tokens; stop at EOS |

## 6. Random seeds

| Use | Seed |
|---|---|
| Split, vocabulary, original models, Skip-gram | 42 |
| Part II repeated runs | 42, 43, 44 |
| Bootstrap and random error samples | 4213 |
| Generation | 100000 + model_index×1000 + prompt_index×10 + sample_index |

Seeds 43/44 vary LM initialization and batch order only; pretrained tables stay fixed. Exact numerical identity across devices is not guaranteed.

## 7. Expected results (test, seed 42 unless stated)

| Experiment | Result |
|---|---|
| Part I test PPL | LSTM 170.46, RNN 182.34, Transformer 205.10, trigram 220.60 (grid: 216.02); Transformer cap 40: 204.28 |
| Part II test PPL, mean ± SD over 3 seeds | random trainable 171.89 ± 1.31; frozen Skip-gram 170.94 ± 0.92; frozen GloVe 173.80 ± 2.40; random frozen (1 seed) 208.90 |
| Part III test accuracy | LSTM 69.1%, Transformer 70.7%, LSTM+PCA100 67.0%, TF-IDF 86.3% |

On 16 October 2026, classifiers and paired statistics were rerun from saved features with the expanded C grid; LM results were unchanged. Selected C: LSTM 0.1, Transformer 0.01, PCA 0.001, TF-IDF 10. The partial update is recorded in `results/run_provenance.json`.

Timings depend on hardware and runtime load and will differ.

## 8. Report-to-file map

| Report item | Source |
|---|---|
| Table 1 | `data/split_config.json`, `data/*_tokens.jsonl` |
| Table 2 | Model definitions in `Assignment.ipynb`, `results/*_results.json` |
| Table 3, Fig. 1, Table B1 | `results/*_results.json`, `*_history.json`, `results/supplementary/timing.json` |
| Table 5, Fig. 2, Table A1 | `results/`, `results/supplementary/multiseed.json`, `results/supplementary/training/` |
| Table 6, Table D2 | `results/part3/`, `results/supplementary/classification.json` |
| Table 7, Table D1, Appendix D random examples | `results/supplementary/error_analysis.json`; paired subgroup check uses `test_predictions.npz` and `data/test_tokens.jsonl` |
| Appendix D high-confidence examples (#605, #447) | `results/part3/error_examples.json` |
| Table 4, Table B2, Appendix C | `results/supplementary/generation.json` |
| Table B3 | `results/supplementary/embedding_diagnostics.json` |

To audit saved results without retraining, open Assignment.ipynb from the project directory and restart the kernel. Run the two code cells under “Environment”, followed by the code cell under “Execution contract and traceability”. Then run only the code cell under “Final result consistency checks”. Do not run the intervening training cells or the “Run provenance” section.

## 9. Attribution and tool use

- IMDb (Hugging Face `stanfordnlp/imdb`); GloVe (Pennington et al., 2014); Skip-gram (Mikolov et al., 2013); LSTM (Hochreiter & Schmidhuber, 1997); Transformer (Vaswani et al., 2017); Adam (Kingma & Ba, 2015). Libraries: PyTorch, gensim, scikit-learn, Hugging Face `datasets`.
- AI tools: OpenAI Codex assisted with code and PDF preparation. I am responsible for the experimental design and conclusions and have checked them.
- Limitations: one split, three optimization seeds, supplementary experiments designed after seeing original test results.
