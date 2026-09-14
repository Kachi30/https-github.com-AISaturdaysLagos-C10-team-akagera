#C10-team-Akagera/ 
#Agriculture & Climate SLM Challenge

## Overview
This project builds a compact agriculture/climate question-answering system from a supplied factsheet corpus. It combines **TF-IDF retrieval** with **Qwen2.5-0.5B-Instruct**, fine-tuned using **LoRA**. The design prioritizes factual grounding, concise answers, leakage-safe evaluation, and reproducibility.

## 1. Dataset
The competition data contain four CSV files:

- `documents.csv` — supplied agriculture/climate factsheets; the notebook describes **24 documents**, with three factsheets per topic.
- `train_qa.csv` — **45 supervised question-answer pairs**, with question, topic, crop, agro-zone, document ID, and reference answer.
- `test_questions.csv` — **12 test questions**.
- `sample_submission.csv` — required submission structure.

The notebook automatically searches `/kaggle/input/` for these files and performs missing-value, topic, document-coverage, and answer-length checks. The supplied corpus is described as synthetic; no unrelated external agricultural corpus is added.

One document, `doc_fer_002` (“Organic manure application rates”), has no training question but remains available to retrieval.

### Data creation and augmentation
Each training question is joined to its supplied factsheet. Three instruction templates create grounded variants, producing three training rows per original question. The split is made on the **original `QuestionId` before augmentation**, preventing the same underlying question from appearing in both training and validation.

## 2. Training Pipeline

### Retrieval
Documents are represented using title, topic, crop, agro-zone, and factsheet text. A TF-IDF vectorizer uses lowercase unigram/bigram features (`ngram_range=(1,2)`) with sublinear term frequency. Cosine similarity ranks documents, while metadata filters use topic, crop, and agro-zone.

A second diagnostic baseline uses TF-IDF nearest-neighbour matching to reuse the most similar training answer.

### Model
Base model:

`Qwen/Qwen2.5-0.5B-Instruct`

LoRA configuration:

- `r=16`
- `lora_alpha=32`
- dropout `=0.05`
- target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- bias: `none`
- task: causal language modelling

BF16 is used when supported; otherwise FP16 is used on CUDA.

### Training
Each prompt contains instruction, topic, crop, agro-zone, supporting factsheet, and farmer question. The reference answer is the target. Prompt tokens are masked with `-100` so training loss focuses on the answer.

Main settings:

- epochs: `5`
- train/eval batch size: `2`
- gradient accumulation: `8`
- learning rate: `2e-4`
- weight decay: `0.01`
- maximum length: `768`
- seed: `42`
- evaluation and checkpointing: each epoch
- best model: lowest validation loss

The LoRA adapter is saved as `agri_qwen_lora_final/`.

## 3. Evaluation
The competition metric is **mean character-level Levenshtein distance**; lower is better. Hidden test answers are never used.

The notebook evaluates:

1. **TF-IDF retrieval accuracy** — correct-document retrieval.
2. **Leave-one-out nearest-answer baseline** — similarity-based answer reuse without self-retrieval.
3. **Fine-tuned SLM + retrieval** — held-out questions are retrieved from the corpus and answered by the trained SLM.

Approximately **85% of unique questions** are used for training and **15% for validation**. Since the split is by `QuestionId`, all augmented variants remain together.

Generated answers are cleaned by removing common prefixes and retaining the first sentence. This supports the short-answer objective of the competition.

## 4. Final Inference
For every test question:

1. Retrieve the most relevant factsheet.
2. Provide it, together with metadata and the question, to the fine-tuned SLM.
3. Generate conservatively using greedy decoding (`do_sample=False`).
4. Limit generation to `128` new tokens.
5. Clean the answer.
6. Save `QuestionId,Answer`.

The resulting file is `submission.csv`, validated for exact columns, row count/order, and non-missing answers.

## 5. Reproduction

### Kaggle
1. Add the competition dataset containing the four CSV files.
2. Open `agriculture-climate-slm-challenge-notebook-1(2).ipynb`.
3. Run cells **1–22 in order**.
4. Cells 3–8 load/check data and establish retrieval baselines.
5. Cells 9–13 construct training data, fine-tune Qwen with LoRA, and save the adapter.
6. Cells 14–20 perform resume checks, inference, and held-out evaluation.
7. Cells 21–22 generate and validate `submission.csv`.

A Kaggle GPU is recommended.

### Local
The notebook assumes `/kaggle/input` and `/kaggle/working`. For local execution, place the four CSV files in an accessible directory and modify Cell 3/output paths. Internet access is needed to install packages and download the Qwen model unless cached locally.

## 6. Code and Data

### Code
Complete reproducible implementation:

`agriculture-climate-slm-challenge-notebook-1(2).ipynb`

It contains data loading, validation, retrieval, augmentation, LoRA training, inference, evaluation, and submission generation.

### Data
The CSV files are not embedded in this project. The notebook automatically loads `documents.csv`, `train_qa.csv`, `test_questions.csv`, and `sample_submission.csv` from the Kaggle input directory. This avoids redistributing competition data where licensing or competition rules may restrict it.

## 7. Reproducibility Notes
The main safeguards are:

- fixed seed (`42`);
- QuestionId-level splitting;
- no hidden test answers;
- retrieval-grounded generation;
- concise answer post-processing;
- saved LoRA adapter;
- submission-schema validation.

The central design principle is: **retrieval supplies the agricultural facts; the SLM supplies concise, instruction-following answer generation.**

## Appendix — Contributors and Mentors

### Contributors / Team Members
- Everestus Onyeka Onoyima
- Additional contributors: To be completed.

### Mentors
- Mentor(s): To be completed.

## References
1. Qwen Team. *Qwen2.5-0.5B-Instruct*. Hugging Face model repository.
2. Hu, E. J., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. *arXiv:2106.09685*.
3. scikit-learn documentation. TF-IDF feature extraction and cosine similarity.
4. Agriculture & Climate SLM Challenge competition dataset/task specification.
