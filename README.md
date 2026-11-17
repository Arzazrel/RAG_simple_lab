# RAG_simple_lab

A simple, didactic laboratory to learn **Retrieval-Augmented Generation (RAG)**
with a free, local, instruction-tuned LLM.

The project builds a minimal RAG system in three incremental retrieval levels —
from a naive name-based lookup to semantic search and a web fallback — and lets
you **measure the value of retrieval** through a controlled A/B test on a set of
*invented* definitions.

> **Why invented definitions?** The context files contain made-up terms that do
> not exist in the model's training data. So if the model answers a question
> about one of them correctly, that knowledge can only come from the retrieved
> documents, not from the model's memory. This isolates the effect of retrieval
> cleanly.

---

## Table of contents

- [Features](#features)
- [How it works](#how-it-works)
- [Repository structure](#repository-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Supported models](#supported-models)
- [Local model cache](#local-model-cache)
- [Usage](#usage)
  - [Run the interactive app](#run-the-interactive-app)
  - [Test the model alone](#test-the-model-alone)
  - [Test the retrievers alone](#test-the-retrievers-alone)
  - [Run the A/B evaluation](#run-the-ab-evaluation)
- [The A/B evaluation explained](#the-ab-evaluation-explained)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Features

- **Three retrieval levels**, selectable at runtime:
  1. **Name** — lexical match on file names (baseline, shows the limits of
     lexical matching).
  2. **Embedding** — semantic search with sentence-transformers (the core of RAG).
  3. **Web** — Wikipedia fallback when local evidence is weak (Corrective-RAG style).
- **Multiple generator models** (Qwen2.5-Instruct 3B / 7B / 14B), chosen at runtime.
- **4-bit quantization** (bitsandbytes) with an automatic safety check that
  disables it when the model would not fit in VRAM, instead of crashing.
- **Local caching** of both the LLM weights and the embedding model, so they are
  downloaded only once.
- **A/B evaluation** script that measures answer accuracy with and without RAG,
  retrieval hit rate, and hallucination on control questions.

---

## How it works

```
                 INDEXING (offline)                        QUERYING (online)
   docs/*.txt ─► chunking ─► embedding ─► vector index      user question
                                              ▲                   │
                                              │  top-k similarity  ▼
                                              └──────────── retriever
                                                                  │  retrieved chunks
                                                                  ▼
                                           augmented prompt ─► LLM ─► grounded answer
```

The documents in `docs/` are split into chunks (one definition per paragraph),
embedded into vectors, and searched by cosine similarity. The most relevant
chunks are inserted into the prompt, and the LLM answers using that context.

---

## Repository structure

```
RAG_simple_lab/
├── docs/                   # context files (.txt) with invented definitions
│   ├── instruments.txt
│   ├── creatures.txt
│   └── concepts.txt
├── src/
│   ├── model.py            # LLM: structure, download/cache of weights, generate()
│   └── retrieval.py        # NameRetriever, SemanticRetriever, WebRetriever, HybridRetriever
├── eval/
│   ├── questions.json      # evaluation questions + expected answers / key terms
│   └── evaluate.py         # A/B evaluation script
├── main.py                 # interactive user interface (choose model + retriever)
├── requirements.txt        # Python dependencies
├── .gitignore              # excludes the local model cache, etc.
└── README.md
```

---

## Requirements

- Python 3.10+
- An NVIDIA GPU is recommended (the project was developed on an RTX via WSL).
  It also runs on CPU, but generation is slow.
- Dependencies listed in `requirements.txt` (PyTorch, Transformers, Accelerate,
  sentence-transformers, bitsandbytes, requests).

---

## Installation

```bash
# 1. clone
git clone https://github.com/Arzazrel/RAG_simple_lab.git
cd RAG_simple_lab

# 2. (recommended) create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. install dependencies
pip install -r requirements.txt
```

> **Note on bitsandbytes:** 4-bit quantization requires a CUDA GPU. On CPU-only
> machines the code automatically falls back to full precision.

---

## Supported models

Chosen at runtime in the app. Sizes are approximate; the download size is the
disk space pulled once, the 4-bit VRAM is what the GPU needs with quantization.

| Choice  | Model                       | Download | VRAM (4-bit) | VRAM (fp16) |
|---------|-----------------------------|----------|--------------|-------------|
| `small` | Qwen/Qwen2.5-3B-Instruct    | ~6.2 GB  | ~2.8 GB      | ~7.0 GB     |
| `medium`| Qwen/Qwen2.5-7B-Instruct    | ~15.2 GB | ~5.8 GB      | ~16.0 GB    |
| `large` | Qwen/Qwen2.5-14B-Instruct   | ~29.5 GB | ~12.4 GB     | ~30.0 GB    |

All are Apache-2.0, already instruction-tuned, and need **no Hugging Face login**
to download. The app prints these figures and warns when a model likely will not
fit in 4-bit on your detected GPU.

**Embedding model:** `all-MiniLM-L6-v2` (~90 MB), used by the semantic and web
retrievers.

---

## Local model cache

Both the LLM weights and the embedding model are saved into the local `models/`
folder on first use, via the `cache_dir` / `cache_folder` parameters, and reused
afterwards — so you download each model only once. The `models/` folder is listed
in `.gitignore` and is **not** committed to the repository.

To free disk space, simply delete the `models/` folder; it will be re-created on
the next run.

---

## Usage

All commands are run from the repository root.

### Run the interactive app

```bash
python main.py
```

You will be asked to choose a retriever (`name`, `embedding`, or `web`) and a
model (`small`, `medium`, or `large`). For every question you type, the app prints
the answer **without RAG** and **with RAG**, plus the retrieved context, so you
can compare them directly. Leave the question blank to exit.

### Test the model alone

Loads the default model and asks it a question about an invented term (it should
fail without context — that is the point):

```bash
python src/model.py
```

### Test the retrievers alone

Runs the semantic retriever (and the web retriever, which needs internet) against
the `docs/` folder:

```bash
python src/retrieval.py docs
```

### Run the A/B evaluation

```bash
# semantic retriever, 3 chunks per query
python eval/evaluate.py --retriever embedding --k 3 --verbose

# compare with the naive name-based retriever
python eval/evaluate.py --retriever name

# use the web-fallback hybrid retriever
python eval/evaluate.py --retriever web
```

Options:

| Option         | Default              | Meaning                                         |
|----------------|----------------------|-------------------------------------------------|
| `--questions`  | `eval/questions.json`| evaluation set                                  |
| `--docs`       | `docs`               | context folder                                  |
| `--retriever`  | `semantic`           | `name`, `semantic`/`embedding`, or `web`/`hybrid` |
| `--k`          | `3`                  | number of chunks retrieved per query            |
| `--verbose`    | off                  | print every answer                              |

---

## The A/B evaluation explained

The evaluation compares two conditions on the same questions with the same model,
changing only one variable — the presence of retrieved context:

- **A (baseline):** the model answers alone, no context.
- **B (RAG):** the model answers with the retrieved chunks in the prompt.

Because the definitions are invented, the model cannot know them in condition A.
If it answers correctly in condition B, the knowledge can only come from the
retrieved documents. **The accuracy gap B − A is the measure of the value of
retrieval.**

Three metrics are reported:

1. **Answer accuracy** — fraction of the expected *key terms* found in the answer
   (automatic, no human judgement). Expected: low in A, high in B.
2. **Retrieval hit rate** (condition B) — was the correct source file among the
   retrieved chunks? This separates retriever errors from generator errors.
3. **Hallucination rate** on control questions — two questions ask about terms
   that do **not** exist in the corpus; a good RAG system should admit it has no
   information instead of inventing an answer.

---

## Troubleshooting

**`ValueError: Some modules are dispatched on the CPU or the disk` (4-bit).**
The quantized model does not fit in your GPU's VRAM, so part of it is offloaded
and bitsandbytes refuses. The code now detects this and disables 4-bit
automatically, but you can also pick a smaller model (`small`), or edit
`LLMConfig.load_in_4bit`. Rule of thumb: the 7B needs ~6 GB VRAM in 4-bit, the
14B needs ~12 GB.

**The download is very large / slow.** Each model is pulled once into `models/`
and reused. Prefer `small` for a first run (~6 GB).

**The web retriever returns little / nothing.** It queries the English Wikipedia
and needs internet access. For the invented terms it will correctly find nothing
(they do not exist); test it with a real term to see it working.

---

## License

This project is released under the **GNU General Public License v3.0** — see the
[LICENSE](LICENSE) file.

Author: **Alessandro Diana**
