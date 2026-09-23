# Reranker

The reranker is a **cross-encoder**: it reads the query and a candidate passage
*together* and scores their relevance. That is far more accurate than the
bi-encoder used for retrieval, but too slow to run over a whole corpus — which
is exactly why it is applied only to the top candidates returned by dense +
sparse fusion.

It is **enabled by default** and its code ships as a core dependency. What it
needs from you is a model file: a 100–500 MB ONNX graph that cannot live in a
git repository, so it is a one-time download described below.

If you would rather run without it, set `"enabled": false` under `reranker` in
`config.json`. Hybrid search keeps working; you only lose the fine reordering.
The `rerank` argument on each `search_documents` call also overrides it per
query.

---

## Why a second stage at all

Retrieval optimises for *recall*: get the right document into the candidate
pool. Ranking those candidates well is a different problem. A cross-encoder
sees both texts at once, so it can tell that "how do I rotate a log file"
matches a paragraph about *log rotation* even when the wording differs.

The trade-off is cost: scoring is `O(candidates)` forward passes, not a single
vector comparison.

---

## Getting a model

Any ONNX cross-encoder with a compatible tokenizer works. The engine expects
two files in one directory:

```
data/reranker/multilingual-cross-encoder/
├── model.onnx        # graph with 2 output logits per pair
└── tokenizer.json    # HuggingFace fast-tokenizer file
```

A common choice is a multilingual MS MARCO cross-encoder. Export it with
Optimum:

```bash
pip install optimum[exporters] onnx
optimum-cli export onnx \
  --model cross-encoder/ms-marco-MiniLM-L-6-v2 \
  --task text-classification \
  data/reranker/multilingual-cross-encoder/
```

Then point the config at it:

```json
"reranker": {
  "enabled": true,
  "model_dir": "data/reranker/multilingual-cross-encoder",
  "onnx_file": "model.onnx",
  "tokenizer_file": "tokenizer.json",
  "max_length": 512,
  "provider": "cpu",
  "relevant_logit_index": 1
}
```

Verify the install with the `reranker_status` MCP tool, or:

```bash
nisaba-rag search "your query" --mode hybrid --rerank
python tools/verify_reranker.py --model-dir /path/to/model    # full check
```

`tools/verify_reranker.py` is the authoritative check: it loads the graph,
confirms a relevant passage outscores an unrelated one, times the inference,
verifies scoring is batch-independent, and runs the full dense → rerank
pipeline. A healthy run looks like this:

```
-- discrimination --
        relevant margin   : +6.963
        irrelevant margin : +3.366
  ok    relevant passage outscores an unrelated one

-- latency --
        ~15 ms per pair (20 pairs scored)

-- scoring is batch-independent --
        alone   : +6.963265
        in batch: +6.963265
  ok    same pair scores identically regardless of companions
```

That last check is not cosmetic — it is the property that the "score pairs one
at a time" design exists to guarantee.

---

## Two details that decide output quality

Both are implemented in `nisaba_rag/rerank.py`; they are documented here
because getting either wrong quietly degrades ranking.

### 1. Score pairs one at a time

Batching passages means padding them to the longest member of the batch. That
padding changes the attention mask and **shifts the logits**: the same
query-passage pair can score measurably differently depending on which other
passages happened to share its batch. For close candidates that is enough to
invert the order.

Scoring one pair per forward pass removes the dependency entirely. The cost is
negligible relative to retrieval.

### 2. Rank on the logit margin, not the softmax probability

A two-class softmax over this kind of checkpoint saturates hard — outputs like
`0.9990` and `0.9999` for passages of very different quality. Converting to a
probability throws away the resolution you need to order the top candidates,
and the "best" chunk can end up being one that never mentions the query.

The **logit margin** (relevant minus irrelevant) is monotonic with that
probability but keeps its resolution. Nisaba ranks on the margin and exposes a
temperature-scaled sigmoid as a readable `score` in `[0, 1]`.

If your checkpoint puts the relevant class in column 0 instead of 1, set
`reranker.relevant_logit_index` to `0`.

---

## Execution providers

`reranker.provider` selects the ONNX Runtime execution provider:

| Value | Provider | Notes |
|---|---|---|
| `cpu` | CPU | Default. Most portable and predictable. |
| `cuda` | CUDA | NVIDIA GPUs. |
| `dml` | DirectML | Windows GPUs (AMD/Intel/NVIDIA). |
| `coreml` | CoreML | Apple silicon. |

If the requested provider is not present in your ONNX Runtime build, Nisaba
falls back to CPU rather than failing.

> **A caveat on GPU providers.** Some driver/runtime combinations crash
> natively inside the inference thread — an access violation that kills the
> whole process and cannot be caught from Python. If you enable a GPU provider
> and see hard crashes under load, set `provider` back to `cpu` and confirm
> which one is at fault before filing a bug. This is why CPU is the default.

---

## Tuning

| Setting | Effect |
|---|---|
| `max_length` | Longer truncation keeps more context but costs time. 512 is a good default. |
| `temperature` | Only rescales the displayed `score`; it does not change the ranking. |
| `relevant_logit_index` | Which output column is the relevant class. Set `0` if ranking looks inverted. |
