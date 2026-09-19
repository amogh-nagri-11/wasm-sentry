# Deep model track: pretrain + fine-tune an opcode-sequence classifier

Full guide for the "fine-tuned model" discussed as an alternative to the
current linear classifier. **Status: the design, plus working plumbing, and no
result.** Sequence extraction is built and tested
(`core/src/ml/sequences.ts`, `core/scripts/extract-sequences.ts`). The model
and training loop below exist as runnable code in
`training/opcode_transformer.py`, exercised end to end on a CPU with a handful
of sequences. Nothing has been trained on a real corpus, so nothing here says
whether the idea works. `docs/COLAB-TRAINING-GUIDE.md` is the step-by-step;
the code blocks in this file are the design sketch and the script is the
authority where they differ.

## 0. The decision, stated once

**Self-supervised pretrain a small transformer on opcode sequences from
unlabeled Wasm, then fine-tune a classification head on the small labeled
corpus, export to ONNX, run it in-browser via `onnxruntime-web`.**

Why this and not the alternatives already discussed:

- **Not "fine-tune an existing pretrained LM"** (e.g. CodeBERT). Those are
  pretrained on source code with meaningful identifiers and structure.
  Production Wasm — especially the miners you're detecting — is stripped and
  optimized; there are no names left, only opcode mnemonics. A source-code
  tokenizer fragments that badly and the pretraining doesn't transfer. You'd
  be paying the complexity of using someone else's model for none of the
  benefit.
- **Not an API-based LLM**, per the last discussion — wrong fit for a
  per-page-load, privacy-sensitive, real-time check.
- **Is a genuinely new pretraining step**, because WasmBench's ~8,000
  binaries are enough to learn *what normal instruction sequences look like*
  even though only a few hundred are usable as *labeled* fine-tuning
  examples. That asymmetry — lots of unlabeled data, very little labeled
  data — is exactly the situation self-supervised pretraining exists for.
- **Reuses the project's own opcode vocabulary** (`OPCODE_VOCABULARY` in
  `core/src/ml/features.ts`) as the token set, so the deep model and the
  existing linear model agree on what an "operation" is, and the deep model
  isn't a second, disconnected feature-engineering effort.

## 1. Dataset — what you need, and where it differs from `DATASET-PLAN.md`

Two pools, not one:

| | Purpose | Labels needed | Source |
|---|---|---|---|
| **Pretraining pool** | Learn what instruction sequences normally look like | None | All of filtered WasmBench (8,461 modules) + the npm harvest from `DATASET-PLAN.md` §1.2. No manual review needed — mislabeled miners in an *unlabeled* pretraining set cost you nothing, because nothing is being labeled. |
| **Fine-tuning corpus** | Learn benign vs. malicious | Yes | Exactly `$CORPUS` from `docs/DATASET-PLAN.md` — same benign/malicious directories, same filter → dedupe → cluster → holdout pipeline. Don't rebuild this; reuse it. |

So: run `DATASET-PLAN.md` end to end first (you're already doing this). The
only new acquisition step is skipping the miner-keyword/heuristic filtering
on WasmBench for the pretraining pool — grab the whole filtered set as-is:

```bash
export PRETRAIN_POOL=~/wasm-sentry-pretrain-pool
mkdir -p "$PRETRAIN_POOL"
cp /tmp/wasmbench/filtered/* "$PRETRAIN_POOL"/ 2>/dev/null
cp /tmp/npm-wasm/**/*.wasm "$PRETRAIN_POOL"/ 2>/dev/null
find "$PRETRAIN_POOL" -type f | wc -l   # expect ~8,000-9,000
```

## 2. Extract opcode sequences (built: `core/scripts/extract-sequences.ts`)

Neither `analyzeWasm` nor `vectorise` currently emit a raw instruction
*sequence* — only aggregate counts and ratios (that's what the linear model
needs, and it's correct that it doesn't carry more). The deep model needs
the sequence itself, so this is one new script:
`core/scripts/extract-sequences.ts`, run once per pool.

**As built, it differs from the sketch below in three ways**, all recorded in
`core/src/ml/sequences.ts`: the vocabulary has six special tokens (`PAD`,
`MASK`, `CLS`, `UNK`, `OTHER`, `SEP`), 44 ids in all, written to a
`.vocab.json` beside the output so nothing downstream hard-codes a size; a
function too long for the window is cut at the header of its largest loop
rather than from the top, because a kernel's prologue looks like anyone's; and
when the primary function does not fill the window, the next-largest functions
follow behind a `SEP`, up to four. Bodies are re-decoded for the chosen
functions only, so the feature walk did not need a per-instruction hook.

**The original specification:**

1. For each `.wasm` file, run `analyzeWasm` as today to get `kernelCandidate`
   and function boundaries — reuse the existing walk in
   `core/src/wasm/features.ts` rather than re-parsing in Python. Adding a
   sequence-emitting hook there (a callback invoked per decoded instruction,
   alongside the existing counting) is a smaller change than writing a
   second Wasm parser.
2. Per module, select **the kernel-candidate function if one exists,
   otherwise the largest function by instruction count, otherwise
   concatenate the top 4 functions by size** until the length budget below
   is filled. This mirrors the heuristic engine's own judgment about which
   function matters most (`core/src/heuristics.ts:97`), so the deep model is
   looking at the same evidence the rules already look at.
3. Map every instruction to a token: opcodes in `OPCODE_VOCABULARY` keep
   their name; everything else becomes `OTHER`. Add three special tokens:
   `PAD`, `CLS` (prepended, its final hidden state is what the classifier
   head reads), `UNK` (unused today, reserved for vocabulary growth).
4. Fix the sequence length at **512 tokens** — truncate longer, pad shorter.
   (512 covers the median kernel-candidate function with room to spare; this
   is a design choice, not a measured optimum — treat it as a hyperparameter
   to sweep once the pipeline runs end to end, not before.)
5. Write one JSONL file per pool: `{"sha256": "...", "tokens": [1, 45, 2, ...], "label": null}`
   for the pretraining pool (`label` always `null`) and
   `{"sha256": "...", "tokens": [...], "label": 0 | 1}` for the fine-tuning
   corpus (label from which directory — `benign/` or `malicious/` — the file
   came from, same convention as `train-model.ts`).

```bash
npm run extract-sequences -w @wasm-sentry/core -- "$PRETRAIN_POOL" --out pretrain.jsonl --unlabeled
npm run extract-sequences -w @wasm-sentry/core -- "$CORPUS" --out finetune.jsonl
```

(Register these as npm scripts the same way the four corpus scripts were —
`"extract-sequences": "tsx scripts/extract-sequences.ts"` in `core/package.json`.)

## 3. Environment for the Python side

The TS side stays zero-dependency per the project's own design principle;
the training side is a separate, throwaway environment — don't add PyTorch
to `core/package.json` or anything Node touches.

```bash
python3 -m venv ~/wasm-sentry-ml-env
source ~/wasm-sentry-ml-env/bin/activate
pip install torch onnx onnxruntime numpy
```

No `transformers` dependency — the model is small enough (see below) to
write directly in ~150 lines of PyTorch, which avoids pulling in a
multi-gigabyte dependency for a 2-3M parameter model and keeps the
architecture fully visible instead of hidden behind a library default.

## 4. Architecture

Deliberately tiny — this is closer to a toy transformer than a production
LLM, and it should be:

| Hyperparameter | Value | Why |
|---|---|---|
| Vocabulary size | `len(OPCODE_VOCABULARY) + 4` (≈40) | Matches the project's existing opcode set + special tokens |
| Sequence length | 512 | §2.4 |
| Embedding dim | 128 | Small vocab doesn't need more |
| Layers | 4 | Enough to compose "this loop body does bitwise ops on loaded memory," not more |
| Attention heads | 4 | Standard 32-dim-per-head split |
| Feedforward dim | 512 | 4x embedding, standard ratio |
| Parameters | ~2-3M | Trains on a laptop CPU in hours, on the optional GPU in minutes |

This is intentionally far smaller than any general-purpose language model —
the vocabulary is ~40 tokens, not ~50,000, because it only ever has to
represent WebAssembly opcodes, not natural language.

```python
import torch
import torch.nn as nn

class OpcodeTransformer(nn.Module):
    def __init__(self, vocab_size=40, seq_len=512, dim=128, layers=4, heads=4, ff=512):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.pos_emb = nn.Embedding(seq_len, dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=ff, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.mlm_head = nn.Linear(dim, vocab_size)       # pretraining
        self.cls_head = nn.Linear(dim, 1)                 # fine-tuning

    def encode(self, tokens, attention_mask):
        positions = torch.arange(tokens.size(1), device=tokens.device)
        x = self.token_emb(tokens) + self.pos_emb(positions)
        return self.encoder(x, src_key_padding_mask=~attention_mask.bool())

    def forward_mlm(self, tokens, attention_mask):
        return self.mlm_head(self.encode(tokens, attention_mask))

    def forward_cls(self, tokens, attention_mask):
        hidden = self.encode(tokens, attention_mask)
        return self.cls_head(hidden[:, 0, :])   # CLS token
```

## 5. Pretrain — masked opcode modeling

Same idea as BERT's masked language modeling, applied to opcodes instead of
words: mask 15% of tokens, train the model to predict them from context. No
labels required — this is what makes the 8,000-module pool usable at all.

```python
import json, random
import torch
from torch.utils.data import Dataset, DataLoader

MASK_TOKEN = 1   # reserve index 1 for [MASK] in your vocab, alongside PAD=0, CLS=2, UNK=3

class PretrainDataset(Dataset):
    def __init__(self, jsonl_path):
        self.rows = [json.loads(l) for l in open(jsonl_path)]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        tokens = list(self.rows[i]["tokens"])
        labels = [-100] * len(tokens)   # -100 = ignored by cross-entropy
        for j in range(len(tokens)):
            if tokens[j] == 0:  # don't mask padding
                continue
            if random.random() < 0.15:
                labels[j] = tokens[j]
                tokens[j] = MASK_TOKEN
        mask = [1 if t != 0 else 0 for t in self.rows[i]["tokens"]]
        return torch.tensor(tokens), torch.tensor(mask), torch.tensor(labels)

def pretrain(jsonl_path, epochs=20, batch_size=32, lr=1e-4):
    model = OpcodeTransformer()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    loader = DataLoader(PretrainDataset(jsonl_path), batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0.0
        for tokens, mask, labels in loader:
            opt.zero_grad()
            logits = model.forward_mlm(tokens, mask)
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"epoch {epoch}: loss {total_loss / len(loader):.4f}")

    torch.save(model.state_dict(), "pretrained.pt")
    return model
```

Expect this phase to take the bulk of the wall-clock time in the whole
track — hours on CPU, well under an hour on the optional GPU the synopsis's
own hardware table allows for.

## 6. Fine-tune — classification head on the labeled corpus

Load the pretrained weights, replace nothing (the `cls_head` already exists,
just wasn't trained yet), train on `finetune.jsonl` with the same discipline
already established in `core/scripts/train-model.ts`: class-weighted loss
(the corpus is realistically 20:1+ benign:malicious, same argument as
`DATASET-PLAN.md` §3.5), k-fold cross-validation, and a held-out set you
touch exactly once.

```python
def finetune(jsonl_path, pretrained_path="pretrained.pt", epochs=10, lr=2e-5):
    model = OpcodeTransformer()
    model.load_state_dict(torch.load(pretrained_path), strict=False)

    rows = [json.loads(l) for l in open(jsonl_path)]
    n_pos = sum(r["label"] for r in rows)
    n_neg = len(rows) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)])   # same balancing intent as balanceClasses in train.ts

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    # k-fold split, cross-validate, report precision/recall/F1/AUC --
    # same shape as crossValidate() in core/src/ml/evaluate.ts. Do not
    # skip this: a single train/test split on a few hundred examples is
    # noise, exactly as already argued for the linear model.
    ...
```

**Compare against two baselines on the same folds, not one:**

1. The existing heuristic engine (`assessRisk`) — the project's own floor.
2. The existing trained logistic regression (`model.json`, once it exists
   from `DATASET-PLAN.md`) — the cheaper model this one has to beat, not
   just tie.

If the deep model doesn't clear both by a real margin, on the held-out set,
the honest conclusion is the same one `train-model.ts` already prints for
the linear model: ship the simpler thing.

## 7. Export for the browser

```python
dummy_tokens = torch.zeros(1, 512, dtype=torch.long)
dummy_mask = torch.ones(1, 512, dtype=torch.long)
torch.onnx.export(
    model, (dummy_tokens, dummy_mask), "model.onnx",
    input_names=["tokens", "attention_mask"], output_names=["logit"],
    dynamic_axes={"tokens": {0: "batch"}, "attention_mask": {0: "batch"}},
    opset_version=17,
)
```

Then quantize to int8 (roughly a 4x size cut, usually a small accuracy
cost worth re-measuring on the holdout set before deciding it's acceptable):

```bash
python -m onnxruntime.quantization.quantize --input model.onnx --output model.quant.onnx --per_channel
```

Expect single-digit MB even quantized — compare that explicitly against the
few-KB `model.json` the linear model produces before deciding this belongs
in the default install versus an optional download.

## 8. Extension integration (scoped out of this guide)

Loading `model.quant.onnx` via `onnxruntime-web` inside the service worker,
lazy-loading it (MV3's 30-second idle kill makes a large synchronous load at
startup risky, per the concern already raised about the capture layer), and
wiring its output into the same risk-aggregation path the linear model
writes to — that's a real implementation task, not a training-guide
footnote. Treat it as the next phase once §6's comparison actually justifies
shipping this at all.

## 9. What to expect

Realistically: on a fine-tuning set of a few hundred examples, a 2-3M
parameter transformer — even pretrained — is not guaranteed to beat a
73-feature linear model that already encodes the exact structural signals
(bitwise ratio, kernel candidate, memory shape) domain knowledge says matter.
Pretraining buys generalization *within the range of what WasmBench's normal
Wasm looks like*; it does not manufacture malicious examples you don't have.
Report the same things `DATASET-PLAN.md` §5 asks for, plus which of the two
existing baselines (if either) it actually beat.
