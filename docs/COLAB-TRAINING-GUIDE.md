# Training the opcode transformer on Google Colab

The practical walkthrough for the deep-model track. `docs/DEEP-MODEL-GUIDE.md`
says why the model looks the way it does; this says which buttons to press.

**What has and has not been run.** `extract-sequences` is tested
(`core/test/sequences.test.ts`). `training/opcode_transformer.py` has been run
end to end on a CPU against a few dozen sequences: every subcommand completes,
the exported ONNX model reproduces the PyTorch logit to within 1e-7, and
int8 quantisation runs. That
proves the plumbing. It has **not** been run on Colab, and no run on a real
corpus is reported here, so there is no result in this file to quote.

Depends on `docs/DATASET-PLAN.md` having been followed first. A transformer
needs more labelled positives than the linear model, not fewer.

## 1. Locally: extract the sequences

Colab never sees a `.wasm` file. Token sequences are all that leaves the
machine, which matters when half the corpus is live mining payloads that an
antivirus product will (correctly) quarantine in a Drive folder.

```bash
# absolute paths: `npm run -w` executes from core/, so a relative path resolves there
export CORPUS=~/wasm-sentry-corpus
export PRETRAIN_POOL=~/wasm-sentry-pretrain-pool
export OUT=~/wasm-sentry-sequences && mkdir -p "$OUT"

npm run extract-sequences -w @wasm-sentry/core -- "$PRETRAIN_POOL"   --out "$OUT/pretrain.jsonl" --unlabeled
npm run extract-sequences -w @wasm-sentry/core -- "$CORPUS"          --out "$OUT/finetune.jsonl"
npm run extract-sequences -w @wasm-sentry/core -- "$CORPUS/holdout"  --out "$OUT/holdout.jsonl"
```

Each writes a `.jsonl` and a `.jsonl.vocab.json`. Read the summary line: how
many sequences were led by a kernel candidate, how many were padded, and that
nothing was skipped. About 4 KB per module, so ~35 MB for a WasmBench-sized pool.

**Keep the holdout out of the pretraining pool.** Pretraining is unlabelled,
but a model that has already seen the holdout's sequences has still seen them.
If the pool was built from all of WasmBench, remove the holdout hashes first:

```bash
ls "$CORPUS/holdout/benign" "$CORPUS/holdout/malicious" | grep .wasm | while read -r f; do rm -f "$PRETRAIN_POOL/$f"; done
```

## 2. Colab: set up

Runtime → Change runtime type → **T4 GPU**. Upload the six data files and
`training/opcode_transformer.py` with the file browser, or from Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/wasm-sentry      # wherever you put the seven files
!pip -q install onnx onnxruntime
import torch; print(torch.__version__, torch.cuda.is_available())   # expect True
```

## 3. Pretrain (masked opcode modelling)

```python
!python opcode_transformer.py pretrain pretrain.jsonl --out pretrained.pt --epochs 20
```

Watch the **validation** loss, which is computed on 5% of the pool the model
never trains on. Uniform guessing over a 44-token vocabulary is 3.78; a useful
run ends well below the loss of always predicting the commonest opcode. If
validation loss turns upward while training loss keeps falling, the saved
checkpoint is already the best one (only improvements are saved) — stop there.

## 4. Cross-validate the fine-tune

```python
!python opcode_transformer.py crossval finetune.jsonl --pretrained pretrained.pt --out oof.json
!python opcode_transformer.py crossval finetune.jsonl --out oof-scratch.json      # ablation: no pretraining
```

Every sample is scored exactly once, by a model that never trained on it, and
the metrics are computed over those out-of-fold scores. Weights are reloaded
from `pretrained.pt` for every fold; carrying a fine-tuned model across folds
would leak the previous fold's test set into the next one's training.

The second command is not optional. If pretraining does not beat training from
scratch, the pretraining pool bought nothing and the write-up should say so.

Put the `transformer` row next to the two rows `npm run train` prints for the
same `$CORPUS`:

```bash
npm run train -w @wasm-sentry/core -- "$CORPUS"      # prints `classifier` and `heuristics`
```

The folds differ between the two tools; the comparison is still fair because
each reports out-of-fold predictions over the identical set of files.

## 5. The bar, decided before looking

The transformer is integrated only if it **clearly beats both** the heuristics
and the linear classifier — on F1 and on false positives at matched recall, not
on accuracy, which at 100:1 rewards answering "benign" every time. "Close" is a
loss: it costs megabytes and an ONNX runtime where the alternative costs
kilobytes. If it loses, that is the result, and it gets reported as one.

## 6. Only if it cleared the bar: final model, holdout, export

```python
!python opcode_transformer.py finetune finetune.jsonl --pretrained pretrained.pt --out finetuned.pt
!python opcode_transformer.py evaluate holdout.jsonl --model finetuned.pt        # once. not a tuning loop.
!python opcode_transformer.py export --model finetuned.pt --out model.onnx
```

`export` re-runs the ONNX file under onnxruntime and prints its largest
disagreement with PyTorch; anything above ~1e-4 means the export is wrong, not
that the model is noisy. Quantise afterwards and **re-run the holdout on the
quantised file** before believing it is equivalent:

```python
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic("model.onnx", "model.quant.onnx", weight_type=QuantType.QInt8)
```

Skip onnxruntime's suggested pre-processing pass: its symbolic shape inference
fails on this graph, and the warning about it is safe to ignore. Measured on the
plumbing model: 3.6 MB to 1.1 MB, and the logit moved by 0.008 — small, but
that is a number about an untrained model, so measure it again on the real one.

Extension integration is out of scope here, as in the design guide §8.

## 7. What to write down

Corpus composition by family (from `manifest.csv`), near-duplicate clusters
collapsed, the three metric rows on the same files, pretraining vs scratch, the
single holdout number, and the sequence schema version from the `.vocab.json`.
A model is only meaningful alongside the tokeniser version that fed it.
