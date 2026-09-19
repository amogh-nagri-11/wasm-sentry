"""
Deep-model track: pretrain, cross-validate, fine-tune and export the
opcode-sequence transformer described in docs/DEEP-MODEL-GUIDE.md.

One file on purpose: it is uploaded to Colab as-is (docs/COLAB-TRAINING-GUIDE.md)
and the whole architecture stays readable in one place. Depends on torch and
numpy only; `export` additionally needs onnx.

Input is what `npm run extract-sequences` writes: a JSONL of
{"sha256", "tokens", "label"} rows and a sibling <file>.vocab.json. The
vocabulary size and sequence length are read from that file, never assumed.

    python opcode_transformer.py pretrain  pretrain.jsonl --out pretrained.pt
    python opcode_transformer.py crossval  finetune.jsonl --pretrained pretrained.pt --out oof.json
    python opcode_transformer.py finetune  finetune.jsonl --pretrained pretrained.pt --out finetuned.pt
    python opcode_transformer.py evaluate  holdout.jsonl  --model finetuned.pt
    python opcode_transformer.py export    --model finetuned.pt --out model.onnx

Nothing here claims a detection rate. Every number it prints describes the
corpus it was given and nothing else.
"""
import argparse
import json
import math
import random
import sys

import numpy as np
import torch
import torch.nn as nn

PAD, MASK, CLS = 0, 1, 2          # fixed by core/src/ml/sequences.ts
FIRST_MASKABLE = 3                # PAD, MASK and CLS are never masked or predicted


# --------------------------------------------------------------------------- #
# Data                                                                        #
# --------------------------------------------------------------------------- #

def load_vocab(jsonl_path):
    with open(jsonl_path + ".vocab.json", encoding="utf8") as fh:
        vocab = json.load(fh)
    names = vocab["vocabulary"]
    assert names[PAD] == "PAD" and names[MASK] == "MASK" and names[CLS] == "CLS", \
        "token table does not match this script; re-run extract-sequences"
    return {"size": len(names), "seq_len": vocab["sequenceLength"], "schema": vocab["schemaVersion"]}


def load_rows(jsonl_path, require_labels):
    rows, seen = [], set()
    with open(jsonl_path, encoding="utf8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["sha256"] in seen:       # the extractor already de-duplicates; this is a guard
                continue
            seen.add(row["sha256"])
            if require_labels and row["label"] not in (0, 1):
                raise SystemExit(f"{jsonl_path}: row {row['sha256'][:12]} has no label")
            rows.append(row)
    if not rows:
        raise SystemExit(f"{jsonl_path}: no rows")
    return rows


def tensors(rows):
    tokens = torch.tensor([r["tokens"] for r in rows], dtype=torch.long)
    return tokens, (tokens != PAD)


# --------------------------------------------------------------------------- #
# Model                                                                       #
# --------------------------------------------------------------------------- #

class OpcodeTransformer(nn.Module):
    def __init__(self, vocab_size, seq_len, dim=128, layers=4, heads=4, ff=512, dropout=0.1):
        super().__init__()
        self.config = dict(vocab_size=vocab_size, seq_len=seq_len, dim=dim, layers=layers,
                           heads=heads, ff=ff, dropout=dropout)
        self.token_emb = nn.Embedding(vocab_size, dim, padding_idx=PAD)
        self.pos_emb = nn.Embedding(seq_len, dim)
        layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=ff,
                                           dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.mlm_head = nn.Linear(dim, vocab_size)
        self.cls_head = nn.Linear(dim, 1)

    def encode(self, tokens, attention_mask):
        positions = torch.arange(tokens.size(1), device=tokens.device)
        x = self.token_emb(tokens) + self.pos_emb(positions)
        x = self.encoder(x, src_key_padding_mask=~attention_mask.bool())
        return self.norm(x)

    def forward_mlm(self, tokens, attention_mask):
        return self.mlm_head(self.encode(tokens, attention_mask))

    def forward(self, tokens, attention_mask):
        """Classification logit from the CLS position. This is what gets exported."""
        return self.cls_head(self.encode(tokens, attention_mask)[:, 0, :]).squeeze(-1)


def save(model, path, extra=None):
    torch.save({"config": model.config, "state": model.state_dict(), **(extra or {})}, path)


def load(path, device):
    blob = torch.load(path, map_location=device)
    model = OpcodeTransformer(**blob["config"]).to(device)
    model.load_state_dict(blob["state"])
    return model, blob


def new_model(vocab, device, pretrained=None):
    if pretrained:
        model, _ = load(pretrained, device)
        if model.config["vocab_size"] != vocab["size"] or model.config["seq_len"] != vocab["seq_len"]:
            raise SystemExit("pretrained model was built for a different vocabulary or window")
        return model
    return OpcodeTransformer(vocab["size"], vocab["seq_len"]).to(device)


# --------------------------------------------------------------------------- #
# Pretraining: masked opcode modelling                                        #
# --------------------------------------------------------------------------- #

def mask_batch(tokens, vocab_size, generator):
    """BERT's 80/10/10 rule. Returns (corrupted tokens, targets with -100 where ignored)."""
    maskable = tokens >= FIRST_MASKABLE
    chosen = (torch.rand(tokens.shape, generator=generator) < 0.15) & maskable
    targets = torch.where(chosen, tokens, torch.full_like(tokens, -100))
    roll = torch.rand(tokens.shape, generator=generator)
    corrupted = tokens.clone()
    corrupted[chosen & (roll < 0.8)] = MASK
    swap = chosen & (roll >= 0.8) & (roll < 0.9)
    corrupted[swap] = torch.randint(FIRST_MASKABLE, vocab_size, tokens.shape, generator=generator)[swap]
    return corrupted, targets


def pretrain(args, device):
    vocab = load_vocab(args.data)
    tokens, attention = tensors(load_rows(args.data, require_labels=False))
    generator = torch.Generator().manual_seed(args.seed)

    # A slice is held back so the loss printed is about sequences the model has
    # not seen. A falling training loss alone cannot tell learning from memorising.
    order = torch.randperm(len(tokens), generator=generator)
    n_val = max(1, len(tokens) // 20)
    val_idx, train_idx = order[:n_val], order[n_val:]
    if len(train_idx) == 0:
        raise SystemExit("pretraining pool is too small to split")

    model = new_model(vocab, device)
    print(f"{sum(p.numel() for p in model.parameters()):,} parameters, "
          f"{len(train_idx)} train / {len(val_idx)} validation sequences, device {device}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * math.ceil(len(train_idx) / args.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def run(indices, train):
        model.train(train)
        total, batches = 0.0, 0
        for start in range(0, len(indices), args.batch_size):
            batch = indices[start:start + args.batch_size]
            corrupted, targets = mask_batch(tokens[batch], vocab["size"], generator)
            if (targets != -100).sum() == 0:
                continue
            with torch.set_grad_enabled(train):
                logits = model.forward_mlm(corrupted.to(device), attention[batch].to(device))
                loss = loss_fn(logits.view(-1, logits.size(-1)), targets.to(device).view(-1))
            if train:
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
            total, batches = total + loss.item(), batches + 1
        return total / max(batches, 1)

    best = float("inf")
    for epoch in range(args.epochs):
        shuffled = train_idx[torch.randperm(len(train_idx), generator=generator)]
        train_loss, val_loss = run(shuffled, True), run(val_idx, False)
        note = ""
        if val_loss < best:
            best = val_loss
            save(model, args.out, {"vocab": vocab, "stage": "pretrained"})
            note = "  saved"
        print(f"epoch {epoch + 1:3d}  train {train_loss:.4f}  val {val_loss:.4f}{note}")
    print(f"\nBest validation loss {best:.4f} (uniform guessing would be "
          f"{math.log(vocab['size']):.4f}). Wrote {args.out}")


# --------------------------------------------------------------------------- #
# Fine-tuning and evaluation                                                  #
# --------------------------------------------------------------------------- #

def fit_classifier(model, tokens, attention, labels, args, device, seed):
    generator = torch.Generator().manual_seed(seed)
    positives = labels.sum().item()
    if positives == 0 or positives == len(labels):
        raise SystemExit("a training split contains one class only; too few samples for this many folds")
    # Same intent as balanceClasses in core/src/ml/train.ts.
    pos_weight = torch.tensor((len(labels) - positives) / positives, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    for _ in range(args.epochs):
        model.train()
        order = torch.randperm(len(tokens), generator=generator)
        for start in range(0, len(order), args.batch_size):
            batch = order[start:start + args.batch_size]
            loss = loss_fn(model(tokens[batch].to(device), attention[batch].to(device)),
                           labels[batch].float().to(device))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return model


@torch.no_grad()
def score(model, tokens, attention, device, batch_size=64):
    model.eval()
    out = []
    for start in range(0, len(tokens), batch_size):
        logits = model(tokens[start:start + batch_size].to(device),
                       attention[start:start + batch_size].to(device))
        out.append(torch.sigmoid(logits).cpu())
    return torch.cat(out).numpy()


def roc_auc(labels, scores):
    """Rank-based AUC, tied scores sharing a rank."""
    labels, scores = np.asarray(labels), np.asarray(scores)
    pos, neg = int((labels == 1).sum()), int((labels == 0).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[labels == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def report(name, labels, scores, threshold=0.5):
    labels = np.asarray(labels)
    predicted = np.asarray(scores) >= threshold
    tp = int((predicted & (labels == 1)).sum())
    fp = int((predicted & (labels == 0)).sum())
    tn = int((~predicted & (labels == 0)).sum())
    fn = int((~predicted & (labels == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    auc = roc_auc(labels, scores)
    print(f"  {name:<12} acc={(tp + tn) / len(labels):.3f} prec={precision:.3f} rec={recall:.3f} "
          f"f1={f1:.3f} auc={auc:.3f} (tp={tp} fp={fp} tn={tn} fn={fn})")
    return {"precision": precision, "recall": recall, "f1": f1, "auc": auc,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def stratified_folds(labels, k, seed):
    rng = random.Random(seed)
    folds = [[] for _ in range(k)]
    for cls in (0, 1):
        members = [i for i, label in enumerate(labels) if label == cls]
        rng.shuffle(members)
        for position, index in enumerate(members):
            folds[position % k].append(index)
    return folds


def crossval(args, device):
    vocab = load_vocab(args.data)
    rows = load_rows(args.data, require_labels=True)
    tokens, attention = tensors(rows)
    labels = torch.tensor([r["label"] for r in rows])
    positives = int(labels.sum())
    print(f"{len(rows)} sequences: {len(rows) - positives} benign, {positives} malicious, device {device}")
    k = min(args.folds, positives, len(rows) - positives)
    if k < 2:
        raise SystemExit("need at least two samples of each class to cross-validate")
    if k < args.folds:
        print(f"  using {k} folds so every fold tests at least one sample of each class")
    if positives < 25:
        print("  WARNING: this few positives cannot support a conclusion. The numbers below are\n"
              "  facts about these files, not a detection rate.")

    oof = np.zeros(len(rows))
    for number, test in enumerate(stratified_folds(labels.tolist(), k, args.seed)):
        held = set(test)
        test = torch.tensor(test)
        train = torch.tensor([i for i in range(len(rows)) if i not in held])
        model = new_model(vocab, device, args.pretrained)     # fresh weights every fold
        fit_classifier(model, tokens[train], attention[train], labels[train], args, device, args.seed + number)
        oof[test.numpy()] = score(model, tokens[test], attention[test], device)
        print(f"  fold {number + 1}/{k} done")

    print(f"\n{k}-fold out-of-fold, {len(rows)} sequences:")
    metrics = report("transformer", labels.numpy(), oof)
    if args.out:
        with open(args.out, "w", encoding="utf8") as fh:
            json.dump({"folds": k, "pretrained": bool(args.pretrained), "metrics": metrics,
                       "predictions": [{"sha256": r["sha256"], "label": r["label"], "score": float(s)}
                                       for r, s in zip(rows, oof)]}, fh, indent=1)
        print(f"Wrote {args.out}")
    print("\nCompare against BOTH rows `npm run train` prints for the same corpus (classifier and\n"
          "heuristics). Integrate only if this clearly beats both; otherwise report that it did not.")


def finetune(args, device):
    vocab = load_vocab(args.data)
    rows = load_rows(args.data, require_labels=True)
    tokens, attention = tensors(rows)
    labels = torch.tensor([r["label"] for r in rows])
    model = new_model(vocab, device, args.pretrained)
    fit_classifier(model, tokens, attention, labels, args, device, args.seed)
    save(model, args.out, {"vocab": vocab, "stage": "finetuned", "trained_on": len(rows)})
    print(f"Trained on all {len(rows)} sequences. Wrote {args.out}")
    print("These are training-set numbers and are not an evaluation:")
    report("transformer", labels.numpy(), score(model, tokens, attention, device))


def evaluate(args, device):
    rows = load_rows(args.data, require_labels=True)
    tokens, attention = tensors(rows)
    model, _ = load(args.model, device)
    print(f"{len(rows)} held-out sequences:")
    report("transformer", [r["label"] for r in rows], score(model, tokens, attention, device))


def export(args, device):
    model, _ = load(args.model, "cpu")
    model.eval()
    seq_len = model.config["seq_len"]
    tokens = torch.full((1, seq_len), PAD, dtype=torch.long)
    tokens[0, :8] = torch.tensor([CLS, 6, 7, 8, 9, 10, 11, 12])
    mask = tokens != PAD
    torch.onnx.export(model, (tokens, mask), args.out, input_names=["tokens", "attention_mask"],
                      output_names=["logit"], opset_version=17,
                      dynamo=False,   # the tracing exporter; the dynamo one fails on a cp1252 console
                      dynamic_axes={"tokens": {0: "batch"}, "attention_mask": {0: "batch"},
                                    "logit": {0: "batch"}})
    print(f"Wrote {args.out}")
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed; the exported model was not checked against PyTorch")
        return
    session = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    got = session.run(None, {"tokens": tokens.numpy(), "attention_mask": mask.numpy()})[0]
    want = model(tokens, mask).detach().numpy()
    print(f"ONNX vs PyTorch logit difference: {float(np.abs(got - want).max()):.2e}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    defaults = {"pretrain": (20, 1e-3), "crossval": (10, 5e-5), "finetune": (10, 5e-5)}
    for name, (epochs, lr) in defaults.items():
        p = sub.add_parser(name)
        p.add_argument("data")
        p.add_argument("--out", required=name != "crossval")
        p.add_argument("--epochs", type=int, default=epochs)
        p.add_argument("--lr", type=float, default=lr)
        p.add_argument("--batch-size", type=int, default=32)
        p.add_argument("--seed", type=int, default=0)
        if name != "pretrain":
            p.add_argument("--pretrained")
        if name == "crossval":
            p.add_argument("--folds", type=int, default=5)
    p = sub.add_parser("evaluate")
    p.add_argument("data")
    p.add_argument("--model", required=True)
    p = sub.add_parser("export")
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)

    args = parser.parse_args()
    torch.manual_seed(getattr(args, "seed", 0))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    {"pretrain": pretrain, "crossval": crossval, "finetune": finetune,
     "evaluate": evaluate, "export": export}[args.command](args, device)


if __name__ == "__main__":
    sys.exit(main())
