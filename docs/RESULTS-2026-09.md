# Corpus rebuild and retraining, September 2026

What was measured when the malicious set was expanded from 6 samples to 76 and
the pipeline in `docs/DATASET-PLAN.md` was run end to end. Every number here
describes one corpus assembled on one machine. None of it is a detection rate.

**The decision first: no model ships.** The linear classifier beat the
heuristics on this corpus, and it still does not clear the bar, because after
near-duplicate collapse the corpus holds 14 training positives and the plan's
own floor for running the trainer at all is 25. The holdout was not evaluated.

## 1. Two defects found on the way, both of which invalidate earlier numbers

**The benign set contained miners.** The provenance keyword filter excluded 27
WasmBench binaries, leaving 8,434 "benign". Checked by content, 56 of those are
verified miners, nine of them exporting `_cryptonight_hash` outright. The
previously reported comparison (classifier 135 false alarms against 46 for the
rules, on 8,434 benign and 6 malicious) was scored against those labels, so an
unknown share of its "false alarms" were correct. It should not be quoted.

**The holdout split was biased by the clustering step before it.**
`cluster-corpus` keeps the alphabetically first file per cluster; in a
hash-named corpus that is the smallest hash; `split-holdout` held out hashes
starting `0`, `1` or `2`. Survivors of large clusters therefore went to the
holdout almost every time: 4,921 of 5,538 collapsed files sat behind a held-out
survivor, and the holdout took 40% of the malicious class. It now reads the
hash's last digit. Re-measured: 23%, and a 17% benign holdout.

## 2. The corpus

Sources: WasmBench's filtered release (8,461 binaries), seven payloads
committed to open-source miner repositories, and two modules carved out of
base64 in live mining pages. `manifest.csv` carries a one-sentence
justification per malicious sample; labels rest on contents (export tables,
strings), not on where a file was found.

| Family | Verified | Source |
|---|---|---|
| Equihash, `jazecminer` | 30 | SEISMIC reproduction: one miner and its diversified variants |
| CryptoNight, API renamed to a random string | 13 | MinerRay crawl; SEISMIC's antivirus-evasion sample |
| CryptoNight, plain exports | 12 | MinerRay, SEISMIC, deepMiner, CryptoNoter, pool miners |
| CryptoNight, export names minified | 12 | SEISMIC reproduction, an open-source miner, a live mining page |
| Argon2d, Nimiq | 2 | crawled from nimiq.watch and cdn.nimiq.com (dual-use) |
| RandomX | 2 | a mining proxy's kernel; a "RandomX farm" page |
| Equihash `hushminer`, obfuscated CryptoNight, `webminer_v1`, minero.cc, yespower `power2b` | 1 each | SEISMIC, a crawl, a web-miner repository |
| **Total** | **76** | |

Thirty candidates were left out of **both** classes: SEISMIC's instrumented
(`*profiled*`) copies, stripped modules with no content evidence either way,
two ambiguous wallet-or-miner modules, and one file Windows Defender
quarantined as `Trojan:Win32/CoinMiner` on extraction.

| Stage | Benign | Malicious |
|---|---|---|
| Assembled | 8,361 | 76 |
| `filter-corpus` (649 too small, 118 low coverage, 65 too large) | 7,529 | 76 |
| `dedupe-corpus` | 7,529 | 76 |
| `cluster-corpus` at cosine 0.999 (731 clusters) | 2,047 | 20 |
| `split-holdout` | 1,694 train / 353 held out | 14 train / 6 held out |
| + npm harvest (63 new modules, 22 of them carved from `hash-wasm`), pipeline re-run | 1,721 train / 356 held out | 14 train / 6 held out |

Seventy-six samples are twenty programs. That is the honest size of the
malicious class. What the variety bought is four proof-of-work algorithms in
training (CryptoNight, Equihash, RandomX, yespower) where there was one.

## 3. Linear classifier against the heuristics, same folds

5-fold, `npm run train`, before and after the npm hard negatives
(`hash-wasm`'s Argon2, scrypt and friends, `argon2-browser`, libsodium) were added:

| | Benign | Precision | Recall | F1 | AUC | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|
| classifier | 1,694 | 0.302 | 0.929 | 0.456 | 0.995 | 13 | 30 | 1 |
| heuristics | 1,694 | 0.167 | 0.214 | 0.188 | 0.716 | 3 | 15 | 11 |
| classifier, with hard negatives | 1,721 | 0.250 | 0.786 | 0.379 | 0.989 | 11 | 33 | 3 |
| heuristics, with hard negatives | 1,721 | 0.150 | 0.214 | 0.176 | 0.714 | 3 | 17 | 11 |

Twenty-seven memory-hard hashing libraries cost the classifier two detections
and three more false alarms. The plan said this is the test that matters, and
the model does not pass it cleanly. The second pair of rows is the result.

Read with the base rate in mind: 33 false alarms is 1.9% of benign modules, and
at that rate nearly every alert a user saw would be wrong. Strongest weights
were `log_memoryInitialPages`, `log_kernel_loopSize`, `memory_bounded`,
`op_i32.shl`, `op_i64.load`; `stripped` carried −0.30. Memory shape is a real
CryptoNight signal (the 2 MB scratchpad) and also exactly the kind of
collection artifact §4.2 of the plan warns about. With 14 positives the two
cannot be told apart.

## 4. What this says about the rules

The more useful result is about the heuristics, not the model. Scored over all
76 verified samples before clustering:

| Family | Flagged medium or above | Score range |
|---|---|---|
| CryptoNight, all four variants | 38 / 38 | 39–80 |
| Argon2d (Nimiq) | 2 / 2 | 30 |
| RandomX | 1 / 2 | 14–31 |
| Equihash (`jazecminer`, `hushminer`) | 0 / 31 | 5–10 |
| `webminer_v1`, minero.cc, yespower | 0 / 3 | 6 |

The rules were calibrated on CryptoNight and they find CryptoNight, including
the renamed and minified builds. They score every Equihash miner as benign.
Equihash is memory-bound sorting and collision search, not a tight bitwise
loop, so `hash-loop-density` has nothing to fire on. That is a rule gap a
corpus this size is big enough to demonstrate, and fixing it does not need a
model.

## 5. Transformer track

Not run to completion, so there is no transformer result. Sequences were
extracted for real (1,708 fine-tuning, 359 holdout, 5,966 pretraining, with
the holdout and the near-duplicates of held-out samples kept out of the pool).
A CPU pretraining run reached two epochs, validation loss 2.05 then 1.81
against 3.78 for uniform guessing, before the machine ran short of memory and
the job was stopped. No fine-tuning or cross-validation ran. With 14 training
positives the comparison would not have supported a conclusion either way; the
commands in `docs/COLAB-TRAINING-GUIDE.md` run it on a GPU in minutes.

## 6. What is still missing

More distinct malicious *programs*, not more files: 25 training positives is
the floor, 60 is where numbers start to mean something, and there are 14. The
routes not yet tried are the ones in the plan that need a human — asking the
MINOS, MineSweeper and Outguard authors for their sample lists, and compiling
reference miners at several optimisation levels. Runtime thresholds remain
uncalibrated; that needs the extension running against live samples in a real
browser, which a headless session cannot do.
