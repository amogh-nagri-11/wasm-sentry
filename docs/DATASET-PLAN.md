# Dataset acquisition & preprocessing plan

Actionable version of `CORPUS-GUIDE.md` (which stays as working notes, kept
outside the repo). This file answers two questions decisively: **what to go
get**, and **exactly how to turn it into a trainable corpus**. Written against
the code as it stands: feature schema v1 (`core/src/ml/features.ts`,
`FEATURE_COUNT` columns), `analyzeWasm` as the parse path, `crossValidate` as
the reporting rule, trainer at `core/scripts/train-model.ts`.

Target sizes (from the trainer's own floor and the point where numbers stop
being noise):

| | Minimum to run | Enough to report |
|---|---|---|
| benign | 25 | 500+ |
| malicious | 25 | 60+ |

The malicious side is the real constraint — see below.

### Where things live

Every command below assumes:

```bash
export CORPUS=~/wasm-sentry-corpus     # OUTSIDE the repo -- pick any path you like, just export it once per shell
mkdir -p "$CORPUS/benign" "$CORPUS/malicious"
```

- **`$CORPUS`** — the actual corpus. Always referenced as `"$CORPUS"`, never as a
  bare `corpus/`, so it doesn't matter what your shell's `cwd` is when you run
  a command.
- **`/tmp/wasmbench`, `/tmp/npm-wasm`** — throwaway download/extraction scratch
  space. Nothing here is the corpus; every harvesting step below ends by
  copying the files you want into `$CORPUS/benign` or `$CORPUS/malicious`.
- **The `npm run <script> -w @wasm-sentry/core -- ...` commands** (filter-corpus,
  dedupe-corpus, cluster-corpus, split-holdout, inspect, calibrate, train) must
  be run from the **wasm-sentry project root** — that's where the root
  `package.json`'s `workspaces` field lives, which `-w` resolves against. Where
  you are does not matter for `$CORPUS` itself, since it's an absolute path.

---

## 1. What to get — benign

Go get these four things, in this order. Stop once you clear ~500 modules
spanning at least 4 toolchains; more volume from a single source (e.g. only
WasmBench) doesn't help, diversity of origin does.

1. **WasmBench** (`github.com/sola-st/WasmBench`) — 8,461 unique real-world
   binaries. Download and extract the filtered release (already deduplicated
   by the authors; use `all-binaries-metadata.7z` instead only if you
   specifically want the pre-dedup set):
   ```bash
   mkdir -p /tmp/wasmbench && cd /tmp/wasmbench
   curl -LO https://github.com/sola-st/WasmBench/releases/download/v1.0/filtered-binaries-metadata.7z
   7z x filtered-binaries-metadata.7z   # brew install p7zip if 7z is missing
   # extracts to: filtered/ (the binaries), filtered.pretty.json, filtered.list.txt
   ls filtered | head -3   # confirm whether files already carry a .wasm suffix before the next step
   ```
   **Do not use it unfiltered** — the paper's own crawl includes cryptominers,
   and (checked directly against the live metadata) **there is no mining/
   category field** to filter on — it's per-binary provenance only
   (`files[].repository`, `files[].absolute_path`, `files[].collection_method`,
   `size_bytes`, `producers.language`, etc.), not a classification. Filter
   with both of these instead:
   ```bash
   # 1. exclude anything whose provenance string names a known miner/pool
   jq -r 'to_entries[] | select(.value.files[]? |
     (.repository // "") + " " + (.absolute_path // "") |
     test("coinhive|coin-hive|cryptonight|crypto-?loot|coinimp|webminepool|jsecoin|monerise|xmr|deepminer|miner|monero|mining|minero|randomx|yespower|nimiq|webchain"; "i")
   ) | .key' filtered.pretty.json > /tmp/wasmbench-exclude-hashes.txt
   grep -vFf /tmp/wasmbench-exclude-hashes.txt filtered.list.txt > /tmp/wasmbench-keep-hashes.txt
   ```
   **The keyword list above was widened on 2026-09-20, and the earlier one
   was measurably wrong.** The first version named products (`coinhive`,
   `deepminer`, ...) and excluded 27 binaries. Checked against content, it
   left **56 verified miners in `benign/`**: thirty builds of the `jazecminer`
   Equihash miner and a `hushminer` from the SEISMIC dataset, twelve MinerRay
   crawl samples whose CryptoNight API is renamed to a random string, two Nimiq
   Argon2d miners, a `minero.cc` payload -- and nine modules that export
   `_cryptonight_hash` under a filename that says nothing. Any number measured
   on a corpus filtered that way (8,434 benign) counted those as false
   positives when the detector was right. The wide list over-matches on purpose
   (EOS contracts called `*miner`, `hoffmann-mineral.com`); what it catches is
   sorted by content, not thrown away -- see §2, step 0.

   ```bash
   # 2. copy survivors into your corpus, forcing a .wasm suffix (the trainer
   #    and the corpus scripts below only pick up files ending in .wasm)
   mkdir -p "$CORPUS/benign"
   while read -r h; do
     f="filtered/$h"; [ -f "$f" ] || f="filtered/$h.wasm"
     [ -f "$f" ] && cp "$f" "$CORPUS/benign/$h.wasm"
   done < /tmp/wasmbench-keep-hashes.txt
   ```
   ```bash
   # 3. run the project's own heuristics over what you copied and hand-check anything it flags
   npm run inspect -w @wasm-sentry/core -- $CORPUS/benign/*.wasm | grep -B1 "risk:.*\(medium\|high\)"
   ```
   Anything caught by either check gets excluded or moved to `malicious/`
   with a one-sentence justification (§3.6) after manual review — never
   trusted into `benign/` unreviewed.

2. **These specific npm packages**, pulled with `npm pack` (downloads without
   installing):
   ```bash
   mkdir -p /tmp/npm-wasm && cd /tmp/npm-wasm
   npm pack @sqlite.org/sqlite-wasm esbuild-wasm @ffmpeg/core \
     @jsquash/avif @jsquash/webp @jsquash/jxl @resvg/resvg-wasm \
     @swc/wasm-web wasm-brotli @bokuweb/zstd-wasm argon2-browser \
     hash-wasm blake3-wasm libsodium-wrappers opencv-wasm \
     tesseract.js-core duckdb-wasm pyodide php-wasm
   for t in *.tgz; do tar -xzf "$t" -C .; done
   find . -name '*.wasm' -size +1k
   ```
   `hash-wasm` and `argon2-browser` are not optional — they're memory-hard
   KDFs (Argon2, scrypt) that look like mining kernels to a static detector.
   If the model can't keep these out of the malicious class, it hasn't
   learned anything the heuristics didn't already know.

3. **Your own captures**: export real modules seen by the extension itself
   from the dashboard. Highest-fidelity benign source that exists — it's
   drawn from the exact distribution the tool runs against in production.

4. **A handful of self-compiled modules** to fill toolchain/optimization-level
   gaps: `rustc --target wasm32-unknown-unknown -O` over a few crates, `emcc`
   at `-O0/-O2/-O3`, one TinyGo build. Do this only if step 1–3 leave a
   toolchain underrepresented — don't manufacture volume here.

Skip (not worth the effort for this project): wapm/WASI registries, generic
GitHub code search. They add population diversity you don't need once 1–3
are done.

## 2. What to get — malicious

This is the actual blocker, not a code problem. CoinHive shut down in March
2019 and was the overwhelming majority of in-browser cryptojacking — the
population is historical and family-concentrated. State this explicitly in
any write-up: you will not get thousands of samples, and a model trained on
survivors generalizes to those families, not to "cryptojacking" in general.

Go get these things, in this order:

0. **What the WasmBench keyword filter catches (§1, step 1) -- you already
   have it.** WasmBench swallowed two cryptojacking research datasets whole:
   SEISMIC (`wenhao1006/SEISMIC`, `Jacarte/SEISMIC_reproduction`) and
   MinerRay's crawl samples (`miner-ray/miner-ray.github.io`). Between them:
   CryptoNight in plain, name-randomised, export-minified and obfuscated
   builds; two Equihash miners; Nimiq's Argon2d search; a RandomX-era
   `minero.cc` payload. That is the family variety route 1 alone never had.
   **Provenance is a lead, not a label.** MinerRay's samples are "WebAssembly
   found while crawling" and are mostly *not* miners (Unity games,
   post-quantum crypto libraries, a barcode reader). Label each file from its
   contents:

   ```bash
   npm run label-evidence -w @wasm-sentry/core -- /path/to/candidates > evidence.jsonl
   ```

   prints exports, imports, mining-related strings and the static analysis per
   file. `_cryptonight_hash`, `_hash_cn`, `_loot_cn`, a `_<random>_create` /
   `_destroy` / `_hash` triple, `mine` beside Zcash strings, or
   `_nimiq_argon2_target` is a justification. A stripped module with minified
   exports and no strings is not: leave it out of **both** classes. Leave out
   the `*profiled*` files too (and anything exporting `_getI32AddCountLo`):
   those are SEISMIC's instrumented copies, research artifacts rather than
   payloads. Files the filter caught by accident go back into `benign/`, and
   the crypto libraries among them are exactly the hard negatives worth having.

   Expect antivirus to quarantine some of these on extraction (Windows Defender
   took one as `Trojan:Win32/CoinMiner`). That is an independent confirmation
   of the label and a sample you do not get; do not weaken the machine's
   protection to keep it.

   GitHub's code search does not index binaries, so searching for `.wasm`
   miners finds almost nothing; listing the git trees of miner repositories
   found ~10 distinct payloads, and carving `AGFzbQ...` base64 out of miner
   pages found two more (one RandomX). Low yield, but it is where the
   non-CryptoNight families outside WasmBench came from.


1. **Open-source browser-miner repos with the compiled `.wasm` committed to
   git.** Real mining kernels, no malware handling involved. Search GitHub
   for: `cryptonight wasm`, `coinhive`, `webminer wasm`, `monero miner
   javascript`, filtered to repos containing a `.wasm` file. Specifically
   look for: **deepMiner**, **webminepool**, **CoinIMP**, **Crypto-Loot**,
   **JSECoin**, **Monerise**, and CoinHive mirror/clone repos (several
   archival forks of `coinhive.min.js` + `cryptonight.wasm` survive). This
   route alone plausibly clears the trainer's floor (30–80 defensible
   positives).

2. **Self-compiled CryptoNight reference implementation** — `monero-project/
   monero` or a standalone `cryptonight` C library, built under Emscripten
   at a few optimization levels. A kernel you compiled yourself has a label
   you can defend absolutely, and gives you legitimate variants (grouped
   into one fold each — see §3.3 below — not treated as independent samples).

3. **Email the authors of MINOS (NDSS 2021)** for their labelled Wasm corpus
   or sample list — this is the single most direct source of already-curated
   cryptojacking Wasm and academic artifact requests for defensive tooling
   are routinely answered. If that stalls, the same ask to MineSweeper (CCS
   2018) and Outguard is worth sending in parallel.

Don't chase MalwareBazaar/VirusTotal/Common Crawl as primary sources — they're
real but low-yield relative to effort for this project (VirusTotal binary
download needs a paid/academic key; MalwareBazaar's Wasm-tagged samples are
sparse). Use them only later, to *verify* labels on what you already have via
VirusTotal lookup (free tier supports lookup, not download).

**What does not count as malicious data:** `testbed/*.wasm` fixtures (built
to fire the existing rules — using them grades the model against the answer
key) and anything labeled malicious only by where it was found. If you can't
write one sentence in the manifest justifying the label, it doesn't go in.

---

## 3. Preprocessing procedure

Four scripts were added under `core/scripts/` specifically for this (typecheck
and smoke-tested against the testbed fixtures — see each file's header
comment for exact behavior). Run them in this exact order — each one assumes
the previous has already run:

```bash
npm run filter-corpus  -w @wasm-sentry/core -- "$CORPUS"    # 3.1 + 3.4: parseable, high-coverage, right-sized
npm run dedupe-corpus  -w @wasm-sentry/core -- "$CORPUS"    # 3.2: exact SHA-256 dedup, within and across classes
npm run cluster-corpus -w @wasm-sentry/core -- "$CORPUS"    # 3.3: near-duplicate collapse by feature-vector similarity
npm run split-holdout  -w @wasm-sentry/core -- "$CORPUS"    # 3.7: 20% held out, never trained on
```

Nothing is deleted by any of them — rejects, duplicates, and collapsed
near-duplicates are moved into sibling directories (`benign-rejected/`,
`benign-duplicates/`, `benign-near-duplicates/cluster-N/`, `holdout/benign/`,
same for `malicious`) so every decision is inspectable and reversible.

### 3.1 + 3.4 — Filter to real, parseable, right-sized Wasm

`filter-corpus.ts` runs three gates per file, in order, and moves the first
failure into `<class>-rejected/<reason>__<filename>`:

1. **Magic bytes** (`isWasm` from `core/src/sniff.ts`) — catches `.wasm`-named
   files that are actually gzip, an HTML error page, or a Git LFS pointer (a
   real hazard harvesting from a GitHub clone: a tree of 130-byte text files
   with `.wasm` names).
2. **Size floor/ceiling** — defaults 1 KB / 20 MB, override with
   `--min-bytes` / `--max-bytes`.
3. **Parses via `analyzeWasm`, and coverage ≥ 0.9** (`decoded / (decoded +
   skipped + truncated)`), override with `--min-coverage`. A module that
   "parsed" at 40% coverage produces a feature vector describing 40% of a
   program.

Read the printed rejection counts by reason before moving on — that's your
§3.1/3.4 audit trail, no separate step needed.

### 3.2 — Deduplicate by SHA-256 (non-negotiable)

The same module is routinely served under many cache-busted URLs. If the same
bytes end up in both a training and a test fold, `crossValidate` is reporting
memorization, not detection. `dedupe-corpus.ts` hashes every file
(`core/src/hash.ts`, the same code path the extension uses), keeps the
alphabetically-first file per hash, moves the rest to `<class>-duplicates/`.

**Cross-class hashes are never auto-resolved.** A hash appearing in both
`benign/` and `malicious/` is a labelling contradiction — the script prints
both paths and exits 1 without moving anything. Delete or relabel one by hand,
then re-run.

### 3.3 — Near-duplicate control

Exact-hash dedup isn't enough. CryptoNight-family builds often differ only by
a compiler flag, version string, or embedded pool URL while being the same
program underneath. Twenty near-copies in a 60-sample malicious class means
5-fold CV trains on 16 near-copies of what it then tests on — F1 hits 0.98
and means nothing. `cluster-corpus.ts` vectorizes every file with `vectorise`
(`core/src/ml/features.ts`), unions any pair with cosine similarity ≥ 0.999
(override with `--threshold`), keeps one representative per cluster, moves
the rest to `<class>-near-duplicates/cluster-N/`.

If you compiled variants yourself (§2, step 2), that's the correct outcome —
the script's own note tells you to open each cluster and check the
auto-picked representative is a reasonable one before trusting it, since
"structurally similar" and "the same source" aren't always the same claim.

### 3.5 — Class balance

Do **not** downsample benign to match malicious:

1. The trainer already balances — `balanceClasses` defaults on, weighting
   positives by `n / (2 · positives)` (`core/src/ml/train.ts:103`).
2. The real world is overwhelmingly benign; precision at a realistic base
   rate is the number that matters. An artificially 50/50 corpus reports a
   precision you will never see in production.

A 20:1–30:1 benign:malicious ratio is realistic and fine. Read precision,
recall, and F1 — never accuracy (at 30:1, always answering "benign" scores
0.968 accuracy on its own).

Keep enough positives that no 5-fold split leaves a fold's training set
single-class — 25+ positives is comfortable; `crossValidate` throws otherwise.

### 3.6 — Write the manifest as you go

One row per file, written during collection, not reconstructed afterward:

```csv
sha256,path,label,source,why,collected_at
9f86d0…,benign/sqlite3.wasm,0,npm:@sqlite.org/sqlite-wasm@3.46,"official SQLite build",2026-08-24
a1b2c3…,malicious/cryptonight-v1.wasm,1,github:<repo>@<commit>,"committed miner payload, CryptoNight",2026-08-24
```

The `why` column is the actual discipline — if you can't fill it, the sample
doesn't go in. It's also what goes into the trained model's
`metadata.corpus` field.

### 3.7 Hold out a final test set, before you tune anything

`crossValidate` is for model selection only. If you tune the threshold, L2,
or epochs against CV output, CV stops being an unbiased estimate. Run this
last, after 3.1–3.3 — `split-holdout.ts` moves any file whose SHA-256 **ends**
in `0`, `1`, or `2` (≈18.75% of the corpus, reproducible, stable as the
corpus grows) from `$CORPUS/<class>/` into `$CORPUS/holdout/<class>/`; override
the digits with `--digits`. (It used to read the leading digit. Clustering
keeps the alphabetically first member, which in a hash-named corpus is the
smallest hash, so nearly every large cluster's survivor was being held out;
the script's header has the measurement.) Because it runs after clustering, an entire
near-duplicate cluster naturally lands on one side or the other — it can't be
split across the boundary.

Never train against `$CORPUS/holdout/`. Point `npm run train` at it exactly
once, at the very end, to report the final number.

---

## 4. Running it

### 4.1 Verify before training

```bash
npm run inspect -w @wasm-sentry/core -- $CORPUS/benign/*.wasm    | tee benign-inspect.txt
npm run inspect -w @wasm-sentry/core -- $CORPUS/malicious/*.wasm | tee malicious-inspect.txt
grep -c FAILED benign-inspect.txt malicious-inspect.txt
```

`inspect` prints `engineValid`, size, parse time, function/instruction
counts, loops, bitwise/float ratios, memory pages, shared-memory flag,
warnings, and the heuristic verdict per file. If the existing rules already
flag most of the malicious set at high risk, the model has little room to add
value — that's a finding to report, not a problem to fix.

```bash
npm run calibrate -w @wasm-sentry/core -- $CORPUS/malicious/*.wasm
```

Sanity-checks that the positives actually contain hashing kernels.

### 4.2 Evaluate before saving anything

```bash
npm run train -w @wasm-sentry/core -- "$CORPUS"
```

No `--out`, so nothing is written yet. Read, in order: the skip list on
stderr (any `skipped X: reason` here means `filter-corpus` missed something —
it should already be empty); the
`N modules: B benign, M malicious` line against your manifest; the
`classifier` vs `heuristics` metric rows on the same folds (F1/AUC, not
accuracy); the verdict sentence; `Final loss` and the convergence warning;
and the top-10 standardized weights — if the strongest is `log_byteLength` or
`stripped` rather than something like `has_kernel` or `kernel_bitwiseRatio`,
the model found a collection artifact (miners all small, all stripped, all
one toolchain) and the fix is to the corpus, not the model.

### 4.3 Save it

```bash
npm run train -w @wasm-sentry/core -- "$CORPUS" --out extension/public/model.json
npm run build -w extension
```

`extension/public/` is a Vite static dir — rebuild or the new model isn't in
the extension. Confirm in the service worker console:
`[wasm-sentry] classifier loaded: trained on N malicious and M benign modules`.

---

## 5. Reporting the result

Likely outcome: the classifier ties or loses to the heuristics —
`train-model.ts` says so up front, and `crossValidate` prints "ship the
rules, not the model" when it happens. That's a valid result, not a failed
run; report it as such.

Always report: corpus size and composition by source (WasmBench count, npm
count, self-compiled count, per-miner-family count); how many near-duplicate
clusters were collapsed and their sizes; the benign:malicious ratio;
classifier and heuristic metrics on the same folds; precision at the
realistic base rate (not accuracy); and which features had the top weights,
flagged for collection artifacts. Never report a single "detection rate" — no
60-positive, family-concentrated corpus supports one.

---

## 6. Checklist

Each box is one command — run them in this order, on `$CORPUS`:

```
[ ] $CORPUS lives OUTSIDE the repo (e.g. ~/wasm-sentry-corpus), with $CORPUS/benign/ and $CORPUS/malicious/
[ ] benign set spans multiple toolchains AND includes hash-wasm/argon2-browser (§1, step 2)
[ ] every malicious sample has a one-sentence justification in manifest.csv (§3.6, written as you go)
[ ] no testbed/*.wasm fixtures included
[ ] npm run filter-corpus  -w @wasm-sentry/core -- "$CORPUS"     (magic bytes + parses + coverage>=0.9 + size)
[ ] npm run dedupe-corpus  -w @wasm-sentry/core -- "$CORPUS"     (exit code 0 -- no unresolved cross-class hash)
[ ] npm run cluster-corpus -w @wasm-sentry/core -- "$CORPUS"     (near-duplicate clusters reviewed, not just moved)
[ ] npm run split-holdout  -w @wasm-sentry/core -- "$CORPUS"     ($CORPUS/holdout/ populated, never trained on)
[ ] npm run inspect -w @wasm-sentry/core -- $CORPUS/benign/*.wasm $CORPUS/malicious/*.wasm   (skip list empty)
[ ] evaluated with no --out first, all outputs read
[ ] top weights checked for collection artifacts
[ ] npm run build -w extension after writing model.json
```
