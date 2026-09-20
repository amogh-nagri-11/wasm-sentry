/**
 * Turn a corpus into fixed-length opcode-token sequences for the deep-model
 * track (`docs/DEEP-MODEL-GUIDE.md`, `docs/COLAB-TRAINING-GUIDE.md`).
 *
 *   npm run extract-sequences -w @wasm-sentry/core -- <corpus-dir> --out finetune.jsonl
 *   npm run extract-sequences -w @wasm-sentry/core -- <pool-dir>   --out pretrain.jsonl --unlabeled
 *
 * Labelled mode reads <corpus-dir>/benign and <corpus-dir>/malicious, the same
 * convention as `train-model.ts`, and never descends into `holdout/` -- point
 * it at <corpus-dir>/holdout separately, once, at the end. Unlabelled mode
 * reads every `.wasm` file directly inside <pool-dir> and writes `label: null`.
 *
 * One JSON object per line:
 *
 *   {"sha256": "...", "tokens": [2, 25, ...], "label": 0 | 1 | null,
 *    "source": "file.wasm", "kernel": true, "realTokens": 512}
 *
 * A sibling `<out>.vocab.json` records the token table and schema version, so
 * the Python side reads the vocabulary size instead of hard-coding it.
 *
 * Rows are de-duplicated by SHA-256 across the whole run: the same bytes under
 * two names would otherwise be able to sit on both sides of a fold.
 */
import { closeSync, openSync, readdirSync, readFileSync, statSync, writeFileSync, writeSync } from "node:fs";
import { join } from "node:path";
import { analyzeWasm } from "../src/analysis.js";
import { sha256 } from "../src/hash.js";
import {
  extractSequence,
  SEQUENCE_LENGTH,
  SEQUENCE_SCHEMA_VERSION,
  SEQUENCE_VOCABULARY,
} from "../src/ml/sequences.js";

const args = process.argv.slice(2);
const flagValue = (flag: string): string | undefined => {
  const index = args.indexOf(flag);
  return index >= 0 ? args[index + 1] : undefined;
};
const valueFlags = new Set(["--out", "--length"]);
const inputDir = args.find((arg, i) => !arg.startsWith("--") && !valueFlags.has(args[i - 1] ?? ""));
const outPath = flagValue("--out");
const unlabeled = args.includes("--unlabeled");
const length = Number(flagValue("--length") ?? SEQUENCE_LENGTH);

if (!inputDir || !outPath || !Number.isInteger(length) || length < 2) {
  console.error("usage: extract-sequences <dir> --out <file.jsonl> [--unlabeled] [--length 512]");
  console.error("  labelled (default): <dir> contains benign/ and malicious/");
  console.error("  --unlabeled:        <dir> contains .wasm files directly");
  process.exit(2);
}

const out = openSync(outPath, "w");
const seen = new Set<string>();
let written = 0;
let skipped = 0;
let duplicates = 0;
let withKernel = 0;
let padded = 0;

async function processDirectory(dir: string, label: 0 | 1 | null): Promise<number> {
  let names: string[];
  try {
    names = readdirSync(dir).filter((name) => name.endsWith(".wasm")).sort();
  } catch {
    console.error(`  ${dir}: not readable`);
    return 0;
  }

  let count = 0;
  for (const name of names) {
    const path = join(dir, name);
    if (!statSync(path).isFile()) continue;
    const bytes = readFileSync(path);

    const hash = await sha256(bytes);
    if (seen.has(hash)) {
      duplicates++;
      continue;
    }

    // Every function row is needed to find the largest function honestly.
    const result = analyzeWasm(bytes, { maxFunctionRows: Number.MAX_SAFE_INTEGER });
    if (!result.ok) {
      console.error(`  skipped ${name}: ${result.reason}`);
      skipped++;
      continue;
    }

    const sequence = extractSequence(result.module, result.features, length);
    if (sequence.functions.length === 0) {
      console.error(`  skipped ${name}: no decodable function body`);
      skipped++;
      continue;
    }

    seen.add(hash);
    if (sequence.fromKernelCandidate) withKernel++;
    if (sequence.realTokens < length) padded++;
    writeSync(
      out,
      `${JSON.stringify({
        sha256: hash,
        tokens: sequence.tokens,
        label,
        source: name,
        kernel: sequence.fromKernelCandidate,
        realTokens: sequence.realTokens,
      })}\n`,
    );
    written++;
    count++;
  }
  return count;
}

if (unlabeled) {
  const count = await processDirectory(inputDir, null);
  console.log(`unlabeled: ${count} sequences`);
} else {
  const benign = await processDirectory(join(inputDir, "benign"), 0);
  const malicious = await processDirectory(join(inputDir, "malicious"), 1);
  console.log(`benign: ${benign} sequences, malicious: ${malicious} sequences`);
}
closeSync(out);

const vocabPath = `${outPath}.vocab.json`;
writeFileSync(
  vocabPath,
  `${JSON.stringify(
    { schemaVersion: SEQUENCE_SCHEMA_VERSION, sequenceLength: length, vocabulary: SEQUENCE_VOCABULARY },
    null,
    2,
  )}\n`,
);

console.log(
  `\nWrote ${written} rows to ${outPath} (${length} tokens each, vocabulary ${SEQUENCE_VOCABULARY.length}).\n` +
    `  ${withKernel} led by a kernel candidate, ${padded} shorter than the window and padded,\n` +
    `  ${skipped} skipped, ${duplicates} duplicate hashes dropped.\n` +
    `Wrote ${vocabPath}`,
);
if (written === 0) {
  // `npm run -w` runs from core/, so a relative path resolves there, not where
  // the command was typed. An empty file that exits 0 is how that goes unnoticed.
  console.error(`
No sequences written. Is ${inputDir} an absolute path to the right directory?`);
  process.exit(1);
}
if (skipped > 0) {
  console.log("Skips mean filter-corpus has not been run over this directory, or missed something.");
}
