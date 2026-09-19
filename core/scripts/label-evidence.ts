/**
 * Print the evidence a label has to stand on, one JSON object per module.
 *
 *   npm run label-evidence -w @wasm-sentry/core -- <file-or-dir>... > evidence.jsonl
 *
 * `docs/DATASET-PLAN.md` §3.6 says a sample without a one-sentence
 * justification does not go in the corpus, and that where a file was found is
 * not a justification. This gathers what is: the names a module declares
 * (exports, imports, and the name section if it was left in), mining-related
 * strings anywhere in its bytes, and what the project's own analysis makes of
 * its structure.
 *
 * It decides nothing. A `signatures` hit is a lead for a human to read, not a
 * label: `_cryptonight_hash` in an export table is strong, `hash` in a name
 * section is nothing. Stripped modules with no hits cannot be verified this
 * way and should stay out of `malicious/` unless something else vouches for them.
 */
import { readdirSync, readFileSync, statSync } from "node:fs";
import { basename, join } from "node:path";
import { analyzeWasm } from "../src/analysis.js";
import { sha256 } from "../src/hash.js";
import { summarise } from "../src/report.js";

/**
 * Terms that name a proof-of-work algorithm, a miner product or a pool
 * protocol. Deliberately excludes general-purpose primitives on their own
 * (sha256, blake2, aes): every TLS or wallet library contains those.
 */
const SIGNATURES =
  /crypto-?night|cryptonote|\bcn_(?:slow_)?hash|_hash_cn|_loot_cn|coin-?hive|coinimp|crypto-?loot|deepminer|webminer|minero|equihash|randomx|yespower|yescrypt|argon2d|cuckoo|nimiq|webdollar|stratum|getwork|\bnonce\b|jsecoin|monero|\bxmr\b|zcash|hushminer|yazec|jazec/gi;

/** Printable ASCII runs, the way `strings` finds them. */
function asciiRuns(bytes: Uint8Array, minLength = 5): string[] {
  const runs: string[] = [];
  let start = -1;
  for (let i = 0; i <= bytes.length; i++) {
    const byte = i < bytes.length ? bytes[i]! : 0;
    const printable = byte >= 0x20 && byte < 0x7f;
    if (printable && start < 0) start = i;
    if (!printable && start >= 0) {
      if (i - start >= minLength) runs.push(String.fromCharCode(...bytes.subarray(start, Math.min(i, start + 200))));
      start = -1;
    }
  }
  return runs;
}

function collect(paths: string[]): string[] {
  const files: string[] = [];
  for (const path of paths) {
    if (statSync(path).isDirectory()) {
      for (const name of readdirSync(path).sort()) {
        if (name.endsWith(".wasm")) files.push(join(path, name));
      }
    } else {
      files.push(path);
    }
  }
  return files;
}

const inputs = process.argv.slice(2);
if (inputs.length === 0) {
  console.error("usage: label-evidence <file-or-dir>...");
  process.exit(2);
}

for (const path of collect(inputs)) {
  const bytes = readFileSync(path);
  const hash = await sha256(bytes);
  const result = analyzeWasm(bytes);
  if (!result.ok) {
    console.log(JSON.stringify({ file: basename(path), sha256: hash, bytes: bytes.length, failed: result.reason }));
    continue;
  }

  const f = result.features;
  const hits = new Map<string, string>();
  for (const run of asciiRuns(bytes)) {
    for (const match of run.matchAll(SIGNATURES)) {
      const key = match[0].toLowerCase();
      if (!hits.has(key)) hits.set(key, run.slice(0, 80));
    }
  }

  const risk = summarise("cli", result).risk;
  console.log(
    JSON.stringify({
      file: basename(path),
      sha256: hash,
      bytes: bytes.length,
      engineValid: WebAssembly.validate(bytes),
      stripped: f.stripped,
      functions: f.functionCount,
      instructions: f.instructionCount,
      bitwiseRatio: Number(f.bitwiseRatio.toFixed(4)),
      floatRatio: Number(f.floatRatio.toFixed(4)),
      memoryPages: f.memoryInitialPages,
      kernel: f.kernelCandidate
        ? {
            loopSize: f.kernelCandidate.loopSize,
            bitwiseRatio: Number(f.kernelCandidate.bitwiseRatio.toFixed(4)),
          }
        : null,
      risk: risk ? { score: risk.score, level: risk.level, findings: risk.findings.map((x) => x.id) } : null,
      exports: f.exportNames.slice(0, 40),
      imports: f.importNames.slice(0, 40),
      signatures: Object.fromEntries(hits),
    }),
  );
}
