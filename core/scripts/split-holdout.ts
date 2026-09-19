/**
 * Split a final, cleaned corpus into a training set and a held-out test set,
 * by the LAST hex digit of the SHA-256 -- reproducible, and stable as the
 * corpus grows (a file added later lands wherever its hash says, not wherever
 * it happened to sort).
 *
 *   npm run split-holdout -w @wasm-sentry/core -- <corpus-dir> [--digits 0,1,2]
 *
 * The last digit, not the first, and the difference is not cosmetic.
 * `cluster-corpus` keeps the alphabetically first file of every cluster, and a
 * corpus harvested from WasmBench is named by hash -- so a cluster's survivor
 * is the *smallest* hash among its members, which starts with 0, 1 or 2 far
 * more often than chance. Splitting on the leading digit therefore sent the
 * representatives of the big clusters to the holdout almost without
 * exception: measured on a real corpus, 4,921 of 5,538 collapsed files
 * belonged to a cluster whose survivor was held out, the holdout took 29% of
 * benign and 40% of malicious instead of 19%, and the families with the most
 * variants -- the ones clustering exists for -- vanished from training. The
 * minimum of a set of hashes is only biased in its leading digits; the
 * trailing ones are still uniform.
 *
 * Run this LAST, after filter-corpus, dedupe-corpus and cluster-corpus.
 * Moves matching files from <corpus-dir>/<class>/ into
 * <corpus-dir>/holdout/<class>/. Never train on the holdout directory; run
 * `npm run train` against it only once, at the end, to report the final number.
 */
import { mkdirSync, readdirSync, readFileSync, renameSync, statSync } from "node:fs";
import { join } from "node:path";
import { sha256 } from "../src/hash.js";

const args = process.argv.slice(2);
const corpusDir = args.find((arg, i) => !arg.startsWith("--") && args[i - 1] !== "--digits");
if (args.includes("--prefixes")) {
  console.error("--prefixes was replaced by --digits (the split now reads the hash's last digit; see the header).");
  process.exit(2);
}
const digitIndex = args.indexOf("--digits");
const digits = (digitIndex >= 0 ? args[digitIndex + 1]! : "0,1,2").split(",");

if (!corpusDir) {
  console.error("usage: split-holdout <corpus-dir> [--digits 0,1,2]");
  process.exit(2);
}

async function splitClass(className: "benign" | "malicious"): Promise<void> {
  const dir = join(corpusDir!, className);
  const holdoutDir = join(corpusDir!, "holdout", className);
  let names: string[];
  try {
    names = readdirSync(dir).filter((name) => name.endsWith(".wasm"));
  } catch {
    console.log(`${className}: ${dir} not readable, skipping`);
    return;
  }

  let moved = 0;
  for (const name of names) {
    const path = join(dir, name);
    if (!statSync(path).isFile()) continue;
    const hash = await sha256(readFileSync(path));
    if (digits.some((d) => hash.endsWith(d))) {
      mkdirSync(holdoutDir, { recursive: true });
      renameSync(path, join(holdoutDir, name));
      moved++;
    }
  }
  console.log(`${className}: ${moved}/${names.length} moved to holdout (last hash digit in ${digits.join(",")})`);
}

await splitClass("benign");
await splitClass("malicious");
console.log(
  "\nTrain only against <corpus-dir> (holdout/ is a sibling, not a subdirectory " +
    "the trainer reads). Evaluate the saved model against <corpus-dir>/holdout once, at the end.",
);
