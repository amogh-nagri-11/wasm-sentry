/**
 * Turning a module into a fixed-length opcode-token sequence.
 *
 * The linear classifier reads a bag of opcode shares; the deep-model track
 * (`docs/DEEP-MODEL-GUIDE.md`) reads the *order*. This file is the only place
 * that order is decided, so the training notebook and any later in-browser
 * inference tokenise identically -- a model fed sequences cut differently from
 * the ones it was trained on scores confidently and wrongly, which is the same
 * failure the feature schema version exists to prevent.
 *
 * Three decisions are made here, and each is a choice rather than a measurement:
 *
 * **Which function.** The kernel candidate if the module has one, otherwise the
 * largest function. That is the heuristic engine's own judgement about where
 * the evidence is, so the model looks at what the rules look at. When the
 * primary function does not fill the window, the next-largest functions are
 * appended behind a `SEP` (up to four functions in all) so a module made of
 * many small functions is not represented by padding.
 *
 * **Which part of it.** A function longer than the window is not cut from the
 * top. A hashing kernel's prologue is locals and loads like any other
 * function's; what makes it a kernel is the loop. So the window opens at the
 * header of the largest loop, pulled back only as far as needed to keep the
 * window full.
 *
 * **Which tokens.** `OPCODE_VOCABULARY` exactly, so the two models agree on
 * what an operation is, plus `OTHER` for everything outside it.
 */
import { buildCfg } from "../wasm/cfg.js";
import { decodeExpression } from "../wasm/decode.js";
import type { ModuleFeatures } from "../wasm/features.js";
import type { CodeEntry, WasmModule } from "../wasm/module.js";
import { Reader } from "../wasm/reader.js";
import { OPCODE_VOCABULARY } from "./features.js";

/**
 * Bumped whenever token ids, selection or windowing change. Written into every
 * output file; a model must record the version it was trained on.
 */
export const SEQUENCE_SCHEMA_VERSION = 1;

/** Tokens per sequence, `CLS` included. A hyperparameter, not an optimum. */
export const SEQUENCE_LENGTH = 512;

/** At most this many functions contribute to one sequence. */
export const MAX_FUNCTIONS_PER_SEQUENCE = 4;

export const SPECIAL_TOKENS = ["PAD", "MASK", "CLS", "UNK", "OTHER", "SEP"] as const;

/** Index is the token id. `PAD` must stay 0: the embedding's padding index. */
export const SEQUENCE_VOCABULARY: readonly string[] = [...SPECIAL_TOKENS, ...OPCODE_VOCABULARY];

export const TOKEN = {
  PAD: 0,
  MASK: 1,
  CLS: 2,
  UNK: 3,
  OTHER: 4,
  SEP: 5,
} as const;

const TOKEN_IDS: ReadonlyMap<string, number> = new Map(
  OPCODE_VOCABULARY.map((opcode, index) => [opcode, SPECIAL_TOKENS.length + index]),
);

/** Token id for one instruction name. Total: unknown names are `OTHER`. */
export function tokenFor(instructionName: string): number {
  return TOKEN_IDS.get(instructionName) ?? TOKEN.OTHER;
}

export interface SequenceResult {
  /** Exactly `length` token ids, `CLS` first, `PAD`-filled at the end. */
  tokens: number[];
  /** Defined-function indices that contributed, in order. */
  functions: number[];
  /** True when the primary function is the module's kernel candidate. */
  fromKernelCandidate: boolean;
  /** Non-padding tokens, `CLS` and `SEP` included. */
  realTokens: number;
}

/** Tokenise one function body, windowed onto its largest loop if too long. */
function tokeniseFunction(module: WasmModule, entry: CodeEntry, room: number): number[] {
  const reader = new Reader(module.bytes, entry.bodyStart);
  const { instructions } = decodeExpression(reader, entry.bodyEnd);
  if (instructions.length <= room) return instructions.map((i) => tokenFor(i.name));

  let start = 0;
  const loops = buildCfg(instructions).loops;
  if (loops.length > 0) {
    const largest = loops.reduce((best, loop) => (loop.size > best.size ? loop : best));
    start = Math.min(largest.header, instructions.length - room);
  }
  return instructions.slice(start, start + room).map((i) => tokenFor(i.name));
}

/**
 * Build the sequence for a parsed module.
 *
 * `features` must come from `extractFeatures` over the same module with
 * `maxFunctionRows` high enough to cover every function, or the "largest
 * function" is only the largest among the rows that were kept.
 */
export function extractSequence(
  module: WasmModule,
  features: ModuleFeatures,
  length: number = SEQUENCE_LENGTH,
): SequenceResult {
  const byIndex = new Map(module.code.map((entry) => [entry.index, entry]));
  const bySize = [...features.functions]
    .filter((row) => row.instructionCount > 0)
    // Ties broken on index so the same module always yields the same sequence.
    .sort((a, b) => b.instructionCount - a.instructionCount || a.index - b.index)
    .map((row) => row.index);

  const kernel = features.kernelCandidate?.functionIndex;
  const order = kernel !== undefined ? [kernel, ...bySize.filter((index) => index !== kernel)] : bySize;

  const tokens: number[] = [TOKEN.CLS];
  const used: number[] = [];
  for (const index of order) {
    if (used.length >= MAX_FUNCTIONS_PER_SEQUENCE) break;
    // A SEP with nothing after it says nothing; need room for it and one token.
    const separator = used.length > 0 ? 1 : 0;
    const room = length - tokens.length - separator;
    if (room < 1) break;
    const entry = byIndex.get(index);
    if (!entry) continue;
    const body = tokeniseFunction(module, entry, room);
    if (body.length === 0) continue;
    if (separator) tokens.push(TOKEN.SEP);
    tokens.push(...body);
    used.push(index);
  }

  const realTokens = tokens.length;
  while (tokens.length < length) tokens.push(TOKEN.PAD);

  return {
    tokens,
    functions: used,
    fromKernelCandidate: kernel !== undefined && used[0] === kernel,
    realTokens,
  };
}
