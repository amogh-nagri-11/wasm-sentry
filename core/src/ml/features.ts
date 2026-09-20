/**
 * Turning a module into a fixed numeric vector.
 *
 * The classifier reads the same `ModuleFeatures` the rules read, from the same
 * code path -- so a rule and a model can never disagree about what they were
 * shown. This file is only the arrangement: which measurements, in which order,
 * on which scale.
 *
 * Three properties matter more than the choice of features themselves.
 *
 * **Order is part of the model.** A vector is meaningless without knowing what
 * column 34 was, so the schema is a named, versioned list, the model records
 * which version it was trained on, and inference refuses a mismatch. A model
 * silently scoring the wrong columns is the failure mode that produces
 * confident nonsense.
 *
 * **Everything is bounded.** Counts are log-scaled and ratios are already 0..1,
 * so a 50 MB module cannot produce a feature a thousand times larger than the
 * one beside it and dominate the fit for no reason.
 *
 * **Nothing here is a verdict.** These are measurements. What they mean is the
 * model's problem, and whether the model is any good is a question about a
 * corpus this project does not yet have.
 */
import type { ModuleFeatures, OpcodeCategory } from "../wasm/features.js";

/**
 * Bumped whenever the columns change in any way -- added, removed, reordered or
 * rescaled. A model trained on an older schema is rejected rather than
 * reinterpreted.
 */
export const FEATURE_SCHEMA_VERSION = 1;

/** Compress an unbounded count into a comparable scale. */
function log1p(value: number): number {
  return Math.log1p(Math.max(0, value));
}

/** Safe division that treats "nothing to divide" as zero rather than NaN. */
function ratio(numerator: number, denominator: number): number {
  return denominator > 0 ? numerator / denominator : 0;
}

const CATEGORIES: OpcodeCategory[] = [
  "control", "call", "variable", "memory", "integer", "bitwise",
  "float", "simd", "atomic", "reference", "other",
];

/**
 * The opcode vocabulary, as a bag rather than a sequence.
 *
 * Deep-Wasm reads opcode *sequences*; this is the honest simplification of that
 * for a linear model -- the shares of a fixed vocabulary. The list is frozen
 * because it is part of the schema: adding an opcode shifts every column after
 * it, which is what the version number is for.
 *
 * Chosen as the operations that distinguish arithmetic kernels from everything
 * else, plus enough ordinary opcodes that "not a kernel" has somewhere to sit.
 */
export const OPCODE_VOCABULARY = [
  "i32.add", "i32.sub", "i32.mul", "i32.and", "i32.or", "i32.xor",
  "i32.shl", "i32.shr_u", "i32.shr_s", "i32.rotl", "i32.rotr",
  "i64.add", "i64.mul", "i64.xor", "i64.shl", "i64.rotl",
  "i32.load", "i32.store", "i64.load", "i64.store",
  "local.get", "local.set", "local.tee", "global.get",
  "call", "call_indirect", "br_if", "br", "loop", "block", "if", "return",
  "f64.add", "f64.mul", "f32.add", "f32.mul",
  "memory.grow", "memory.size",
] as const;

/**
 * Column names, in order. Exported because a feature vector nobody can read is
 * a model nobody can debug: the training CLI prints weights against these.
 */
export const FEATURE_NAMES: readonly string[] = [
  "log_byteLength",
  "log_instructionCount",
  "log_functionCount",
  "log_importedFunctionCount",
  "log_exportCount",
  "exports_per_function",
  "bitwiseRatio",
  "floatRatio",
  ...CATEGORIES.map((category) => `cat_${category}`),
  "loops_per_function",
  "backEdges_per_function",
  "log_maxNesting",
  "indirectCall_ratio",
  "log_memoryInitialPages",
  "memory_bounded",
  "memoryShared",
  "log_memoryGrowSites",
  "dataSection_share",
  "stripped",
  "coverage",
  "has_kernel",
  "kernel_bitwiseRatio",
  "kernel_arithmeticRatio",
  "kernel_callRatio",
  "log_kernel_loopSize",
  ...OPCODE_VOCABULARY.map((opcode) => `op_${opcode}`),
];

/** How many columns a vector has. Asserted against in a test. */
export const FEATURE_COUNT = FEATURE_NAMES.length;

/**
 * Project a module's features into the model's input space.
 *
 * Total and deterministic: the same module always produces the same vector, and
 * no input produces a `NaN`. A `NaN` reaching a trainer poisons every weight it
 * touches and does it silently.
 */
export function vectorise(features: ModuleFeatures): number[] {
  const f = features;
  const instructions = f.instructionCount;
  const functions = Math.max(f.functionCount, 1);
  const kernel = f.kernelCandidate;

  const vector: number[] = [
    log1p(f.byteLength),
    log1p(instructions),
    log1p(f.functionCount),
    log1p(f.importedFunctionCount),
    log1p(f.exportCount),
    ratio(f.exportCount, functions),
    f.bitwiseRatio,
    f.floatRatio,
    ...CATEGORIES.map((category) => ratio(f.categoryCounts[category], instructions)),
    ratio(f.totalLoops, functions),
    ratio(f.totalBackEdges, functions),
    log1p(f.maxNesting),
    ratio(f.indirectCalls, instructions),
    log1p(f.memoryInitialPages),
    f.memoryMaxPages === null ? 0 : 1,
    f.memoryShared ? 1 : 0,
    log1p(f.memoryGrowSites),
    ratio(f.dataSectionBytes, f.byteLength),
    f.stripped ? 1 : 0,
    ratio(f.decodedFunctions, f.decodedFunctions + f.skippedFunctions + f.truncatedFunctions),
    kernel ? 1 : 0,
    kernel?.bitwiseRatio ?? 0,
    kernel?.arithmeticRatio ?? 0,
    kernel?.callRatio ?? 0,
    log1p(kernel?.loopSize ?? 0),
    ...OPCODE_VOCABULARY.map((opcode) => ratio(f.opcodeCounts[opcode] ?? 0, instructions)),
  ];

  // A single NaN silently poisons every weight it touches during training, and
  // there is no way to notice afterwards. Cheaper to refuse to emit one.
  return vector.map((value) => (Number.isFinite(value) ? value : 0));
}
