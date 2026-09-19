import { test } from "node:test";
import assert from "node:assert/strict";
import { analyzeWasm } from "../src/analysis.js";
import { OPCODE_VOCABULARY } from "../src/ml/features.js";
import {
  extractSequence,
  SEQUENCE_LENGTH,
  SEQUENCE_VOCABULARY,
  TOKEN,
  tokenFor,
} from "../src/ml/sequences.js";
import { buildCfg } from "../src/wasm/cfg.js";
import { decodeExpression } from "../src/wasm/decode.js";
import { Reader } from "../src/wasm/reader.js";
import { benignModule, minerLikeModule, syntheticMinerModule } from "./fixtures.js";

function sequenceOf(bytes: Uint8Array, length?: number) {
  const result = analyzeWasm(bytes, { maxFunctionRows: Number.MAX_SAFE_INTEGER });
  assert.equal(result.ok, true);
  if (!result.ok) throw new Error("unreachable");
  return { result, sequence: extractSequence(result.module, result.features, length) };
}

test("token ids are the vocabulary's own indices, with PAD at zero", () => {
  assert.equal(SEQUENCE_VOCABULARY[TOKEN.PAD], "PAD");
  assert.equal(SEQUENCE_VOCABULARY[TOKEN.MASK], "MASK");
  assert.equal(SEQUENCE_VOCABULARY[TOKEN.CLS], "CLS");
  assert.equal(SEQUENCE_VOCABULARY[TOKEN.OTHER], "OTHER");
  assert.equal(SEQUENCE_VOCABULARY[TOKEN.SEP], "SEP");
  for (const opcode of OPCODE_VOCABULARY) {
    assert.equal(SEQUENCE_VOCABULARY[tokenFor(opcode)], opcode);
  }
  assert.equal(new Set(SEQUENCE_VOCABULARY).size, SEQUENCE_VOCABULARY.length);
});

test("an opcode outside the vocabulary is OTHER, never a special token", () => {
  assert.equal(tokenFor("f64.sqrt"), TOKEN.OTHER);
  assert.equal(tokenFor("PAD"), TOKEN.OTHER);
  assert.equal(tokenFor(""), TOKEN.OTHER);
});

test("every sequence is exactly the window, CLS first, padding only at the end", () => {
  for (const bytes of [benignModule(), minerLikeModule(), syntheticMinerModule()]) {
    const { sequence } = sequenceOf(bytes);
    assert.equal(sequence.tokens.length, SEQUENCE_LENGTH);
    assert.equal(sequence.tokens[0], TOKEN.CLS);
    assert.ok(sequence.tokens.every((t) => Number.isInteger(t) && t >= 0 && t < SEQUENCE_VOCABULARY.length));
    assert.ok(!sequence.tokens.includes(TOKEN.MASK), "masking belongs to training, not extraction");
    const firstPad = sequence.tokens.indexOf(TOKEN.PAD);
    if (firstPad >= 0) {
      assert.equal(firstPad, sequence.realTokens);
      assert.ok(sequence.tokens.slice(firstPad).every((t) => t === TOKEN.PAD));
    } else {
      assert.equal(sequence.realTokens, SEQUENCE_LENGTH);
    }
  }
});

test("a kernel candidate leads the sequence, and its tokens are its own instructions", () => {
  const { result, sequence } = sequenceOf(syntheticMinerModule());
  const kernel = result.features.kernelCandidate;
  assert.ok(kernel, "fixture should have a kernel candidate");
  assert.equal(sequence.fromKernelCandidate, true);
  assert.equal(sequence.functions[0], kernel.functionIndex);

  const entry = result.module.code.find((e) => e.index === kernel.functionIndex);
  assert.ok(entry);
  const { instructions } = decodeExpression(new Reader(result.module.bytes, entry.bodyStart), entry.bodyEnd);
  const expected = instructions.map((i) => tokenFor(i.name)).slice(0, SEQUENCE_LENGTH - 1);
  assert.deepEqual(sequence.tokens.slice(1, 1 + expected.length), expected);
});

test("a function longer than the window is cut at its largest loop, not from the top", () => {
  const { result, sequence } = sequenceOf(syntheticMinerModule({ rounds: 64 }), 64);
  assert.equal(sequence.tokens.length, 64);
  assert.equal(sequence.realTokens, 64);
  assert.equal(sequence.functions.length, 1, "a full window leaves no room for a second function");

  const entry = result.module.code.find((e) => e.index === sequence.functions[0]);
  assert.ok(entry);
  const { instructions } = decodeExpression(new Reader(result.module.bytes, entry.bodyStart), entry.bodyEnd);
  assert.ok(instructions.length > 63, "fixture must overflow the window for this test to mean anything");
  const largest = buildCfg(instructions).loops.reduce((best, loop) => (loop.size > best.size ? loop : best));
  const start = Math.min(largest.header, instructions.length - 63);
  assert.ok(start > 0, "fixture's loop must sit past the top, or this cannot tell the two cuts apart");
  assert.deepEqual(
    sequence.tokens.slice(1),
    instructions.slice(start, start + 63).map((i) => tokenFor(i.name)),
  );
});

test("extraction is deterministic", () => {
  const a = sequenceOf(syntheticMinerModule()).sequence;
  const b = sequenceOf(syntheticMinerModule()).sequence;
  assert.deepEqual(a, b);
});

test("small functions are joined by SEP, up to four, and never end on a dangling SEP", () => {
  const { sequence } = sequenceOf(benignModule());
  const seps = sequence.tokens.filter((t) => t === TOKEN.SEP).length;
  assert.equal(seps, sequence.functions.length - 1);
  assert.ok(sequence.functions.length <= 4);
  assert.notEqual(sequence.tokens[sequence.realTokens - 1], TOKEN.SEP);
});
