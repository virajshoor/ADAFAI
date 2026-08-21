/**
 * Self-check for the browser port (docs/detector.js). No test framework:
 *   node tests/test_web_ui.mjs
 * Mirrors tests/test_stylometry.py and tests/test_spans.py so the web UI and
 * the Python package cannot silently drift apart on basic behavior. Full
 * field-level parity with Python is verified by tests/parity fixtures used
 * during development; cross-language CI comparison is intentionally out of
 * scope for a no-dependency repo.
 */
import assert from "node:assert/strict";
import { analyze, analyzeSpans, analyzeStylometry, mattr } from "../docs/detector.js";

const AI_LIKE =
  "In today's world, it is important to note that technology plays a pivotal role. " +
  "Furthermore, this cutting-edge approach will delve into the intricate landscape. " +
  "Moreover, we must underscore the robust, holistic paradigm shift. " +
  "In conclusion, this rich tapestry stands as a testament to innovation.";

const HUMAN_LIKE =
  "so I tried fixing the sink again lol. didn't work. of course. " +
  "Called a plumber, guy showed up 3 hrs late but fixed it in like 5 min?? " +
  "cost me $180 which is insane tbh but whatever, it's done. " +
  "anyway dinner was good, made the pasta thing from last week again.";

// directionality
{
  const ai = analyzeStylometry(AI_LIKE);
  const human = analyzeStylometry(HUMAN_LIKE);
  assert.ok(ai.score > human.score, { ai, human });
  assert.ok(ai.ai_phrase_density_per_1000w > 0);
}

// empty input
{
  const r = analyzeStylometry("");
  assert.equal(r.score, 0.0);
  assert.equal(r.word_count, 0);
  assert.deepEqual(analyzeSpans("").sentences, []);
}

// MATTR == naive window scan
{
  const vocab = Array.from({ length: 40 }, (_, i) => `w${i}`);
  const ws = Array.from({ length: 500 }, (_, i) => vocab[(i * 7 + Math.floor(i / 11)) % 40]);
  const naive = (xs, window = 50) => {
    if (xs.length <= window) return new Set(xs).size / xs.length;
    let total = 0;
    for (let i = 0; i + window <= xs.length; i++) total += new Set(xs.slice(i, i + window)).size / window;
    return total / (xs.length - window + 1);
  };
  assert.ok(Math.abs(mattr(ws) - naive(ws)) < 1e-12);
}

// splitter protects abbreviations and decimals
{
  const sents = analyzeSpans("Dr. Smith paid 3.14 today. Then he left. J. R. R. wrote!");
  assert.deepEqual(sents.sentences.map((s) => s.text),
    ["Dr. Smith paid 3.14 today.", "Then he left.", "J. R. R. wrote!"]);
}

// spans: mixed document localizes the AI section
{
  const mixed = `${HUMAN_LIKE}\n\n${AI_LIKE}\n\n${HUMAN_LIKE}`;
  const r = analyzeSpans(mixed);
  const aiScores = r.sentences.filter((s) => /pivotal|delve|underscore/.test(s.text.toLowerCase())).map((s) => s.score);
  const humanScores = r.sentences.filter((s) => /plumber|pasta|sink/.test(s.text)).map((s) => s.score);
  assert.ok(aiScores.length && humanScores.length);
  assert.ok(Math.min(...aiScores) > Math.max(...humanScores), { aiScores, humanScores });
}

// spans: invisible character overrides statistical score, with a named reason
{
  const r = analyzeSpans("Perfectly ordinary short line.\u200b With a stowaway.");
  const hit = r.sentences.find((s) => s.reasons.some((x) => x.includes("ZERO WIDTH SPACE")));
  assert.ok(hit.score >= 0.85);
  assert.equal(hit.low_confidence, false);
}

// spans: offsets index the original text (needed for in-place highlighting)
{
  const text = `${HUMAN_LIKE}\n\n${AI_LIKE}`;
  for (const s of analyzeSpans(text).sentences) {
    assert.ok(text.slice(s.start, s.end).trim().startsWith(s.text.slice(0, 20)));
  }
}

// ensemble abstains under the word floor
{
  assert.equal(analyze("short text").verdict, "insufficient evidence");
}

console.log("ok");
