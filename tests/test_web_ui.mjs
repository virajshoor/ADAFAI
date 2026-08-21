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
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { analyze, analyzeDiscourse, analyzeSpans, analyzeStylometry, mattr } from "../docs/detector.js";

const fixture = (name) =>
  readFileSync(fileURLToPath(new URL(`./fixtures/${name}`, import.meta.url)), "utf8");

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

/* ---- discourse signal (mirrors tests/test_discourse.py) ---- */

// casual multi-paragraph human prose with two of the tics on purpose: one
// habit alone must not move the verdict
const HUMAN_MULTI =
  "update on the shed roof: it's leaking again. of course it is.\n\n" +
  "I climbed up there saturday with the tar patch stuff and honestly I think " +
  "I made it worse. There's a whole section near the gutter where the plywood " +
  "feels spongy, which dan says means the rot goes deeper than the shingles. " +
  "He offered to come look next weekend but he's said that before.\n\n" +
  "So now I'm watching youtube videos about roof repair at 1am like that's " +
  "going to help. The quotes I got last year were all around 4k which we " +
  "absolutely do not have. Might just let it leak into the bucket for another " +
  "winter and deal with it in spring. that's a problem for future me.\n\n" +
  "Anyway. The tomatoes are doing great at least. Deb came by sunday with her " +
  "ladder and we just sat on the porch instead, drinking her homemade lemonade " +
  "and pretending the roof doesn't exist. honestly? best afternoon in weeks.";

// empty input
{
  const d = analyzeDiscourse("");
  assert.equal(d.score, 0.0);
  assert.equal(d.punch_line_ratio, null);
  assert.equal(d.paragraph_length_cv, null);
}

// both fixtures from the false-negative report are no longer cleared as human
{
  const original = analyze(fixture("ai_essay_original.txt"));
  assert.notEqual(original.verdict, "likely human", original.score);
  assert.ok(original.signals.discourse.negation_pivots_per_1000w >= 5);
  assert.ok(original.signals.discourse.tricolons_per_1000w >= 8);

  const humanized = analyze(fixture("ai_essay_humanized.txt"));
  assert.notEqual(humanized.verdict, "likely human", humanized.score);
  const d = humanized.signals.discourse;
  assert.ok(d.score > humanized.signals.stylometry.score);
  assert.ok(d.self_qa_per_1000w > 0);
  assert.ok(d.negation_pivots_per_1000w > 0);
  assert.ok(d.punch_line_ratio !== null && d.punch_line_ratio >= 0.5);

  assert.equal(analyze(HUMAN_MULTI).verdict, "likely human");
}

// spans: discourse tells get named reasons
{
  const r = analyzeSpans("Was it worth it? Absolutely.");
  const answer = r.sentences.find((s) => s.text === "Absolutely.");
  assert.ok(answer.reasons.some((x) => x.includes("self-answered question")));

  const p = analyzeSpans("This is not merely good. It is great.");
  const pivot = p.sentences.find((s) => s.text.includes("not merely"));
  assert.ok(pivot.reasons.some((x) => x.includes("negation pivot")));
}

console.log("ok");
