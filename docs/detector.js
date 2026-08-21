/**
 * ADAFAI core, ported to dependency-free JavaScript so detection runs entirely
 * in the browser (GitHub Pages has no backend, and nothing is uploaded).
 *
 * This is a line-by-line port of adafai/stylometry.py, adafai/unicode_forensics.py,
 * adafai/detector.py (always-on signals only), and adafai/spans.py. Thresholds and
 * weights are identical to the Python originals; keep the two in lockstep.
 * The torch-based signals (perplexity, Binoculars, watermarks) cannot run in a
 * browser and are intentionally absent - the UI says so.
 */

const WORD_RE = /[A-Za-z']+/g;
const SENT_RE = /[^.!?]+[.!?]+|[^.!?]+$/gd;
const PUNCT_CHARS = ".,;:!?—–-\"'()";

/* Same boundary protection as stylometry.py: abbreviation, initial, and
   decimal periods are not sentence boundaries. */
const PLACEHOLDER = "\uE000";
const ABBREV_RE = /\b(?:Mr|Mrs|Ms|Dr|St|vs|Jr|Sr|Prof|cf|etc|Inc|Ltd|No|Fig|Eq|al|e\.g|i\.e)\./g;
const INITIAL_RE = /\b[A-Z]\./g;
const DECIMAL_RE = /(?<=\d)\.(?=\d)/g;

export const AI_TELL_PHRASES = [
  "delve into", "delve", "boasts", "boast", "bolster", "bolstered",
  "underscore", "underscores", "underscoring", "tapestry", "testament",
  "meticulous", "meticulously", "intricate", "intricacies", "interplay",
  "pivotal", "landscape", "realm", "beacon", "cacophony", "myriad",
  "plethora", "holistic", "paradigm shift", "cutting-edge", "game-changer",
  "leverage", "leveraging", "harness", "illuminate", "facilitate",
  "navigate the complexities", "unlock the potential", "seamlessly",
  "in today's world", "in the ever-evolving", "when it comes to",
  "it is important to note", "it's important to note", "in conclusion",
  "in summary", "in essence", "on the other hand", "furthermore",
  "moreover", "notably", "commendable", "robust", "at its core",
  "plays a pivotal role", "stands as a testament", "rich tapestry",
  "as an ai language model", "vibrant",
];

function words(text) {
  return text.toLowerCase().match(WORD_RE) ?? [];
}

function protect(text) {
  return text
    .replace(DECIMAL_RE, PLACEHOLDER)
    .replace(ABBREV_RE, (m) => m.replaceAll(".", PLACEHOLDER))
    .replace(INITIAL_RE, (m) => m.replaceAll(".", PLACEHOLDER));
}

export function sentenceSpans(text) {
  const restored = protect(text);
  const spans = [];
  for (const m of restored.matchAll(SENT_RE)) {
    if (m[0].trim()) {
      spans.push({ start: m.indices[0][0], end: m.indices[0][1],
                   text: m[0].replaceAll(PLACEHOLDER, ".").trim() });
    }
  }
  return spans;
}

export function mattr(wordList, window = 50) {
  const n = wordList.length;
  if (n === 0) return 0.0;
  if (n <= window) return new Set(wordList).size / n;
  const counts = new Map();
  for (const w of wordList.slice(0, window)) counts.set(w, (counts.get(w) ?? 0) + 1);
  let distinct = counts.size;
  let total = distinct;
  for (let i = window; i < n; i++) {
    const oldW = wordList[i - window];
    const oldC = counts.get(oldW) - 1;
    counts.set(oldW, oldC);
    if (oldC === 0) distinct -= 1;
    const newW = wordList[i];
    const newC = counts.get(newW) ?? 0;
    if (newC === 0) distinct += 1;
    counts.set(newW, newC + 1);
    total += distinct;
  }
  return total / ((n - window + 1) * window);
}

function sentenceLengthCV(sentences) {
  const lengths = sentences.map((s) => words(s).length).filter((n) => n > 0);
  if (lengths.length < 2) return 1.0;
  const mean = lengths.reduce((a, b) => a + b, 0) / lengths.length;
  if (mean === 0) return 1.0;
  const variance = lengths.reduce((a, x) => a + (x - mean) ** 2, 0) / lengths.length;
  return Math.sqrt(variance) / mean;
}

function punctuationEntropy(text) {
  const counts = new Map();
  for (const c of text) if (PUNCT_CHARS.includes(c)) counts.set(c, (counts.get(c) ?? 0) + 1);
  const total = [...counts.values()].reduce((a, b) => a + b, 0);
  if (total === 0) return 0.0;
  let entropy = 0.0;
  for (const c of counts.values()) {
    const p = c / total;
    entropy -= p * Math.log2(p);
  }
  return entropy;
}

function repeatedTrigramRatio(wordList) {
  if (wordList.length < 3) return 0.0;
  const n = wordList.length - 2;
  const counts = new Map();
  for (let i = 0; i < n; i++) {
    const key = `${wordList[i]} ${wordList[i + 1]} ${wordList[i + 2]}`;
    counts.set(key, (counts.get(key) ?? 0) + 1);
  }
  let repeated = 0;
  for (const c of counts.values()) if (c > 1) repeated += c - 1;
  return repeated / n;
}

function countOccurrences(haystack, needle) {
  if (!needle) return 0;
  return haystack.split(needle).length - 1;
}

function aiPhraseDensity(text, wordCount) {
  if (wordCount === 0) return 0.0;
  const low = text.toLowerCase();
  const hits = AI_TELL_PHRASES.reduce((a, p) => a + countOccurrences(low, p), 0);
  return (hits * 1000) / wordCount;
}

const round2 = (x) => Math.round(x * 100) / 100;
const round4 = (x) => Math.round(x * 1e4) / 1e4;

export function analyzeStylometry(text) {
  const w = words(text);
  const sentences = sentenceSpans(text).map((s) => s.text);
  const wordCount = w.length;
  if (wordCount === 0) {
    return { word_count: 0, sentence_count: 0, lexical_diversity_mattr: 0.0,
             sentence_length_cv: 0.0, repeated_trigram_ratio: 0.0,
             ai_phrase_density_per_1000w: 0.0, punctuation_entropy: 0.0, score: 0.0 };
  }

  const diversity = mattr(w);
  const cv = sentenceLengthCV(sentences);
  const repTri = repeatedTrigramRatio(w);
  const phraseDensity = aiPhraseDensity(text, wordCount);
  const punctEnt = punctuationEntropy(text);

  const phraseTerm = Math.min(phraseDensity / 5.0, 1.0);
  const cvTerm = Math.max(0.0, (0.6 - cv) / 0.6);
  const diversityTerm = Math.max(0.0, (0.75 - diversity) / 0.75);
  const repTerm = Math.min(repTri / 0.15, 1.0);

  const score = 0.35 * phraseTerm + 0.25 * cvTerm + 0.20 * diversityTerm + 0.20 * repTerm;

  return {
    word_count: wordCount,
    sentence_count: sentences.length,
    lexical_diversity_mattr: round4(diversity),
    sentence_length_cv: round4(cv),
    repeated_trigram_ratio: round4(repTri),
    ai_phrase_density_per_1000w: round2(phraseDensity),
    punctuation_entropy: round4(punctEnt),
    score: round4(Math.min(Math.max(score, 0.0), 1.0)),
  };
}

/* Tables keyed by explicit escapes - the characters themselves are invisible
   or confusable, which is the entire point of the check. */
const INVISIBLE = new Map([
  ["\u200B", "ZERO WIDTH SPACE"],
  ["\u200C", "ZERO WIDTH NON-JOINER"],
  ["\u200D", "ZERO WIDTH JOINER"],
  ["\u2060", "WORD JOINER"],
  ["\uFEFF", "ZERO WIDTH NO-BREAK SPACE (BOM)"],
  ["\u00AD", "SOFT HYPHEN"],
  ["\u180E", "MONGOLIAN VOWEL SEPARATOR"],
  ["\u061C", "ARABIC LETTER MARK"],
  ["\u200E", "LEFT-TO-RIGHT MARK"],
  ["\u200F", "RIGHT-TO-LEFT MARK"],
]);

const EXOTIC_SPACES = new Map([
  ["\u00A0", "NO-BREAK SPACE"],
  ["\u202F", "NARROW NO-BREAK SPACE"],
  ["\u2009", "THIN SPACE"],
  ["\u2002", "EN SPACE"],
  ["\u2003", "EM SPACE"],
  ["\u2007", "FIGURE SPACE"],
]);

const HOMOGLYPHS = new Map([
  ["\u0410", "A"], ["\u0412", "B"], ["\u0415", "E"], ["\u041A", "K"], ["\u041C", "M"],
  ["\u041D", "H"], ["\u041E", "O"], ["\u0420", "P"], ["\u0421", "C"], ["\u0422", "T"],
  ["\u0425", "X"], ["\u0430", "a"], ["\u0435", "e"], ["\u043E", "o"], ["\u0440", "p"],
  ["\u0441", "c"], ["\u0445", "x"], ["\u0443", "y"], ["\u0391", "A"], ["\u0392", "B"],
  ["\u0395", "E"], ["\u0397", "H"], ["\u0399", "I"], ["\u039A", "K"], ["\u039C", "M"],
  ["\u039D", "N"], ["\u039F", "O"], ["\u03A1", "P"], ["\u03A4", "T"], ["\u03A7", "X"],
]);

const UNICODE_WORD_RE = /[\p{L}\p{M}]+/gu;

function findIn(text, table) {
  const hits = {};
  for (const [ch, name] of table) {
    const n = countOccurrences(text, ch);
    if (n) hits[name] = { codepoint: `U+${ch.codePointAt(0).toString(16).toUpperCase().padStart(4, "0")}`, count: n };
  }
  return hits;
}

const SCRIPT_RES = [
  ["LATIN", /\p{Script=Latin}/u], ["GREEK", /\p{Script=Greek}/u],
  ["CYRILLIC", /\p{Script=Cyrillic}/u], ["ARABIC", /\p{Script=Arabic}/u],
  ["HEBREW", /\p{Script=Hebrew}/u], ["CJK", /\p{Script=Han}/u],
  ["HIRAGANA", /\p{Script=Hiragana}/u], ["KATAKANA", /\p{Script=Katakana}/u],
  ["HANGUL", /\p{Script=Hangul}/u], ["DEVANAGARI", /\p{Script=Devanagari}/u],
  ["THAI", /\p{Script=Thai}/u], ["ARMENIAN", /\p{Script=Armenian}/u],
  ["GEORGIAN", /\p{Script=Georgian}/u],
];

export function analyzeUnicode(text) {
  const wordCount = (text.match(UNICODE_WORD_RE) ?? []).length;
  if (!text) {
    return { word_count: 0, invisible_characters: {}, exotic_spaces: {},
             homoglyphs: {}, typography: {}, scripts: [], flags: [], score: 0.0 };
  }

  const invisible = findIn(text, INVISIBLE);
  const spaces = findIn(text, EXOTIC_SPACES);
  const homoglyphs = findIn(text, HOMOGLYPHS);

  const curly = [..."‘’“”"].reduce((a, c) => a + countOccurrences(text, c), 0);
  const straight = countOccurrences(text, "'") + countOccurrences(text, '"');
  const emDash = countOccurrences(text, "—");
  const enDash = countOccurrences(text, "–");
  const ellipsis = countOccurrences(text, "…");
  const totalQuotes = curly + straight;

  const typography = {
    curly_quotes: curly,
    straight_quotes: straight,
    curly_quote_ratio: totalQuotes ? round4(curly / totalQuotes) : null,
    em_dashes: emDash,
    en_dashes: enDash,
    ellipsis_char: ellipsis,
    em_dashes_per_1000w: wordCount ? round2((emDash * 1000) / wordCount) : 0.0,
  };

  const flags = [];
  if (Object.keys(invisible).length) flags.push("invisible characters present - anomalous in typed text");
  if (Object.keys(homoglyphs).length) flags.push("mixed-script homoglyphs - possible detector evasion");
  if (Object.keys(spaces).length) flags.push("exotic space characters present");
  // Python: not text.isprintable() and no \n / \t anywhere. isprintable()
  // is False for all C* categories (incl. zero-width/format chars) and for
  // separator characters other than the plain space.
  const hasNonPrintable = [...text].some(
    (c) => /[\p{C}\p{Zl}\p{Zp}]/u.test(c) || (c !== " " && /\p{Zs}/u.test(c)),
  );
  if (hasNonPrintable && !text.includes("\n") && !text.includes("\t")) {
    flags.push("non-printable characters present");
  }

  const sumCounts = (obj) => Object.values(obj).reduce((a, v) => a + v.count, 0);
  const invisibleTerm = Object.keys(invisible).length ? Math.min(sumCounts(invisible) / 3.0, 1.0) : 0.0;
  const homoglyphTerm = Object.keys(homoglyphs).length ? 1.0 : 0.0;
  const spaceTerm = Object.keys(spaces).length ? Math.min(sumCounts(spaces) / 5.0, 1.0) : 0.0;

  let typoTerm = 0.0;
  if (totalQuotes >= 4 && curly / totalQuotes > 0.9) typoTerm += 0.5;
  if (wordCount >= 100 && (emDash * 1000) / wordCount > 8) typoTerm += 0.5;
  typoTerm = Math.min(typoTerm, 1.0);

  const score = Math.min(
    0.45 * invisibleTerm + 0.30 * homoglyphTerm + 0.10 * spaceTerm + 0.15 * typoTerm,
    1.0,
  );

  const scripts = new Set();
  for (const c of text) {
    if (/\p{L}/u.test(c)) {
      const hit = SCRIPT_RES.find(([, re]) => re.test(c));
      scripts.add(hit ? hit[0] : "OTHER");
    }
  }

  return {
    word_count: wordCount,
    invisible_characters: invisible,
    exotic_spaces: spaces,
    homoglyphs,
    typography,
    scripts: [...scripts].sort(),
    flags,
    score: round4(score),
  };
}

export const MIN_WORDS_FOR_VERDICT = 150;

export function analyze(text) {
  const signals = { stylometry: analyzeStylometry(text), unicode: analyzeUnicode(text) };
  const weights = { stylometry: 0.30, unicode: 0.10 };

  const total = Object.values(weights).reduce((a, b) => a + b, 0);
  const score = Object.entries(weights).reduce((a, [k, w]) => a + signals[k].score * w, 0) / total;

  const wordCount = signals.stylometry.word_count;
  let verdict;
  if (wordCount < MIN_WORDS_FOR_VERDICT) verdict = "insufficient evidence";
  else if (score < 0.30) verdict = "likely human";
  else if (score < 0.70) verdict = "uncertain";
  else verdict = "likely AI-generated";

  const result = { score: round4(score), verdict, word_count: wordCount, signals };
  result.note = verdict === "insufficient evidence"
    ? `${wordCount} words is below the ${MIN_WORDS_FOR_VERDICT}-word floor for a statistical verdict; the score is reported but not meaningful.`
    : "Heuristic ensemble with unfitted weights - triage signal, not proof. Never use as sole evidence in an academic-integrity or disciplinary decision.";
  if (signals.unicode.flags.length) result.provenance_flags = signals.unicode.flags;
  return result;
}

/* ---- spans.py port: per-sentence localization + reasons ---- */

export const MIN_WORDS_LOCAL = 8;
export const CONTEXT_WORDS = 100;

function present(text, table) {
  const out = [];
  for (const [ch, name] of table) {
    if (text.includes(ch)) out.push([name, countOccurrences(text, ch)]);
  }
  return out;
}

function bisectLeft(arr, x) {
  let lo = 0;
  let hi = arr.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (arr[mid] < x) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

export function analyzeSpans(text) {
  const spans = sentenceSpans(text);
  const wordMatches = [...text.toLowerCase().matchAll(/[A-Za-z']+/gd)];
  const wordList = wordMatches.map((m) => m[0]);
  const wordStarts = wordMatches.map((m) => m.indices[0][0]);
  const half = CONTEXT_WORDS / 2;

  const sentences = spans.map(({ start, end, text: sent }) => {
    const low = sent.toLowerCase();
    const nWords = (low.match(WORD_RE) ?? []).length;
    const reasons = [];

    const phraseHits = AI_TELL_PHRASES.filter((p) => low.includes(p)).sort();
    const hits = AI_TELL_PHRASES.reduce((a, p) => a + countOccurrences(low, p), 0);

    const i0 = bisectLeft(wordStarts, start);
    const i1 = bisectLeft(wordStarts, end);
    const ctx = wordList.slice(Math.max(0, i0 - half), i1 + half);
    const diversity = mattr(ctx);
    const rep = repeatedTrigramRatio(ctx);

    const invisible = present(sent, INVISIBLE);
    const exotic = present(sent, EXOTIC_SPACES);
    const homoglyphs = present(sent, HOMOGLYPHS);
    const typo = [..."‘’“”"].reduce((a, c) => a + countOccurrences(sent, c), 0) + countOccurrences(sent, "—");

    const phraseTerm = Math.min(hits / 2.0, 1.0);
    const diversityTerm = Math.max(0.0, (0.75 - diversity) / 0.75);
    const repTerm = Math.min(rep / 0.15, 1.0);
    const typoTerm = typo ? 1.0 : 0.0;

    let score = 0.40 * phraseTerm + 0.20 * diversityTerm + 0.20 * repTerm + 0.10 * typoTerm;
    let artifact = false;
    if (invisible.length) { score = Math.max(score, 0.85); artifact = true; }
    if (homoglyphs.length) { score = Math.max(score, 0.90); artifact = true; }
    if (exotic.length) { score = Math.min(1.0, score + 0.15); artifact = true; }

    if (phraseHits.length) {
      const shown = phraseHits.slice(0, 4).map((p) => `'${p}'`).join(", ");
      const more = phraseHits.length > 4 ? ` (+${phraseHits.length - 4} more)` : "";
      reasons.push(`AI-tell phrase${phraseHits.length > 1 ? "s" : ""}: ${shown}${more}`);
    }
    for (const [name, c] of invisible) reasons.push(`invisible character ${name} x${c} - anomalous in typed text`);
    for (const [name, c] of homoglyphs) reasons.push(`lookalike character ${name} x${c} - possible detector evasion`);
    for (const [name, c] of exotic) reasons.push(`exotic space character ${name} x${c}`);
    if (diversityTerm > 0 && ctx.length >= 30) {
      reasons.push(`low local lexical diversity (MATTR ${diversity.toFixed(2)}; human baseline ~0.75+)`);
    }
    if (rep > 0) reasons.push(`${(rep * 100).toFixed(1)}% of nearby word trigrams repeat`);
    if (typo) reasons.push("typographic polish (em dash / curly quotes) - weak signal");
    if (!reasons.length) reasons.push("no AI signals in this sentence");

    return {
      start, end, text: sent, word_count: nWords,
      score: round4(Math.min(Math.max(score, 0.0), 1.0)),
      low_confidence: nWords < MIN_WORDS_LOCAL && !artifact,
      reasons,
    };
  });

  const flagged = sentences.filter((s) => s.score >= 0.5 && !s.low_confidence).length;
  const note = wordList.length < 150
    ? `${wordList.length} words is below the 150-word floor: sentence scores are shown for navigation only and are not meaningful statistically.`
    : null;

  return { sentences, sentence_count: sentences.length, flagged, note };
}
