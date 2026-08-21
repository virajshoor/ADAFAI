import {
  analyze,
  analyzeSpans,
  AI_TELL_PHRASES,
  MIN_WORDS_FOR_VERDICT,
} from "./detector.js";

const $ = (id) => document.getElementById(id);
const input = $("input");
const results = $("results");

const SAMPLE_AI = `In today's world, it is important to note that technology plays a pivotal role in shaping the intricate landscape of modern education. Furthermore, this cutting-edge report will delve into the myriad ways in which artificial intelligence can bolster learning outcomes and unlock the potential of every student. Moreover, it is crucial to underscore the robust interplay between educators and these seamlessly integrated tools. The rich tapestry of digital resources stands as a testament to human ingenuity, and navigating the complexities of this paradigm shift requires a holistic approach. When it comes to assessment, these platforms facilitate meticulous feedback at unprecedented scale, illuminating pathways that were previously obscured. On the other hand, critics argue that over-reliance on such systems may erode critical thinking. Notably, the commendable efforts of early adopters serve as a beacon for institutions still hesitant to embrace change. In conclusion, the vibrant ecosystem of educational technology represents a game-changer, and its transformative impact underscores the importance of thoughtful implementation for generations to come.\u200b`;

const SAMPLE_HUMAN = `okay so we finally got the apartment sorted. Took THREE trips to the hardware store because I kept buying the wrong size anchors, which, yeah. my bad. The shelves are up now though and honestly they look decent?? not perfect, one of them tilts a little if you put anything heavy on the left side, but whatever. Jess came over saturday and we painted the kitchen that green color she kept texting me about. it dried darker than the sample card. neither of us love it. we're keeping it anyway because repainting sounds like a nightmare and rent's due. oh and the landlord still hasn't answered about the leak under the sink - classic. I put a bucket there, problem "solved". If you're visiting next month just know the couch is also the guest bed and it squeaks. bring coffee, the machine here is ancient and makes this horrible grinding noise but it works. mostly.`;

function wordCount(text) {
  return (text.toLowerCase().match(/[A-Za-z']+/g) ?? []).length;
}

function updateCount() {
  const n = wordCount(input.value);
  $("wordCount").textContent = `${n.toLocaleString()} word${n === 1 ? "" : "s"}`;
}

input.addEventListener("input", updateCount);
input.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") runAnalysis();
});

$("pasteBtn").addEventListener("click", async () => {
  try {
    input.value = await navigator.clipboard.readText();
    updateCount();
    input.focus();
  } catch {
    input.placeholder = "Clipboard blocked by the browser — paste manually with Ctrl/Cmd+V.";
  }
});
$("sampleAiBtn").addEventListener("click", () => { input.value = SAMPLE_AI; updateCount(); });
$("sampleHumanBtn").addEventListener("click", () => { input.value = SAMPLE_HUMAN; updateCount(); });
$("clearBtn").addEventListener("click", () => { input.value = ""; updateCount(); results.hidden = true; });
$("analyzeBtn").addEventListener("click", runAnalysis);

const VERDICT_CLASS = {
  "likely human": "human",
  "uncertain": "uncertain",
  "likely AI-generated": "ai",
  "insufficient evidence": "abstain",
};

async function runAnalysis() {
  const text = input.value;
  if (!text.trim()) {
    input.focus();
    input.placeholder = "Paste some text first.";
    return;
  }
  const btn = $("analyzeBtn");
  btn.disabled = true;
  btn.textContent = "Analyzing…";
  await new Promise((r) => setTimeout(r, 30)); // let the button repaint before heavy work

  const result = analyze(text);
  const spans = analyzeSpans(text);

  renderVerdict(result);
  renderWhy(result, text);
  renderSignals(result);
  renderDocument(text, spans);

  btn.disabled = false;
  btn.textContent = "Analyze";
  results.hidden = false;
  results.scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderVerdict(result) {
  const cls = VERDICT_CLASS[result.verdict] ?? "abstain";
  $("verdictBanner").className = `card verdict-banner ${cls}`;
  const badge = $("verdictBadge");
  badge.className = `verdict-badge ${cls}`;
  badge.textContent = result.verdict;
  $("verdictScore").textContent = `score ${result.score.toFixed(2)}`;
  $("verdictWords").textContent = `${result.word_count.toLocaleString()} words · 0 = human, 1 = AI`;
  $("verdictNote").textContent = result.note;

  const flags = result.provenance_flags ?? [];
  $("provenance").hidden = flags.length === 0;
  $("provenanceList").replaceChildren(...flags.map((f) => li(`! ${f}`)));
}

function renderWhy(result, text) {
  const ai = [];
  const human = [];
  const s = result.signals.stylometry;
  const u = result.signals.unicode;

  const low = text.toLowerCase();
  const hits = AI_TELL_PHRASES
    .map((p) => [p, low.split(p).length - 1])
    .filter(([, n]) => n > 0)
    .sort((a, b) => b[1] - a[1]);
  if (hits.length) {
    const shown = hits.slice(0, 6).map(([p, n]) => `${p} ×${n}`).join(", ");
    ai.push(`${s.ai_phrase_density_per_1000w} AI-tell phrases per 1,000 words: ${shown}${hits.length > 6 ? ", …" : ""}`);
  } else {
    human.push("no AI-tell phrases (delve, tapestry, pivotal, furthermore, …)");
  }

  if (s.sentence_count >= 2) {
    if (s.sentence_length_cv < 0.6) {
      ai.push(`uniform sentence lengths (CV ${s.sentence_length_cv}; human writing is typically 0.6+)`);
    } else {
      human.push(`varied sentence lengths (CV ${s.sentence_length_cv} — “burstiness”)`);
    }
  }

  if (s.lexical_diversity_mattr < 0.75) {
    ai.push(`low lexical diversity (MATTR ${s.lexical_diversity_mattr}; human baseline ~0.75+)`);
  } else {
    human.push(`healthy lexical diversity (MATTR ${s.lexical_diversity_mattr})`);
  }

  if (s.repeated_trigram_ratio > 0) {
    ai.push(`${(s.repeated_trigram_ratio * 100).toFixed(1)}% of word trigrams repeat verbatim`);
  } else {
    human.push("no repeated word trigrams");
  }

  for (const f of u.flags) ai.push(`provenance artifact: ${f}`);
  if (Object.keys(u.invisible_characters).length === 0) human.push("no invisible characters or lookalike glyphs");

  const t = u.typography;
  if (t.curly_quote_ratio !== null) {
    if (t.curly_quote_ratio > 0.9) ai.push(`${Math.round(t.curly_quote_ratio * 100)}% curly quotes (keyboards produce straight ones — weak signal, autocorrect does this too)`);
    else human.push("plain keyboard typography");
  }

  if (result.word_count < MIN_WORDS_FOR_VERDICT) {
    ai.length = 0;
    ai.push(`below the ${MIN_WORDS_FOR_VERDICT}-word floor — not enough text for any statistical claim`);
  }
  if (!ai.length) ai.push("nothing — no signal fired above its threshold");
  if (!human.length) human.push("nothing stood out as human-typical");

  $("whyAi").replaceChildren(...ai.map(li));
  $("whyHuman").replaceChildren(...human.map(li));
}

function renderSignals(result) {
  const s = result.signals.stylometry;
  const u = result.signals.unicode;
  $("stylometryScore").textContent = s.score.toFixed(2);
  $("unicodeScore").textContent = u.score.toFixed(2);

  $("stylometryMetrics").replaceChildren(
    ...Object.entries({
      "lexical diversity (MATTR)": s.lexical_diversity_mattr,
      "sentence-length CV": s.sentence_length_cv,
      "repeated trigrams": s.repeated_trigram_ratio,
      "AI-tell phrases / 1,000w": s.ai_phrase_density_per_1000w,
      "punctuation entropy": s.punctuation_entropy,
      sentences: s.sentence_count,
    }).flatMap(([k, v]) => [el("dt", k), el("dd", v)]),
  );

  const uRows = [
    ["curly quotes", `${u.typography.curly_quotes} of ${u.typography.curly_quotes + u.typography.straight_quotes}`],
    ["em dashes / 1,000w", u.typography.em_dashes_per_1000w],
    ["ellipsis char", u.typography.ellipsis_char],
    ["invisible characters", nameCounts(u.invisible_characters)],
    ["exotic spaces", nameCounts(u.exotic_spaces)],
    ["homoglyph lookalikes", nameCounts(u.homoglyphs)],
    ["scripts", (u.scripts ?? []).join(", ") || "—"],
  ];
  $("unicodeMetrics").replaceChildren(...uRows.flatMap(([k, v]) => [el("dt", k), el("dd", v)]));
}

function nameCounts(obj) {
  const entries = Object.entries(obj);
  if (!entries.length) return "none";
  return entries.map(([name, v]) => `${name} ×${v.count}`).join("; ");
}

let flaggedEls = [];
let flaggedIdx = -1;

function renderDocument(text, spans) {
  const doc = $("document");
  const frag = document.createDocumentFragment();
  let pos = 0;
  flaggedEls = [];
  flaggedIdx = -1;

  for (const s of spans.sentences) {
    if (s.start > pos) frag.appendChild(document.createTextNode(text.slice(pos, s.start)));
    const span = el("span", text.slice(s.start, s.end));
    span.className = `sent ${highlightClass(s)}`;
    span.dataset.score = s.score;
    span.title = `score ${s.score.toFixed(2)} — click for why`;
    span.addEventListener("click", () => showSentenceDetail(span, s));
    if (s.score >= 0.5 && !s.low_confidence) flaggedEls.push(span);
    frag.appendChild(span);
    pos = s.end;
  }
  if (pos < text.length) frag.appendChild(document.createTextNode(text.slice(pos)));
  doc.replaceChildren(frag);
  $("sentenceDetail").hidden = true;

  $("flaggedCount").textContent =
    `${spans.flagged} of ${spans.sentence_count} sentences flagged` + (spans.note ? ` · ${spans.note}` : "");
}

function highlightClass(s) {
  const artifact = s.reasons.some((r) => r.includes("invisible character") || r.includes("lookalike character") || r.includes("exotic space"));
  if (s.low_confidence) return "hl-lc";
  if (artifact) return "hl-high hl-artifact";
  if (s.score >= 0.6) return "hl-high";
  if (s.score >= 0.35) return "hl-med";
  if (s.score >= 0.15) return "hl-low";
  return "";
}

function showSentenceDetail(span, s) {
  document.querySelectorAll(".sent.selected").forEach((x) => x.classList.remove("selected"));
  span.classList.add("selected");
  const detail = $("sentenceDetail");
  detail.hidden = false;
  const conf = s.low_confidence ? " · low confidence (short sentence)" : "";
  detail.replaceChildren(
    el("h3", `Sentence score: ${s.score.toFixed(2)}${conf}`),
    el("p", s.text, "detail-text"),
    el("strong", "Why:"),
    elFrom("ul", s.reasons.map((r) => li(r))),
  );
}

$("nextFlaggedBtn").addEventListener("click", () => {
  if (!flaggedEls.length) return;
  flaggedIdx = (flaggedIdx + 1) % flaggedEls.length;
  flaggedEls[flaggedIdx].scrollIntoView({ behavior: "smooth", block: "center" });
  flaggedEls[flaggedIdx].click();
});

function el(tag, text, cls) {
  const node = document.createElement(tag);
  node.textContent = text;
  if (cls) node.className = cls;
  return node;
}

function elFrom(tag, children) {
  const node = document.createElement(tag);
  node.replaceChildren(...children);
  return node;
}

function li(text) {
  return el("li", text);
}
