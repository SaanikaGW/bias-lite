/******************************************************
 * Bias-Lite (Simple Version)
 * - No auto fetch (so you can open index.html by double click)
 * - Click "Upload model JSON" to load bias_model.json
 * - Paste text, click Analyze, see highlights + score
 ******************************************************/

// The model will live here after you upload it
let MODEL = null;

// Get page elements
const inputEl = document.getElementById("input");
const analyzeBtn = document.getElementById("analyzeBtn");
const previewEl = document.getElementById("preview");
const suggestionsEl = document.getElementById("suggestions");
const probEl = document.getElementById("prob");
const meterEl = document.getElementById("meter");
const useModelEl = document.getElementById("useModel");
const useHighlighterEl = document.getElementById("useHighlighter");

const loadModelBtn = document.getElementById("loadModelBtn");
const modelFile = document.getElementById("modelFile");
const modelStatus = document.getElementById("modelStatus");
const modelBadge = document.getElementById("modelBadge");

// ---------------------------
// 1) MODEL UPLOAD (super simple)
// ---------------------------
loadModelBtn.addEventListener("click", function () {
  modelFile.click();
});

modelFile.addEventListener("change", async function (e) {
  const file = e.target.files && e.target.files[0];
  if (!file) return;
  const text = await file.text();
  try {
    MODEL = JSON.parse(text);
    modelStatus.textContent = "Model: loaded (" + file.name + ")";
    if (modelBadge) {
      modelBadge.textContent = "Model: loaded";
      modelBadge.style.borderColor = "#3a3";
    }
  } catch (err) {
    modelStatus.textContent = "Model: file not valid JSON";
    MODEL = null;
  }
});

// -----------------------------------------
// 2) VERY SIMPLE RULES FOR HIGHLIGHTING
//    (feel free to add or remove patterns)
// -----------------------------------------
const RULES = [
  { word: "female engineer", hint: "Use the role without gender unless relevant." },
  { word: "female leader", hint: "Use the role without gender unless relevant." },
  { word: "girls", hint: "Use when age matters; otherwise consider “women.”" },
  { word: "guys", hint: "Use a more inclusive word like “everyone” or “folks.”" },
  { word: "women shouldn't", hint: "Avoid generalizing about what a whole gender can do." },
  { word: "men shouldn't", hint: "Avoid generalizing about what a whole gender can do." },
  { word: "women can't", hint: "Avoid blanket limitations by gender." },
  { word: "men can't", hint: "Avoid blanket limitations by gender." },
  { word: "hysterical", hint: "Loaded descriptor; try neutral language." },
  { word: "bossy", hint: "Loaded descriptor; try specific behavior instead." },
  { word: "emotional", hint: "Loaded descriptor; be specific and fair." }
];

// This is a super simple highlighter that just replaces matching words
function simpleHighlight(rawText) {
  let outputHTML = rawText;
  const hits = [];

  // We do a basic, case-insensitive find/replace
  for (let i = 0; i < RULES.length; i++) {
    const rule = RULES[i];
    const term = rule.word;
    const termRegex = new RegExp(term.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "gi");

    if (termRegex.test(outputHTML)) {
      outputHTML = outputHTML.replace(termRegex, function (match) {
        hits.push({ match: match, hint: rule.hint });
        return '<mark>' + match + '</mark>';
      });
    }
  }

  return { html: outputHTML, hits: hits };
}

// ------------------------------------------------------
// 3) SIMPLE TOKENIZER so we can score with the tiny model
// ------------------------------------------------------
function tokenize(text) {
  // very basic: lowercase, replace non-letters with spaces, split
  const lowered = text.toLowerCase();
  const clean = lowered.replace(/[^a-z0-9\s]/g, " ");
  const parts = clean.split(/\s+/).filter(Boolean);
  return parts;
}

function makeBigrams(tokens) {
  const grams = [];
  for (let i = 0; i < tokens.length - 1; i++) {
    grams.push(tokens[i] + " " + tokens[i + 1]);
  }
  return grams;
}

// ---------------------------------------------------------
// 4) SIMPLE MODEL PROBABILITY (TF * IDF + logistic weights)
//    This is not fancy. It’s just a straightforward dot product.
// ---------------------------------------------------------
function modelProbability(text) {
  if (!MODEL) return null;

  const vocab = MODEL.vocab;   // { token: index }
  const idf = MODEL.idf;       // [idf per index]
  const coef = MODEL.coef;     // [weight per index]
  const b = MODEL.intercept || 0;

  // Get tokens and bigrams
  const one = tokenize(text);
  const two = makeBigrams(one);

  // Count terms that are in the vocab (raw counts only, super simple)
  const counts = {}; // token -> count
  const all = one.concat(two);
  for (let i = 0; i < all.length; i++) {
    const t = all[i];
    if (vocab.hasOwnProperty(t)) {
      counts[t] = (counts[t] || 0) + 1;
    }
  }

  // Dot product: sum( (count * idf[idx]) * coef[idx] ) + intercept
  let z = b;
  for (const t in counts) {
    const idx = vocab[t];
    const tfidf = counts[t] * idf[idx]; // no normalization to keep code simple
    z += coef[idx] * tfidf;
  }

  // Sigmoid to map to 0..1
  const p = 1 / (1 + Math.exp(-z));
  return p;
}

// ----------------------
// 5) ANALYZE BUTTON
// ----------------------
analyzeBtn.addEventListener("click", function () {
  const text = inputEl.value || "";

  // Highlight (rules)
  if (useHighlighterEl && useHighlighterEl.checked) {
    const result = simpleHighlight(text);
    previewEl.innerHTML = result.html || "<span class='muted'>Nothing to show.</span>";

    if (result.hits.length === 0) {
      suggestionsEl.textContent = "No flags from rules.";
    } else {
      // Show each hit with a small hint
      let s = "";
      for (let i = 0; i < result.hits.length; i++) {
        const h = result.hits[i];
        s += "<div class='hit'><div><strong>" + h.match + "</strong></div>" +
             "<div class='hint'>" + h.hint + "</div></div>";
      }
      suggestionsEl.innerHTML = s;
    }
  } else {
    previewEl.textContent = text;
    suggestionsEl.textContent = "Rule highlighter disabled.";
  }

  // Model score
  if (useModelEl && useModelEl.checked) {
    if (!MODEL) {
      probEl.textContent = "– (upload model)";
      meterEl.style.width = "0%";
    } else {
      const p = modelProbability(text) || 0;
      const pct = Math.round(p * 100);
      probEl.textContent = pct + "%";
      meterEl.style.width = pct + "%";
    }
  } else {
    probEl.textContent = "–";
    meterEl.style.width = "0%";
  }
});
