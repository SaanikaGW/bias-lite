/***** Bias-Lite: Before/After Tabs *****/

let MODEL = null;

// elements
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
const tabOriginal = document.getElementById("tabOriginal");
const tabImproved = document.getElementById("tabImproved");

// auto-load model (works on GitHub Pages)
fetch("./bias_model.json")
  .then(function(r){ if(!r.ok) throw new Error("fetch failed"); return r.json(); })
  .then(function(j){
    MODEL = j;
    modelStatus.textContent = "Model: loaded";
    modelBadge.textContent = "Model: loaded";
    modelBadge.style.borderColor = "#3a3";
  })
  .catch(function(){
    modelStatus.textContent = "Model: not loaded (click Upload)";
    loadModelBtn.classList.remove("hidden");
  });

// upload fallback
loadModelBtn.addEventListener("click", function(){ modelFile.click(); });
modelFile.addEventListener("change", function(e){
  const f = e.target.files && e.target.files[0];
  if (!f) return;
  f.text().then(function(t){
    try{
      MODEL = JSON.parse(t);
      modelStatus.textContent = "Model: loaded (" + f.name + ")";
      modelBadge.textContent = "Model: loaded";
      modelBadge.style.borderColor = "#3a3";
    } catch(err){
      modelStatus.textContent = "Model: file not valid JSON";
      MODEL = null;
    }
  });
});

/* ---------- rules with simple replacements ---------- */
const RULES = [
  { word: "female engineer", hint: "Use the role without gender unless relevant.", replace: "engineer" },
  { word: "female leader",   hint: "Use the role without gender unless relevant.", replace: "leader" },
  { word: "girls",           hint: "Use when age matters; otherwise consider “women.”", replace: "women" },
  { word: "guys",            hint: "Use a more inclusive word like “everyone” or “folks.”", replace: "everyone" },
  { word: "women shouldn't", hint: "Avoid generalizations about a group.", replace: "people shouldn't" },
  { word: "men shouldn't",   hint: "Avoid generalizations about a group.", replace: "people shouldn't" },
  { word: "women can't",     hint: "Avoid blanket limitations by gender.", replace: "people can't" },
  { word: "men can't",       hint: "Avoid blanket limitations by gender.", replace: "people can't" },
  { word: "hysterical",      hint: "Loaded descriptor; try neutral language.", replace: "overwhelmed" },
  { word: "bossy",           hint: "Loaded descriptor; try specific behavior instead.", replace: "assertive" },
  { word: "emotional",       hint: "Loaded descriptor; be specific and fair.", replace: "passionate" }
];

/* ---------- highlight + improved text ---------- */
function analyzeWithRules(rawText){
  let html = rawText;
  let improved = rawText;
  const hits = [];

  for (let i = 0; i < RULES.length; i++) {
    const rule = RULES[i];
    const esc = rule.word.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const re = new RegExp(esc, "gi");

    if (re.test(html)) {
      html = html.replace(re, function(m){ return "<mark>" + m + "</mark>"; });
    }
    if (re.test(improved)) {
      improved = improved.replace(re, rule.replace);
      hits.push({ match: rule.word, hint: rule.hint, replacement: rule.replace });
    }
  }
  return { html: html, improved: improved, hits: hits };
}

/* ---------- tokenizer + simple model score ---------- */
function tokenize(text){
  const lowered = text.toLowerCase();
  const clean = lowered.replace(/[^a-z0-9\s]/g, " ");
  const parts = clean.split(/\s+/).filter(function(x){ return x.length > 0; });
  return parts;
}
function bigrams(tokens){
  const out = [];
  for (let i = 0; i < tokens.length - 1; i++) out.push(tokens[i] + " " + tokens[i+1]);
  return out;
}
function modelProbability(text){
  if (!MODEL) return null;
  const vocab = MODEL.vocab, idf = MODEL.idf, coef = MODEL.coef, b = MODEL.intercept || 0;

  const one = tokenize(text);
  const two = bigrams(one);
  const all = one.concat(two);

  const counts = {};
  for (let i = 0; i < all.length; i++) {
    const t = all[i];
    if (Object.prototype.hasOwnProperty.call(vocab, t)) {
      if (!counts[t]) counts[t] = 0;
      counts[t] += 1;
    }
  }
  let z = b;
  for (const t in counts) {
    const idx = vocab[t];
    const tfidf = counts[t] * idf[idx];
    z += coef[idx] * tfidf;
  }
  const p = 1 / (1 + Math.exp(-z));
  return p;
}

/* ---------- tabs + render ---------- */
let lastOriginalHTML = "";
let lastImprovedText = "";
let showing = "original";

function renderPreview(){
  if (showing === "original") {
    previewEl.innerHTML = lastOriginalHTML || "<span class='muted'>Nothing to show.</span>";
    tabOriginal.classList.add("active");
    tabImproved.classList.remove("active");
  } else {
    if (lastImprovedText && lastImprovedText.length > 0) {
      previewEl.textContent = lastImprovedText;
    } else {
      previewEl.textContent = "No improved text yet. Click Analyze.";
    }
    tabImproved.classList.add("active");
    tabOriginal.classList.remove("active");
  }
}
tabOriginal.addEventListener("click", function(){ showing = "original"; renderPreview(); });
tabImproved.addEventListener("click", function(){ showing = "improved"; renderPreview(); });

/* ---------- analyze button ---------- */
analyzeBtn.addEventListener("click", function(){
  const text = inputEl.value || "";

  // rules
  const res = analyzeWithRules(text);
  if (useHighlighterEl && useHighlighterEl.checked) {
    lastOriginalHTML = res.html;
  } else {
    lastOriginalHTML = text;
  }
  lastImprovedText = res.improved;

  // suggestions list
  if (res.hits.length === 0) {
    suggestionsEl.textContent = "No flags from rules.";
  } else {
    let s = "";
    for (let i = 0; i < res.hits.length; i++) {
      const h = res.hits[i];
      s += '<div class="hit"><div><strong>' + h.match + "</strong> → <em>" +
           h.replacement + "</em></div><div class='hint'>" + h.hint + "</div></div>";
    }
    suggestionsEl.innerHTML = s;
  }

  // score
  if (useModelEl && useModelEl.checked && MODEL) {
    const p = modelProbability(text) || 0;
    const pct = Math.round(p * 100);
    probEl.textContent = pct + "%";
    meterEl.style.width = pct + "%";
  } else {
    probEl.textContent = "–";
    meterEl.style.width = "0%";
  }

  // show Original after analyze
  showing = "original";
  renderPreview();
});
