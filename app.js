/***** Bias-Lite (auto-load + rewrites + download + before/after + popups) *****/
fetch("./bias_model.json")
let MODEL = null;

// Elements
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
const rewriteBtn = document.getElementById("rewriteBtn");
const downloadBtn = document.getElementById("downloadBtn");
const tabOriginal = document.getElementById("tabOriginal");
const tabImproved = document.getElementById("tabImproved");

// Popups
const modal = document.getElementById("modal");
const modalTitle = document.getElementById("modalTitle");
const modalBody = document.getElementById("modalBody");
document.getElementById("closeModal").onclick = () => modal.style.display = "none";
document.getElementById("whatBias").onclick = () => openInfo(
  "What is gender bias?",
  `
  <p>Gender bias is wording that treats people differently based on gender, often by assuming certain traits, roles, or abilities.</p>
  <p>Examples: “girls are better at helping,” “men are natural leaders,” or using “guys” to mean everyone.</p>
  `
);
document.getElementById("whyLanguage").onclick = () => openInfo(
  "Why language matters",
  `
  <p>Language shapes expectations. Biased wording can influence grades, hiring, and confidence. Small changes make writing fairer and clearer.</p>
  `
);
document.getElementById("examples").onclick = () => openInfo(
  "Examples to compare",
  `
  <p><b>Biased:</b> “The boys led while the girls took notes.”<br/>
     <b>Improved:</b> “Some team members led discussion while others recorded decisions.”</p>
  <p><b>Biased:</b> “She’s bossy.” → <b>Improved:</b> “She is assertive in leading discussions.”</p>
  `
);
function openInfo(title, html){ modalTitle.textContent = title; modalBody.innerHTML = html; modal.style.display = "flex"; }

// Auto-load model (works on GitHub Pages)
fetch("./bias_model.json")
  .then(r => { if(!r.ok) throw new Error("fetch"); return r.json(); })
  .then(j => {
    MODEL = j;
    modelStatus.textContent = "Model: loaded";
    modelBadge.textContent = "Model: loaded";
    modelBadge.style.borderColor = "#3a3";
  })
  .catch(() => {
    modelStatus.textContent = "Model: not loaded (click Upload)";
    loadModelBtn.classList.remove("hidden");
  });

// Upload fallback (local file://)
loadModelBtn?.addEventListener("click", () => modelFile.click());
modelFile?.addEventListener("change", async e => {
  const f = e.target.files?.[0]; if (!f) return;
  try {
    MODEL = JSON.parse(await f.text());
    modelStatus.textContent = `Model: loaded (${f.name})`;
    modelBadge.textContent = "Model: loaded";
    modelBadge.style.borderColor = "#3a3";
  } catch {
    modelStatus.textContent = "Model: file not valid JSON";
    MODEL = null;
  }
});

/* ----------------- Rule set with simple rewrites ----------------- */
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

/* --------- Highlight + collect hits + build improved text --------- */
function analyzeWithRules(rawText){
  let html = rawText;
  let improved = rawText;
  const hits = [];

  for (let i=0;i<RULES.length;i++){
    const rule = RULES[i];
    const term = rule.word;
    const esc = term.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const re = new RegExp(esc, "gi");

    if (re.test(html)) {
      // highlight in html
      html = html.replace(re, m => `<mark>${m}</mark>`);
    }
    if (re.test(improved)) {
      // simple rewrite in improved text
      improved = improved.replace(re, rule.replace);
      hits.push({ match: term, hint: rule.hint, replacement: rule.replace });
    }
  }
  return { html, improved, hits };
}

/* ----------------- Tokenizer + simple model score ----------------- */
function tokenize(text){
  const lowered = text.toLowerCase();
  const clean = lowered.replace(/[^a-z0-9\s]/g, " ");
  return clean.split(/\s+/).filter(Boolean);
}
function bigrams(tokens){
  const out=[]; for(let i=0;i<tokens.length-1;i++) out.push(tokens[i]+" "+tokens[i+1]); return out;
}
function modelProbability(text){
  if(!MODEL) return null;
  const vocab = MODEL.vocab, idf = MODEL.idf, coef = MODEL.coef, b = MODEL.intercept||0;
  const one = tokenize(text), two = bigrams(one);
  const all = one.concat(two);
  const counts = {};
  for(let i=0;i<all.length;i++){
    const t = all[i];
    if(Object.prototype.hasOwnProperty.call(vocab,t)) counts[t]=(counts[t]||0)+1;
  }
  let z=b;
  for(const t in counts){
    const idx=vocab[t];
    const tfidf = counts[t]*idf[idx];
    z += coef[idx]*tfidf;
  }
  return 1/(1+Math.exp(-z));
}

/* ----------------- UI: Analyze, Tabs, Rewrite, Download ----------------- */
let lastOriginalHTML = "";      // highlighted original
let lastImprovedText = "";      // plain text after rewrites
let showing = "original";       // which tab is active

function renderPreview(){
  if(showing==="original"){
    previewEl.innerHTML = lastOriginalHTML || "<span class='muted'>Nothing to show.</span>";
    tabOriginal.classList.add("active"); tabImproved.classList.remove("active");
  } else {
    // show improved text (plain)
    previewEl.textContent = lastImprovedText || "No improved text yet. Click “Rewrite text”.";
    tabImproved.classList.add("active"); tabOriginal.classList.remove("active");
  }
}

tabOriginal.onclick = () => { showing="original"; renderPreview(); };
tabImproved.onclick = () => { showing="improved"; renderPreview(); };

analyzeBtn.addEventListener("click", function (){
  const text = inputEl.value || "";

  // Rule analysis
  const res = analyzeWithRules(text);
  lastOriginalHTML = useHighlighterEl.checked ? res.html : text;
  lastImprovedText = res.improved;

  // Suggestions list
  if(res.hits.length===0){
    suggestionsEl.textContent = "No flags from rules.";
  } else {
    suggestionsEl.innerHTML = res.hits.map(h => `
      <div class="hit">
        <div><strong>${h.match}</strong> → <em>${h.replacement}</em></div>
        <div class="hint">${h.hint}</div>
      </div>
    `).join("");
  }

  // Model probability (on ORIGINAL text, simple)
  if(useModelEl.checked && MODEL){
    const p = modelProbability(text) || 0;
    const pct = Math.round(p*100);
    probEl.textContent = pct + "%";
    meterEl.style.width = pct + "%";
  } else {
    probEl.textContent = "–"; meterEl.style.width="0%";
  }

  // Default to Original tab after analyze
  showing = "original";
  renderPreview();
});

// One-click rewrite: replace in the editor with improved version
rewriteBtn.addEventListener("click", function(){
  if(!inputEl.value){ return; }
  const res = analyzeWithRules(inputEl.value);
  inputEl.value = res.improved;
  lastImprovedText = res.improved;
  showing = "improved";
  renderPreview();
});

// Download improved version as .txt
downloadBtn.addEventListener("click", function(){
  const text = lastImprovedText || inputEl.value || "";
  const blob = new Blob([text], {type:"text/plain"});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "biaslite_improved.txt";
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
});
