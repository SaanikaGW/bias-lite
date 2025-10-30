/*
 Bias-Lite
  My gender bias detection web app.
  It detects biased phrasing and gives rewrite suggestions,
  along with a rough probability score for bias.

  What I used to learn this:
  - MDN docs for fetch(), JSON, and DOM
  - Stack Overflow for file uploads
  - W3Schools and FreeCodeCamp for RegEx and event listeners
  - Khan Academy for understanding sigmoid and probabilities
  - CodePen examples for tabs and class toggling */

let MODEL = null; // Model will be stored here after it loads

// 1) CONNECTING HTML ELEMENT,Learned how to do this from MDN’s DOM documentation. */

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

/* 2) Automatic Model Loading
   Uses fetch() to try loading bias_model.json automatically.
   I learned that fetch() returns a Promise, so I have to
   handle it with .then() and .catch().
 */

fetch("./bias_model.json")
  .then(function (r) {
    // Check if file was found
    if (!r.ok) throw new Error("fetch failed");
    // Convert response into usable JSON
    return r.json();
  })
  .then(function (j) {
    MODEL = j; // Store model data globally
    // Update UI to show success
    modelStatus.textContent = "Model: loaded";
    modelBadge.textContent = "Model: loaded";
    modelBadge.style.borderColor = "#3a3";
  })
  .catch(function () {
    // If this fails (common with local HTML files)
    modelStatus.textContent = "Model: not loaded (click Upload)";
    // Show upload button for manual option
    loadModelBtn.style.display = "inline-block";
  });

/* 3) Uploading Model Manually
   This lets the user load bias_model.json from their computer.
   FileReader examples on Stack Overflow helped me understand
   how to read file content with .text().
*/
loadModelBtn.addEventListener("click", function () {
  // Simulate a click on the hidden file input
  modelFile.click();
});

modelFile.addEventListener("change", function (e) {
  // e.target.files is a FileList, I only need the first file
  const f = e.target.files && e.target.files[0];
  if (!f) return; // user canceled
  f.text().then(function (t) {
    try {
      MODEL = JSON.parse(t); // convert string to object
      // Update UI
      modelStatus.textContent = "Model: loaded (" + f.name + ")";
      modelBadge.textContent = "Model: loaded";
      modelBadge.style.borderColor = "#3a3";
    } catch (err) {
      // JSON.parse will throw an error if file is invalid
      modelStatus.textContent = "Model: file not valid JSON";
      MODEL = null;
    }
  });
});

/*  4) rules for detecting biased phrases
   These are rule-based checks for common gendered expressions.
   Each has:
   - 'word': what to detect
   - 'hint': message explaining why
   - 'replace': suggested neutral term
*/

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

/*  5) analyzeWithRules()
  
   This function does 3 things:
   1. Finds biased words using regex
   2. Highlights them in <mark> tags
   3. Builds an “improved” version with replacements
*/
function analyzeWithRules(rawText) {
  let html = rawText;     // keeps highlighted version
  let improved = rawText; // keeps replacement version
  const hits = [];        // stores details for suggestions

  // Loop through each rule and apply its regex
  for (let i = 0; i < RULES.length; i++) {
    const rule = RULES[i];
    // Escape regex special characters (e.g., + or ?)
    const esc = rule.word.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const re = new RegExp(esc, "gi"); // g = global, i = ignore case

    // Check if word exists anywhere in the text
    if (re.test(html)) {
      // Wrap the match with a highlight
      html = html.replace(re, function (m) {
        // “m” is the actual matched text
        return "<mark>" + m + "</mark>";
      });
    }

    // For improved version, replace with neutral term
    if (re.test(improved)) {
      improved = improved.replace(re, rule.replace);
      // Save the result for showing hints later
      hits.push({
        match: rule.word,
        hint: rule.hint,
        replacement: rule.replace
      });
    }
  }

  // Return all 3 results together
  return { html: html, improved: improved, hits: hits };
}

/* 6) TOKENIZATION + BIGRAMS
   
   Tokenization breaks text into words.
   Bigrams are word pairs like “female engineer”.
   This helps the model understand short phrases.
*/
function tokenize(text) {
  // Convert to lowercase to avoid treating “Women” and “women” differently
  const lowered = text.toLowerCase();

  // Replace all non-letter characters with spaces
  // Learned from regex patterns on freeCodeCamp
  const clean = lowered.replace(/[^a-z0-9\s]/g, " ");

  // Split into array of words, remove empty ones
  const parts = clean.split(/\s+/).filter(function (x) {
    return x.length > 0;
  });

  return parts;
}

function bigrams(tokens) {
  const out = [];
  // Loop over tokens to build pairs
  // Example: ["female","engineer","works"] → ["female engineer","engineer works"]
  for (let i = 0; i < tokens.length - 1; i++) {
    out.push(tokens[i] + " " + tokens[i + 1]);
  }
  return out;
}

/*  7) modelProbability()
  
   Calculates the “bias probability” using a simplified
   logistic regression formula I recreated after learning
   about sigmoid and tf-idf weighting.
*/

function modelProbability(text) {
  if (!MODEL) return null;

  // Extract data from the loaded model
  const vocab = MODEL.vocab;  // word → index
  const idf = MODEL.idf;      // importance values
  const coef = MODEL.coef;    // weight per feature
  const b = MODEL.intercept || 0; // base offset

  // Prepare features
  const one = tokenize(text);  // single words
  const two = bigrams(one);    // word pairs
  const all = one.concat(two); // everything together

  // Count how often each word or phrase appears
  const counts = {};
  for (let i = 0; i < all.length; i++) {
    const t = all[i];
    // Only count if the token exists in vocab
    if (Object.prototype.hasOwnProperty.call(vocab, t)) {
      // If first time seeing this word, start at 0
      if (!counts[t]) counts[t] = 0;
      // Add 1 for each appearance
      counts[t] += 1;
    }
  }

  // Compute the weighted sum (dot product)
  let z = b; // start with intercept
  for (const t in counts) {
    const idx = vocab[t];         // find index
    const tfidf = counts[t] * idf[idx]; // weight by importance
    // Multiply by coefficient and add to total
    z += coef[idx] * tfidf;
  }

  // Sigmoid squashes number between 0 and 1
  // I double-checked the math on Khan Academy
  const p = 1 / (1 + Math.exp(-z));

  return p;
}

/* = 8) Tab System
   Shows either:
   - Original text with highlights, OR
   - Improved (clean) text
 */

let lastOriginalHTML = "";
let lastImprovedText = "";
let showing = "original"; // starts with original

function renderPreview() {
  if (showing === "original") {
    // Use innerHTML because we want highlights to render
    previewEl.innerHTML = lastOriginalHTML || "<span class='muted'>Nothing to show.</span>";
    tabOriginal.classList.add("active");
    tabImproved.classList.remove("active");
  } else {
    // Use textContent to prevent HTML formatting
    previewEl.textContent = lastImprovedText || "No improved text yet. Click Analyze.";
    tabImproved.classList.add("active");
    tabOriginal.classList.remove("active");
  }
}

// Tab click events
tabOriginal.addEventListener("click", function () {
  showing = "original";
  renderPreview();
});
tabImproved.addEventListener("click", function () {
  showing = "improved";
  renderPreview();
});

/* 9) Main ""Analyze"" Button 
   
   Runs everything:
   - Rule-based highlighting
   - Suggestions list
   - Bias probability scoring
 */
analyzeBtn.addEventListener("click", function () {
  // Grab text from textarea
  const text = inputEl.value || "";

  // Run rule-based detection
  const res = analyzeWithRules(text);

  // Update previews
  lastOriginalHTML = useHighlighterEl.checked ? res.html : text;
  lastImprovedText = res.improved;

  // Show suggestions or message if nothing was flagged
  if (res.hits.length === 0) {
    suggestionsEl.textContent = "No flags from rules.";
  } else {
    let s = "";
    for (let i = 0; i < res.hits.length; i++) {
      const h = res.hits[i];
      // I used template literals to make HTML cleaner
      s += `
        <div class="hit">
          <div><strong>${h.match}</strong> → <em>${h.replacement}</em></div>
          <div class='hint'>${h.hint}</div>
        </div>`;
    }
    suggestionsEl.innerHTML = s;
  }

  // Only calculate probability if model is toggled ON
  if (useModelEl.checked && MODEL) {
    // Compute probability and round to percentage
    const p = modelProbability(text) || 0;
    const pct = Math.round(p * 100);
    probEl.textContent = pct + "%";
    meterEl.style.width = pct + "%"; // fills progress bar
  } else {
    // No model or disabled → show placeholder
    probEl.textContent = "–";
    meterEl.style.width = "0%";
  }

  // Always start by showing highlighted original
  showing = "original";
  renderPreview();
});
