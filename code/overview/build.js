// Builds Fig. 1 (study overview) as HTML and renders it to PNG with Playwright/Chromium.
// Inputs: sel.json (from overview_data.py) and ../../outputs/numbers2.tex (from analysis2.py).
// Outputs: ../../outputs/figures/fig_overview.png (Fig. 1) and graphical_abstract.png.
// Icons: Tabler Icons (MIT licence); maths: KaTeX; text font: Inter (SIL OFL).
const fs = require("fs");
const path = require("path");
const katex = require("katex");
const { chromium } = require("playwright");

const HERE = __dirname;
const ICONS = path.join(HERE, "node_modules/@tabler/icons/icons");
const sel = JSON.parse(fs.readFileSync(path.join(HERE, "sel.json")));
// every number shown in the figures is read from outputs/numbers2.tex (written by analysis2.py)
const NUM = {};
for (const m of fs.readFileSync(path.join(HERE, "..", "..", "outputs", "numbers2.tex"), "utf8")
                 .matchAll(/\\newcommand\{\\(\w+)\}\{([^}]*)\}/g))
  NUM[m[1]] = m[2].replace(/\\\$/g, "");
const num = k => { if (!(k in NUM)) throw new Error("missing macro " + k); return NUM[k]; };
const r3 = k => { const v = parseFloat(num(k)); return (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(3); };

function icon(name, cls = "", filled = false) {
  let s = fs.readFileSync(path.join(ICONS, filled ? "filled" : "outline", name + ".svg"), "utf8");
  s = s.replace(/<svg[^>]*>/, m => m.replace(/class="[^"]*"/, `class="ic ${cls}"`).replace(/width="24"/, "").replace(/height="24"/, ""));
  return s;
}
const M = (tex, display = false) => katex.renderToString(tex, { displayMode: display, throwOnError: true, output: "html" });
// replace \( ... \) inline and \[ ... \] display math
function math(html) {
  html = html.replace(/\\\[([\s\S]+?)\\\]/g, (_, t) => M(t, true));
  return html.replace(/\\\(([\s\S]+?)\\\)/g, (_, t) => M(t, false));
}

// ---- mini heatmaps of real selection frequencies (10% budget, 50 x 5 runs) --
function heat(mat, title, col, labels = true) {
  const rows = 8, cw = 19, ch = 13.5, x0 = labels ? 40 : 4, y0 = 17;
  const W = x0 + 6 * cw + 4, H = y0 + rows * ch + 20;
  let s = `<svg class="heat" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">`;
  s += `<text x="${x0 + 3 * cw}" y="10" class="ht">${title}</text>`;
  for (let r = 0; r < rows; r++) {
    if (labels) s += `<text x="${x0 - 4}" y="${y0 + r * ch + ch * 0.72}" class="hl">${sel.keys[r]}</text>`;
    for (let c = 0; c < 6; c++) {
      const v = mat[r][c];
      s += `<rect x="${x0 + c * cw + 0.8}" y="${y0 + r * ch + 0.8}" width="${cw - 1.6}" height="${ch - 1.6}" rx="2.2" fill="${col}" fill-opacity="${0.07 + 0.93 * v}"/>`;
    }
  }
  for (let c = 0; c < 6; c++)
    s += `<text x="${x0 + c * cw + cw / 2}" y="${y0 + rows * ch + 11}" class="hx"><tspan class="tm">t</tspan><tspan class="ts" dy="2.2">${c}</tspan></text>`;
  return s + "</svg>";
}

// ---- availability grid (real design: infinity where c_t(e) = infinity) ------
function availGrid() {
  const n = sel.keys.length, cw = 18, ch = 12.4, x0 = 36, y0 = 4;
  const W = x0 + 6 * cw + 4, H = y0 + n * ch + 16;
  let s = `<svg class="avail" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">`;
  for (let r = 0; r < n; r++) {
    s += `<text x="${x0 - 4}" y="${y0 + r * ch + ch * 0.75}" class="hl">${sel.keys[r]}</text>`;
    for (let c = 0; c < 6; c++) {
      const a = sel.avail[r][c];
      s += `<rect x="${x0 + c * cw + 0.8}" y="${y0 + r * ch + 0.8}" width="${cw - 1.6}" height="${ch - 1.6}" rx="2" fill="${a ? "#c7d2fe" : "#fee2e2"}"/>`;
      if (!a) s += `<text x="${x0 + c * cw + cw / 2}" y="${y0 + r * ch + ch * 0.8}" class="inf">∞</text>`;
    }
  }
  for (let c = 0; c < 6; c++)
    s += `<text x="${x0 + c * cw + cw / 2}" y="${y0 + n * ch + 11}" class="hx"><tspan class="tm">t</tspan><tspan class="ts" dy="2.2">${c}</tspan></text>`;
  return s + "</svg>";
}

// ---- cost-to-97% bars (Table 6) --------------------------------------------
function costBars() {
  const d = [["\\ell_1\\text{-cost}", +num("CostNSLone"), "#94a3b8"], ["\\text{NB-VOI}", +num("CostNSVoi"), "#94a3b8"],
             ["\\text{RA-DSS myopic}", +num("CostNSMyo"), "#2563eb"], ["\\text{RA-DSS DP-exact}", +num("CostNSDpx"), "#1d4ed8"],
             ["\\text{Static reduct}", +num("CostNSStatic"), "#f97316"]];
  let s = "";
  for (const [lab, v, col] of d)
    s += `<div class="bar"><span class="bl">${M(lab)}</span><span class="bt"><span class="bf" style="width:${(v / 75) * 100}%;background:${col}"></span></span><span class="bv">$${v}</span></div>`;
  return s;
}

const I = icon;
const html = String.raw`<!doctype html><html><head><meta charset="utf-8">
<link rel="stylesheet" href="file://${HERE}/node_modules/katex/dist/katex.min.css">
<style>body::after{content:"";font-family:KaTeX_Math}</style>
<link rel="stylesheet" href="file://${HERE}/node_modules/@fontsource/inter/400.css">
<link rel="stylesheet" href="file://${HERE}/node_modules/@fontsource/inter/500.css">
<link rel="stylesheet" href="file://${HERE}/node_modules/@fontsource/inter/600.css">
<link rel="stylesheet" href="file://${HERE}/node_modules/@fontsource/inter/700.css">
<style>
:root{--ink:#0f172a;--mut:#475569;--line:#e2e8f0;--c1:#e11d48;--c2:#4f46e5;--c3:#0d9488;--c4:#d97706;--c5:#2563eb}
*{box-sizing:border-box;margin:0;padding:0}
body{width:1310px;background:#fff;font-family:Inter,sans-serif;color:var(--ink);font-size:15px;line-height:1.34}
#wrap{position:relative;padding:14px 16px 16px;background:linear-gradient(180deg,#f8fafc 0%,#f1f5f9 100%)}
.row{display:flex;align-items:stretch;gap:0}
.row+.row{margin-top:34px}
.card{background:#fff;border:1px solid var(--line);border-radius:14px;box-shadow:0 1px 2px rgba(15,23,42,.05),0 4px 14px rgba(15,23,42,.05);overflow:hidden;display:flex;flex-direction:column}
.hd{display:flex;align-items:center;gap:9px;padding:8px 12px;color:#fff;font-weight:700;font-size:16px;letter-spacing:.1px}
.hd .num{width:23px;height:23px;border-radius:50%;background:rgba(255,255,255,.95);display:flex;align-items:center;justify-content:center;font-size:13px;font-weight:700}
.hd .ic{width:21px;height:21px;margin-left:auto;stroke-width:1.9;opacity:.95}
.bd{padding:10px 12px 11px;flex:1;display:flex;flex-direction:column;gap:7px}
.arrow{width:30px;display:flex;align-items:center;justify-content:center;flex:none}
.arrow svg{width:26px;height:26px}
.k1 .hd{background:linear-gradient(90deg,#be123c,#e11d48)} .k1 .num{color:var(--c1)}
.k2 .hd{background:linear-gradient(90deg,#4338ca,#6366f1)} .k2 .num{color:var(--c2)}
.k3 .hd{background:linear-gradient(90deg,#0f766e,#14b8a6)} .k3 .num{color:var(--c3)}
.k4 .hd{background:linear-gradient(90deg,#b45309,#f59e0b)} .k4 .num{color:var(--c4)}
.k5 .hd{background:linear-gradient(90deg,#1d4ed8,#3b82f6)} .k5 .num{color:var(--c5)}
.lead{color:var(--mut);font-size:14px}
.chip{display:flex;align-items:center;gap:8px;padding:5px 8px;border-radius:9px;background:#fff1f2;border:1px solid #ffe4e6}
.chip .ic{width:22px;height:22px;flex:none;stroke:var(--c1);stroke-width:1.8}
.chips{display:grid;grid-template-columns:1fr;gap:5px}
.hero{display:flex;align-items:center;gap:12px;padding:4px 2px 2px}
.hero .ring{width:74px;height:74px;border-radius:50%;background:radial-gradient(circle at 35% 30%,#fff 0%,#ffe4e6 70%);border:2px solid #fecdd3;display:flex;align-items:center;justify-content:center;position:relative;flex:none}
.hero .ring .ic{width:44px;height:44px;stroke:#be123c;stroke-width:1.6}
.hero .ring .mon{position:absolute;right:-8px;top:-6px;width:30px;height:30px;border-radius:50%;background:#fff;border:1.5px solid #fecdd3;display:flex;align-items:center;justify-content:center}
.hero .ring .mon .ic{width:18px;height:18px;stroke:#e11d48}
.hero .ring .tube{position:absolute;left:-8px;bottom:-6px;width:30px;height:30px;border-radius:50%;background:#fff;border:1.5px solid #fecdd3;display:flex;align-items:center;justify-content:center}
.hero .ring .tube .ic{width:18px;height:18px;stroke:#e11d48}
.q{font-weight:600;font-size:15.5px;line-height:1.3}
.box{border-radius:10px;padding:7px 9px;background:#eef2ff;border:1px solid #e0e7ff}
.box.t{background:#f0fdfa;border-color:#ccfbf1}
.eq .katex-display{margin:2px 0}
.eq .katex{font-size:1.02em}
ul.ck{list-style:none;display:flex;flex-direction:column;gap:3.5px}
ul.ck li{display:flex;gap:6px;align-items:flex-start}
ul.ck li .ic{width:15px;height:15px;flex:none;margin-top:2px;stroke-width:2.4}
.k2 ul.ck .ic{stroke:var(--c2)} .k3 ul.ck .ic{stroke:var(--c3)}
.sub{display:flex;align-items:center;gap:7px;font-weight:700;font-size:15px;color:#0f766e;margin-bottom:3px}
.sub .ic{width:20px;height:20px;stroke:#0d9488;stroke-width:1.9}
.two{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.k3 .eq .katex{font-size:.9em}
.m2{display:flex;gap:10px;align-items:center}
.avail{flex:none}
.hl{font:500 10.6px Inter;fill:#334155;text-anchor:end}
.hx{font:500 10.4px Inter;fill:#64748b;text-anchor:middle}
.ht{font:600 11.6px Inter;fill:#0f172a;text-anchor:middle}
.tm{font-family:KaTeX_Math;font-style:italic;font-size:12px;fill:#334155}
.ts{font-family:KaTeX_Main;font-size:8.8px;fill:#334155}
.inf{font:400 11px KaTeX_Main;fill:#e11d48;text-anchor:middle}
.item{display:flex;gap:9px;align-items:flex-start}
.item .b{width:32px;height:32px;border-radius:9px;background:#fffbeb;border:1px solid #fde68a;display:flex;align-items:center;justify-content:center;flex:none}
.item .b .ic{width:20px;height:20px;stroke:#b45309;stroke-width:1.8}
.item b{font-weight:600}
.f3{display:grid;grid-template-columns:300px 250px 1fr;gap:12px;align-items:stretch}
.panel{border:1px solid var(--line);border-radius:10px;padding:7px 8px;background:#fcfdff;display:flex;flex-direction:column;gap:4px}
.pt{font-weight:600;font-size:14px;display:flex;align-items:center;gap:6px}
.pt .ic{width:16px;height:16px;stroke:var(--c5);stroke-width:2}
.heats{display:flex;gap:4px}
.heat{flex:none}
.tile{display:flex;align-items:center;gap:9px;padding:6px 8px;border-radius:9px;background:#eff6ff;border:1px solid #dbeafe}
.tile .v{font-weight:700;font-size:22px;color:#1d4ed8;min-width:78px;letter-spacing:-.3px;white-space:nowrap;flex:none}
.tile .d{font-size:13px;color:#334155;line-height:1.3}
.bar{display:grid;grid-template-columns:138px 1fr 36px;align-items:center;gap:6px;font-size:13.2px}
.bl{text-align:right;color:#334155}
.bl .katex{font-size:1.0em}
.bt{height:13px;background:#f1f5f9;border-radius:7px;overflow:hidden}
.bf{display:block;height:100%;border-radius:7px}
.bv{font-weight:600;font-size:13.2px;color:#0f172a}
.note{font-size:12.8px;color:var(--mut);line-height:1.32}
#conn{position:absolute;left:0;top:0;pointer-events:none}
</style></head><body><div id="wrap">

<div class="row">
 <div class="card k1" style="width:326px" id="c1">
  <div class="hd"><span class="num">1</span>Problem${I("building-hospital")}</div>
  <div class="bd">
   <div class="hero"><div class="ring">${I("bed")}<span class="mon">${I("heart-rate-monitor")}</span><span class="tube">${I("test-pipe")}</span></div>
     <div><div class="q">Which parameters should be observed, and when?</div>
     <div class="lead" style="margin-top:4px">Soft sets take the parameter set as given; in monitoring it must be chosen under cost.</div></div></div>
   <div class="chips">
    <div class="chip">${I("test-pipe-2")}<span>Parameters \(e\in E\): laboratory assays</span></div>
    <div class="chip">${I("clock")}<span>Periods \(t\in T\): six 4-hour windows</span></div>
    <div class="chip">${I("coin")}<span>Price \(c_t(e)\), with \(c_t(e)=\infty\) if unavailable</span></div>
    <div class="chip">${I("wallet")}<span>Budget \(B_t\) in every period</span></div>
    <div class="chip">${I("arrows-exchange")}<span>Switching cost \(\kappa(A_{t-1},A_t)\)</span></div>
   </div>
  </div>
 </div>
 <div class="arrow"><svg viewBox="0 0 24 24"><path d="M4 12h14m-5-5 5 5-5 5" stroke="#94a3b8" stroke-width="2.4" fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg></div>
 <div class="card k2" style="width:336px" id="c2">
  <div class="hd"><span class="num">2</span>Model: RA-DSS${I("math-function")}</div>
  <div class="bd">
   <div class="box eq">\[\mathcal S=(U,E,T,\{F_t\}_{t\in T},\mathcal R)\]\[\mathcal R=(c,B,\kappa),\quad \mathcal A_t=\{A\subseteq E^{\mathrm{av}}_t: c_t(A)\le B_t\}\]</div>
   <div class="m2">
    <ul class="ck" style="flex:1">
     <li>${I("circle-check")}<span>\(F_t:E\to\mathcal P(U)\) in each period</span></li>
     <li>${I("circle-check")}<span>\(\mathcal A_t\): an independence system</span></li>
     <li>${I("circle-check")}<span>\(\mathcal A_t=\mathcal P(D_t)\) iff \(c_t(D_t)\le B_t\)</span></li>
     <li>${I("circle-check")}<span>Output: a policy \(\mathbf A=(A_t)_{t\in T}\)</span></li>
    </ul>
    <div style="text-align:center">${availGrid()}<div class="note" style="font-size:10.2px;margin-top:-2px">case-study availability</div></div>
   </div>
  </div>
 </div>
 <div class="arrow"><svg viewBox="0 0 24 24"><path d="M4 12h14m-5-5 5 5-5 5" stroke="#94a3b8" stroke-width="2.4" fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg></div>
 <div class="card k3" style="flex:1" id="c3">
  <div class="hd"><span class="num">3</span>Selection: algorithms and guarantees${I("binary-tree")}</div>
  <div class="bd"><div class="two">
   <div>
    <div class="sub">${I("target-arrow")}Single period</div>
    <div class="box t eq">\[Q_t(A)=\sum\nolimits_{\pi}w(\pi)\,\varphi\big(|A\cap d_t(\pi)|\big)\]</div>
    <ul class="ck" style="margin-top:6px">
     <li>${I("circle-check")}<span>Normalised, monotone, submodular</span></li>
     <li>${I("circle-check")}<span>\(\textsf{NP}\)-complete to maximise over \(\mathcal A_t\)</span></li>
     <li>${I("circle-check")}<span>Greedy + singleton \(\ge 0.427\,\mathrm{OPT}_t\)</span></li>
     <li>${I("circle-check")}<span>Partial enumeration \(\ge (1-1/e)\,\mathrm{OPT}_t\)</span></li>
     <li>${I("circle-check")}<span>\(Q_t\) brackets the AUROC of the sum score</span></li>
     <li>${I("circle-check")}<span>Also submodular for graded (fuzzy) separation</span></li>
    </ul>
   </div>
   <div>
    <div class="sub">${I("route")}Multi-period</div>
    <div class="box t eq">\[J(\mathbf A)=\sum\nolimits_t Q_t(A_t)-\lambda\sum\nolimits_t\kappa(A_{t-1},A_t)\]</div>
    <ul class="ck" style="margin-top:6px">
     <li>${I("circle-check")}<span>Not submodular in general</span></li>
     <li>${I("circle-check")}<span>Exact DP by distance transform in \(O(|T|\,n\,2^n)\)</span></li>
     <li>${I("circle-check")}<span>Switching-aware Lagrangian certificate \(\mathrm{OPT}\le\mathrm{UB}\)</span></li>
     <li>${I("circle-check")}<span>Depleting budgets and warm-start re-selection</span></li>
    </ul>
   </div>
  </div></div>
 </div>
</div>

<div class="row">
 <div class="card k4" style="width:326px" id="c4">
  <div class="hd"><span class="num">4</span>Intensive-care case study${I("stethoscope")}</div>
  <div class="bd" style="gap:8px">
   <div class="item"><span class="b">${I("flask")}</span><span><b>14 laboratory assays</b> at Medicare fee-schedule prices ($4.29–$39.26)</span></div>
   <div class="item"><span class="b">${I("users")}</span><span><b>3,000 simulated admissions</b>, six windows, regime shift from early to late markers</span></div>
   <div class="item"><span class="b">${I("repeat")}</span><span><b>${num("NRepV")} replicates</b> × 5-fold cross-validation × 15 budgets</span></div>
   <div class="item"><span class="b">${I("scale")}</span><span><b>17 policies</b>: RA-DSS variants, soft-set and test-cost reducts, static panels, \(\ell_1\)-cost, NB-VOI, GA, …</span></div>
   <div class="item"><span class="b">${I("chart-dots")}</span><span><b>Paired statistics</b>: \(t\)-intervals, Wilcoxon and DeLong tests</span></div>
  </div>
 </div>
 <div class="arrow"><svg viewBox="0 0 24 24"><path d="M4 12h14m-5-5 5 5-5 5" stroke="#94a3b8" stroke-width="2.4" fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg></div>
 <div class="card k5" style="flex:1" id="c5">
  <div class="hd"><span class="num">5</span>Findings${I("chart-line")}</div>
  <div class="bd"><div class="f3">
   <div class="panel">
    <div class="pt">${I("layout-grid")}Selected panels at a 10% budget</div>
    <div class="heats">${heat(sel["Static reduct"], "Static reduct", "#f97316")}${heat(sel["RA-DSS DP-exact"], "RA-DSS DP-exact", "#1d4ed8", false)}</div>
    <div class="note">Selection frequency over 250 runs, eight cheapest assays: the RA-DSS panel follows the regime shift.</div>
   </div>
   <div class="panel" style="gap:6px">
    <div class="pt">${I("trending-up")}Gains and speed</div>
    <div class="tile"><span class="v">${r3("DAucDpxSeven")}</span><span class="d">AUROC of DP-exact over the static reduct, 7.5% budget</span></div>
    <div class="tile"><span class="v">${num("VsSexOpDpx")}</span><span class="d">over the exact static panel, mean of 5–30% budgets</span></div>
    <div class="tile"><span class="v">${num("TimeDtFourteen")} ms</span><span class="d">exact multi-period optimum, \(n=14\)</span></div>
   </div>
   <div class="panel" style="gap:5px">
    <div class="pt">${I("coin")}Cost per patient-day to reach 97% of full-panel AUROC</div>
    ${costBars()}
    <div class="note" style="margin-top:2px">Full panel: AUROC ${num("FullAUROC")} at $${num("FullCost")}. NB-VOI is at least as accurate, and \(\ell_1\)-cost is the cheapest route to 97%.</div>
   </div>
  </div></div>
 </div>
</div>
<svg id="conn"></svg>
</div>
<script>
function connect(){
  const w=document.getElementById('wrap').getBoundingClientRect();
  const a=document.getElementById('c3').getBoundingClientRect();
  const b=document.getElementById('c4').getBoundingClientRect();
  const x1=a.left+a.width*0.5-w.left, y1=a.bottom-w.top+2;
  const x2=b.left+b.width*0.5-w.left, y2=b.top-w.top-3;
  const ym=(y1+y2)/2, r=10;
  const s=document.getElementById('conn'); s.setAttribute('width',w.width); s.setAttribute('height',w.height);
  s.innerHTML='<path d="M'+x1+' '+y1+' V'+(ym-r)+' Q'+x1+' '+ym+' '+(x1-r)+' '+ym+' H'+(x2+r)+' Q'+x2+' '+ym+' '+x2+' '+(ym+r)+' V'+(y2-6)+'" stroke="#94a3b8" stroke-width="2.4" fill="none" stroke-linecap="round" stroke-dasharray="0"/>'+
   '<path d="M'+(x2-5.5)+' '+(y2-9)+' L'+x2+' '+(y2-2)+' L'+(x2+5.5)+' '+(y2-9)+'" stroke="#94a3b8" stroke-width="2.4" fill="none" stroke-linecap="round" stroke-linejoin="round"/>';
}
document.fonts.ready.then(connect);
</script>
</body></html>`;


// ---- graphical abstract: 531 x 1328 (h x w), one row, readable at 5 x 13 cm --
const ARW = `<div class="arrow"><svg viewBox="0 0 24 24"><path d="M4 12h14m-5-5 5 5-5 5" stroke="#94a3b8" stroke-width="2.6" fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg></div>`;
const gaHtml = html
  .replace(/<body>[\s\S]*<\/body>/, String.raw`<body class="ga"><div id="wrap"><div class="row ga-row">
 <div class="card k1" style="width:268px">
  <div class="hd"><span class="num">1</span>Problem${I("building-hospital")}</div>
  <div class="bd">
   <div class="q">Which parameters should be observed, and when?</div>
   <div class="lead">Soft sets take the parameter set as given; in monitoring it must be chosen under cost.</div>
   <div class="chips">
    <div class="chip">${I("coin")}<span>Price \(c_t(e)\); \(\infty\) if unavailable</span></div>
    <div class="chip">${I("wallet")}<span>Budget \(B_t\) in every period</span></div>
    <div class="chip">${I("arrows-exchange")}<span>Switching cost \(\kappa(A_{t-1},A_t)\)</span></div>
   </div>
  </div>
 </div>${ARW}
 <div class="card k2" style="width:278px">
  <div class="hd"><span class="num">2</span>Model: RA-DSS${I("math-function")}</div>
  <div class="bd">
   <div class="box eq">\[\mathcal S=(U,E,T,\{F_t\}_{t\in T},\mathcal R)\]\[\mathcal A_t=\{A\subseteq E^{\mathrm{av}}_t: c_t(A)\le B_t\}\]</div>
   <ul class="ck">
    <li>${I("circle-check")}<span>A binding budget gives a feasible family that deactivation cannot express</span></li>
    <li>${I("circle-check")}<span>Unavailability is the price \(c_t(e)=\infty\)</span></li>
    <li>${I("circle-check")}<span>Output: a policy <span style="white-space:nowrap">\(\mathbf A=(A_t)_{t\in T}\)</span></span></li>
   </ul>
  </div>
 </div>${ARW}
 <div class="card k3" style="width:312px">
  <div class="hd"><span class="num">3</span>Selection theory${I("binary-tree")}</div>
  <div class="bd">
   <ul class="ck">
    <li>${I("circle-check")}<span>\(Q_t\) submodular for crisp and graded (fuzzy) separation</span></li>
    <li>${I("circle-check")}<span>\(\textsf{NP}\)-complete to maximise</span></li>
    <li>${I("circle-check")}<span>Greedy \(\ge0.427\,\mathrm{OPT}_t\); partial enumeration \(\ge(1-1/e)\,\mathrm{OPT}_t\)</span></li>
    <li>${I("circle-check")}<span>Multi-period: exact DP in \(O(|T|\,n\,2^n)\)</span></li>
    <li>${I("circle-check")}<span>Switching-aware Lagrangian certificate</span></li>
   </ul>
  </div>
 </div>${ARW}
 <div class="card k5" style="flex:1">
  <div class="hd"><span class="num">4</span>ICU case study${I("chart-line")}</div>
  <div class="bd">
   <div class="lead" style="color:#334155">14 assays at Medicare prices; simulated cohorts, ${num("NRepV")} replicates</div>
   <div class="heats" style="justify-content:center">${heat(sel["Static reduct"], "Static reduct", "#f97316")}${heat(sel["RA-DSS DP-exact"], "RA-DSS DP-exact", "#1d4ed8", false)}</div>
   <div class="tiles2">
    <div class="tile"><span class="v">${num("VsSexOpDpx")}</span><span class="d">AUROC over the exact static panel, 5–30% budgets</span></div>
    <div class="tile"><span class="v">${num("TimeDtFourteen")} ms</span><span class="d">exact multi-period optimum, \(n=14\)</span></div>
   </div>
   <div class="note">NB-VOI is at least as accurate; \(\ell_1\)-cost reaches 97% of full-panel AUROC most cheaply.</div>
  </div>
 </div>
</div></div></body>`)
  .replace("</style>", `.ga{width:1328px}
.ga #wrap{height:531px;padding:14px 14px}
.ga-row{height:503px}
.ga .hd{font-size:20px;padding:10px 13px}
.ga .hd .ic{width:23px;height:23px}
.ga .bd{gap:12px;padding:13px 13px}
.ga .q{font-size:20px;line-height:1.28}
.ga .lead{font-size:17px}
.ga .chips{gap:9px}
.ga .chip{padding:8px 9px}
.ga .chip > span, .ga ul.ck li > span{font-size:17.5px;line-height:1.33}
.ga ul.ck{gap:13px}
.ga ul.ck li .ic{width:18px;height:18px;margin-top:3px}
.ga .box.eq .katex{font-size:1.1em}
.ga .tiles2{display:grid;grid-template-columns:1fr;gap:7px}
.ga .tile{padding:6px 9px}
.ga .tile .v{font-size:24px;min-width:92px}
.ga .tile{gap:12px}
.ga .tile .d{font-size:15px}
.ga .note{font-size:15px}
.ga .heat{zoom:1.02}
</style>`);

(async () => {
  const out = math(html);
  fs.writeFileSync(path.join(HERE, "overview.html"), out);
  fs.writeFileSync(path.join(HERE, "graphical_abstract.html"), math(gaHtml));
  // CHROMIUM_PATH lets the figure be rendered with an existing Chromium build
  const browser = await chromium.launch(process.env.CHROMIUM_PATH ? { executablePath: process.env.CHROMIUM_PATH } : {});
  const page = await browser.newPage({ viewport: { width: 1310, height: 900 }, deviceScaleFactor: 3 });
  await page.goto("file://" + path.join(HERE, "overview.html"));
  await page.evaluate(() => document.fonts.ready);
  await page.waitForTimeout(400);
  const el = await page.$("#wrap");
  await el.screenshot({ path: path.join(HERE, "..", "..", "outputs", "figures", "fig_overview.png") });
  // graphical abstract, 1328 x 531 CSS px rendered at 2x (2656 x 1062 px)
  const ga = await browser.newPage({ viewport: { width: 1328, height: 531 }, deviceScaleFactor: 2 });
  await ga.goto("file://" + path.join(HERE, "graphical_abstract.html"));
  await ga.evaluate(() => document.fonts.ready);
  await ga.waitForTimeout(400);
  const g = await ga.$("#wrap");
  await g.screenshot({ path: path.join(HERE, "..", "..", "outputs", "figures", "graphical_abstract.png") });
  await browser.close();
})();
