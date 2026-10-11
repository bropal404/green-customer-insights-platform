/* GreenInsights dashboard: bank view (overview, products, regions, memory, model checks) and customer view.
   Reads the local API served by scripts/serve_dashboard.py. No build step. */

const $ = (sel, root = document) => root.querySelector(sel);
const view = $("#view");
const state = { overview: null, list: { q: "", product: "", status: "with", offset: 0 }, charts: [] };

/* ---------- formatting ---------- */
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const num = (x, d = 0) => (x === null || x === undefined || Number.isNaN(x)) ? "–" : Number(x).toLocaleString("en-GB", { maximumFractionDigits: d, minimumFractionDigits: d });
const gbp = (x, d = 0) => (x === null || x === undefined) ? "–" : (x < 0 ? "−£" : "£") + num(Math.abs(x), d);
const pct = (x, d = 1) => (x === null || x === undefined) ? "–" : num(x * 100, d) + "%";
const kg = (x) => (x === null || x === undefined) ? "–" : (Math.abs(x) >= 1000 ? num(x / 1000, 1) + " t" : num(x, Math.abs(x) < 10 ? 1 : 0) + " kg");
const label = (s) => esc(String(s ?? "").replace(/_/g, " "));
const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1);
const PRODUCT_COLOURS = { GP01: "#1d4ed8", GP02: "#15803d", GP03: "#b45309", GP04: "#64748b", GP05: "#7c3aed", GP06: "#0f766e" };
const CATEGORY_COLOURS = { fuel: "#b91c1c", electricity: "#1d4ed8", gas: "#b45309", public_transport: "#15803d", home_improvement: "#7c3aed", automotive: "#334155", utilities: "#0f766e", other: "#94a3b8" };

function toast(msg) {
  const t = $("#toast"); t.textContent = msg; t.classList.add("show");
  clearTimeout(t._h); t._h = setTimeout(() => t.classList.remove("show"), 3200);
}
async function api(path, body) {
  const res = await fetch(path, body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {});
  if (!res.ok) throw new Error((await res.json().catch(() => ({}))).error || res.statusText);
  return res.json();
}
function clearCharts() { state.charts.forEach((c) => c.destroy()); state.charts = []; }
function chart(canvas, config) {
  if (!window.Chart) { canvas.parentElement.innerHTML = '<div class="note">Charts need the Chart.js script (internet); the tables below hold the same numbers.</div>'; return; }
  Chart.defaults.font.family = "Inter, sans-serif"; Chart.defaults.font.size = 11; Chart.defaults.color = "#64748b";
  state.charts.push(new Chart(canvas, config));
}
const PALETTE = ["#1d4ed8", "#15803d", "#b45309", "#7c3aed", "#0f766e", "#b91c1c", "#64748b", "#334155", "#db2777", "#0891b2"];
const chartBox = (id, title, sub = "", h = 260) => `<div class="card"><div class="card-h"><h3>${title}</h3>${sub ? `<span class="sub">${sub}</span>` : ""}</div><div class="card-b"><div class="chart-box" style="height:${h}px"><canvas id="${id}"></canvas></div></div></div>`;
const moneyTick = (v) => (v < 0 ? "−£" : "£") + (Math.abs(v) >= 1e6 ? num(Math.abs(v) / 1e6, 1) + "m" : Math.abs(v) >= 1e3 ? num(Math.abs(v) / 1e3, 0) + "k" : num(Math.abs(v)));
const fmtDay = (iso) => iso ? new Date(iso).toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" }) : "";
const windowText = () => { const d = state.overview.dates || {}; const s = new Date(d.training_cutoff); s.setDate(s.getDate() + 1); return `${fmtDay(s)} – ${fmtDay(d.live)}`; };
const afterCutoff = (m) => m > (state.overview.dates?.training_cutoff || "").slice(0, 7);
const monthLabel = (m) => { const [y, mm] = m.split("-"); return new Date(+y, +mm - 1, 1).toLocaleString("en-GB", { month: "short", year: "2-digit" }); };
function plot(id, type, labels, datasets, o = {}) {
  const el = document.getElementById(id); if (!el) return;
  const horizontal = !!o.horizontal, fmt = o.fmt || ((v) => num(v)), axis = horizontal ? "x" : "y";
  const scales = type === "doughnut" ? {} : {
    x: { stacked: !!o.stacked, grid: { display: horizontal, color: "#f1f5f9" }, ticks: horizontal ? { callback: (v) => fmt(v) } : { maxRotation: 0, autoSkip: labels.length > 12, font: { size: labels.length > 4 && labels.length <= 12 ? 10 : 11 } } },
    y: { stacked: !!o.stacked, grid: { display: !horizontal, color: "#f1f5f9" }, ticks: horizontal ? {} : { callback: (v) => fmt(v) }, beginAtZero: true },
  };
  if (o.y2) scales.y2 = { position: "right", grid: { display: false }, beginAtZero: true, ticks: { callback: (v) => (o.fmt2 || num)(v) } };
  const wrap = (l) => (typeof l === "string" && l.length > 10 && l.includes(" ")) ? [l.split(" ").slice(0, -1).join(" "), l.split(" ").slice(-1)[0]] : l;
  const shown = !horizontal && type !== "doughnut" && labels.length <= 12 ? labels.map(wrap) : labels;
  chart(el, { type, data: { labels: shown, datasets: datasets.map((d, i) => ({
      backgroundColor: d.color || (type === "doughnut" ? (o.colors || PALETTE) : PALETTE[i % PALETTE.length]),
      borderColor: d.color || PALETTE[i % PALETTE.length], borderRadius: type === "bar" ? 3 : 0, maxBarThickness: 40,
      pointRadius: 0, borderWidth: (d.type || type) === "line" ? 2 : (type === "doughnut" ? 1 : 0), tension: 0.25, ...d })) },
    options: { maintainAspectRatio: false, indexAxis: horizontal ? "y" : "x", cutout: type === "doughnut" ? "62%" : undefined,
      interaction: { mode: type === "doughnut" ? "nearest" : "index", intersect: false },
      plugins: { legend: { display: o.legend !== false && (datasets.length > 1 || type === "doughnut"), position: type === "doughnut" ? "right" : "bottom", labels: { boxWidth: 10, padding: 10 } },
        tooltip: { callbacks: { label: (c) => `${c.dataset.label || c.label}: ${(c.dataset.yAxisID === "y2" ? (o.fmt2 || num) : fmt)(type === "doughnut" ? c.parsed : c.parsed[axis])}` } } },
      scales } });
}
const pageHead = (title, sub, right = "") => `<div class="page-head"><div><h1>${title}</h1><p>${sub}</p></div>${right}</div>`;
const kpi = (labelText, value, hint = "", cls = "") => `<div class="kpi ${cls}"><div class="label">${labelText}</div><div class="value">${value}</div>${hint ? `<div class="hint">${hint}</div>` : ""}</div>`;
const pill = (text, cls) => `<span class="pill ${cls}">${esc(text)}</span>`;
function decisionPill(d) {
  if (!d) return pill("–", "grey");
  if (d === "recommended") return pill("recommended", "green");
  if (d.startsWith("human review")) return pill("human review", "amber");
  if (d.startsWith("blocked")) return pill(d.replace("blocked: ", "blocked · "), "red");
  if (d.startsWith("memory")) return pill(d, "blue");
  return pill(d, "grey");
}

/* ---------- router ---------- */
const PAGES = { overview: renderOverview, customers: renderCustomers, customer: renderCustomer, products: renderProducts, regions: renderRegions, value: renderValue, memory: renderMemory, models: renderModels };
async function route() {
  const [page, arg] = (location.hash.replace(/^#\//, "") || "overview").split("/");
  document.querySelectorAll("#nav a").forEach((a) => a.classList.toggle("active", a.dataset.page === (page === "customer" ? "customers" : page)));
  clearCharts();
  view.innerHTML = '<div class="empty">Loading…</div>';
  try {
    if (!state.overview) {
      state.overview = await api("/api/overview");
      const asOf = state.overview.insights?.as_of;
      if (asOf) $("#asof").textContent = `Synthetic data · as of ${asOf}`;
    }
    await (PAGES[page] || renderOverview)(arg);
    window.scrollTo(0, 0);
  } catch (e) {
    view.innerHTML = `<div class="card"><div class="empty">Could not load: ${esc(e.message)}</div></div>`;
  }
}
window.addEventListener("hashchange", route);

/* ---------- overview ---------- */
const STEPS = [
  ["Ingest", "FR1 · source systems", "data"], ["Categorise", "FR2 · 8 categories", "data"], ["Features", "Customer 360 + features", "data"],
  ["Propensity", "FR3 · LightGBM", "propensity"], ["Impact", "FR4 · carbon + money", "impact"], ["Recommend", "FR5–7 · thresholds + memory", "recommend"],
  ["Explain", "FR5 · template + LLM rewrite", "explain"], ["Insights", "FR6 · bank view", "insights"], ["Value", "customer + bank return", "value"],
];
function stepper(done) {
  return `<div class="stepper">${STEPS.map(([name, what, key], i) => {
    const ok = key === "data" || done.includes(key);
    return `<div class="step"><div class="dot ${ok ? "" : "todo"}">${ok && i === STEPS.length - 1 ? "✓" : i + 1}</div><div class="name" style="${ok ? "" : "color:var(--faint)"}">${name}</div><div class="what">${what}</div></div>`;
  }).join("")}</div>`;
}
function renderOverview() {
  const o = state.overview, ins = o.insights, ov = ins.overview, d = o.data || {};
  const products = ins.products, t = ins.trends;
  view.innerHTML = pageHead("Green opportunity across the portfolio", `${num(ov.customers)} customers · as of ${esc(ins.as_of)}`,
    `<a class="btn" href="#/customers" style="display:inline-flex;align-items:center;text-decoration:underline">Browse customers →</a>`) + `
    <div class="card"><div class="card-b" style="padding-bottom:6px"><div class="eyebrow">Pipeline progress</div>${stepper(o.done || [])}</div></div>
    <div class="card">
      <div class="status"><div class="led"></div><div><b>Pipeline run complete</b><span>${esc((d.ingestion_summary || [])[0] || "Source systems ingested and audited.")}</span></div></div>
      <div class="tiles">
        <div class="tile"><div class="k">Customers</div><div class="v">${num(d.customers)}</div></div>
        <div class="tile"><div class="k">Transactions</div><div class="v">${num(d.transactions)}</div></div>
        <div class="tile"><div class="k">Audit vs system of record</div><div class="v">${esc(d.audit_tables || "–")} tables exact</div></div>
        <div class="tile"><div class="k">Realism checks</div><div class="v">${esc(d.validation || "–")} passed</div></div>
      </div>
    </div>
    <div class="grid g-4" style="margin-bottom:24px">
      ${kpi("Customers with a suggestion", num(ov.customers_with_a_recommendation), `${num(ov.recommendations)} suggestions, at most ${num(o.config?.max_per_customer)} each`)}
      ${kpi("Expected take-ups", num(ov.expected_takeups), "sum of calibrated propensities", "blue")}
      ${kpi("Expected CO2e avoided", num(ov.expected_co2e_tonnes_per_year, 0) + " t/yr", `${num(ov.co2e_tonnes_per_year_if_all_shown_taken_up)} t/yr if every suggestion were taken up`, "good")}
      ${kpi("Expected customer savings", gbp(o.business?.portfolio?.customer?.expected_saving_gbp_per_year ?? ov.customer_savings_gbp_per_year_expected) + "/yr", "running costs at today's prices, net of any increases")}
    </div>
    <div class="grid g-2" style="margin-bottom:24px">${chartBox("c-sugg", "Suggestions by product", "shown + human review")}${chartBox("c-co2", "Expected CO2e avoided by product", "tonnes a year · propensity × avoided emissions")}</div>
    ${chartBox("c-takeups", "Green product take-ups per month", `new green products opened · observed window ${windowText()}`, 240)}
    ${chartBox("c-funnel", "Why products were or were not shown", "first rule that decided each customer × product", 300)}
    <div class="grid g-2" style="margin-bottom:24px">${chartBox("c-seg", "Customer segments", "share of customers")}${chartBox("c-segco2", "Expected CO2e avoided by segment", "tonnes a year")}</div>
    <div class="grid g-3">${ins.segments.map(segmentCard).join("")}</div>
    <div class="grid g-2" style="margin-top:24px">
      ${kpi("Human review", num(ov.human_review), "credit offers to customers with a vulnerability flag", "alert")}
      ${kpi("Median footprint", kg(ov.footprint_kg_median) + "/yr", "visible fuel and home energy per customer")}
    </div>`;
  const names = products.map((p) => p.name);
  plot("c-sugg", "doughnut", names, [{ data: products.map((p) => p.shown + p.human_review), label: "suggestions" }], { colors: Object.values(PRODUCT_COLOURS) });
  plot("c-co2", "bar", names, [{ data: products.map((p) => p.expected_co2e_tonnes_per_year), label: "t CO2e/yr", color: "#15803d" }], { fmt: (v) => num(v) + " t" });
  const win = t.months.map((m, i) => [m, i]).filter(([m]) => afterCutoff(m));
  if (t.takeups) plot("c-takeups", "bar", win.map(([m]) => monthLabel(m)), Object.entries(t.takeups).map(([k, v], i) => ({ label: k, data: win.map(([, j]) => v[j]), color: Object.values(PRODUCT_COLOURS)[i] })), { stacked: true });
  const keys = Object.keys(FUNNEL_COLOURS);
  plot("c-funnel", "bar", ins.funnel.map((f) => o.products[f.product]), keys.map((k) => ({ label: k, data: ins.funnel.map((f) => f[k] || 0), color: FUNNEL_COLOURS[k] })), { stacked: true, horizontal: true });
  plot("c-seg", "doughnut", ins.segments.map((x) => x.name.replace("Customers who ", "")), [{ data: ins.segments.map((x) => x.customers), label: "customers" }]);
  plot("c-segco2", "bar", ins.segments.map((x) => x.name.replace("Customers who ", "")), [{ data: ins.segments.map((x) => x.expected_co2e_tonnes_per_year), label: "t CO2e/yr", color: "#15803d" }], { horizontal: true, fmt: (v) => num(v) + " t" });
}
const FUNNEL_COLOURS = { recommended: "#15803d", "human review": "#b45309", blocked: "#b91c1c", "no marketing consent": "#94a3b8", "customer memory (paused or stopped)": "#1d4ed8", "impact cannot be estimated": "#fca5a5", "below the propensity threshold": "#cbd5e1", "low confidence": "#fde68a", "impact too small": "#a7f3d0", "routed to a more specific product": "#7c3aed", "lower priority than three other products": "#334155" };
function funnelBars(funnel) {
  const names = state.overview.products;
  const keys = Object.keys(FUNNEL_COLOURS);
  const rows = funnel.map((f) => {
    const total = keys.reduce((s, k) => s + (f[k] || 0), 0) || 1;
    return `<div class="bar-row" style="grid-template-columns:180px 1fr 80px"><span>${esc(names[f.product])}</span><div class="stack">${keys.map((k) => f[k] ? `<i title="${esc(k)}: ${num(f[k])}" style="width:${(100 * f[k] / total).toFixed(2)}%;background:${FUNNEL_COLOURS[k]}"></i>` : "").join("")}</div><span class="mono" style="text-align:right">${num(f.recommended || 0)}</span></div>`;
  }).join("");
  return rows + `<div class="legend">${keys.map((k) => `<span><i style="background:${FUNNEL_COLOURS[k]}"></i>${esc(k)}</span>`).join("")}</div>`;
}
function segmentCard(s) {
  return `<div class="kpi seg"><h4>${esc(s.name)}</h4><div class="meta">${num(s.customers)} customers · ${pct(s.share, 0)} of the book</div>
    <div class="facts" style="margin:0 0 8px"><span>Income <b>${gbp(s.median_income_monthly)}/mo</b></span><span>Fuel-based km <b>${num(s.median_annual_km)}</b></span><span>Footprint <b>${kg(s.footprint_kg_median)}</b></span></div>
    <div>${s.top_products.map((p) => `<span class="chip">${esc(p)}</span>`).join("")}</div>
    <div class="note">${num(s.expected_co2e_tonnes_per_year, 0)} t CO2e/yr expected · ${num(s.recommendations_per_customer, 2)} suggestions per customer</div></div>`;
}

/* ---------- customers ---------- */
async function renderCustomers() {
  const l = state.list, names = state.overview.products;
  const data = await api(`/api/customers?q=${encodeURIComponent(l.q)}&product=${l.product}&status=${l.status}&limit=25&offset=${l.offset}`);
  view.innerHTML = pageHead("Customers", `${num(data.total)} matching`) + `
    <div class="card">
      <div class="filters">
        <input type="search" id="q" placeholder="e.g. C100123, Leeds, North West…" value="${esc(l.q)}">
        <select id="product"><option value="">All products</option>${Object.entries(names).map(([k, v]) => `<option value="${k}" ${l.product === k ? "selected" : ""}>${esc(v)}</option>`).join("")}</select>
        <select id="status">${[["with", "With a suggestion"], ["review", "Human review"], ["none", "No suggestion"], ["", "Everyone"]].map(([k, v]) => `<option value="${k}" ${l.status === k ? "selected" : ""}>${v}</option>`).join("")}</select>
        <button class="btn primary" id="go">Search</button>
      </div>
      <div class="table-wrap"><table><thead><tr><th>Customer</th><th>Area</th><th class="num">Age</th><th>Income band</th><th class="num">Suggestions</th><th>Top suggestion</th><th class="num">Expected CO2e/yr</th><th></th></tr></thead><tbody>
      ${data.rows.map((r) => `<tr class="click" data-id="${esc(r.customer_id)}"><td class="mono">${esc(r.customer_id)}</td><td>${esc(r.la_name)}<div class="note">${esc(r.region)}</div></td><td class="num">${num(r.age)}</td><td>${label(r.declared_income_band)}</td><td class="num">${r.recommendations ? pill(r.recommendations, "green") : pill("0", "grey")}</td><td>${esc(r.top_product || "–")}</td><td class="num mono">${r.expected_co2e ? kg(r.expected_co2e) : "–"}</td><td>${r.review ? pill("review", "amber") : ""}</td></tr>`).join("") || `<tr><td colspan="8" class="empty">No customers match.</td></tr>`}
      </tbody></table></div>
      <div class="pager"><span>${num(data.total)} customers · showing ${data.total ? l.offset + 1 : 0}–${Math.min(l.offset + 25, data.total)}</span>
        <span><button class="btn small" id="prev" ${l.offset === 0 ? "disabled" : ""}>← Previous</button> <button class="btn small" id="next" ${l.offset + 25 >= data.total ? "disabled" : ""}>Next →</button></span></div>
    </div>`;
  const run = () => { l.q = $("#q").value.trim(); l.product = $("#product").value; l.status = $("#status").value; l.offset = 0; renderCustomers(); };
  $("#go").onclick = run; $("#q").onkeydown = (e) => e.key === "Enter" && run();
  $("#product").onchange = run; $("#status").onchange = run;
  $("#prev").onclick = () => { l.offset = Math.max(0, l.offset - 25); renderCustomers(); };
  $("#next").onclick = () => { l.offset += 25; renderCustomers(); };
  view.querySelectorAll("tr.click").forEach((tr) => tr.onclick = () => location.hash = `#/customer/${tr.dataset.id}`);
}

/* ---------- customer ---------- */
async function renderCustomer(id, tab = "recs", doc = null) {
  doc = doc || await api(`/api/customer/${id}`);
  const s = Object.fromEntries(doc.customer_360.map((x) => [x.section, x.fields]));
  const who = s.who_they_are || {}, where = s.where_they_live || {};
  const names = state.overview.products;
  view.innerHTML = pageHead(`Customer ${esc(id)}`, `${num(who.age)}-year-old in ${esc(where.la_name)} (${esc(where.region)}). ${cap(label(who.declared_employment_status))}, ${label(who.declared_residential_status)}, with the bank ${num(who.customer_tenure_years, 1)} years. Data up to ${esc(doc.as_of)}.`,
    `<span style="display:flex;gap:8px"><a class="btn link" href="app/#${esc(id)}" style="display:inline-flex;align-items:center">Open customer's view ↗</a><a class="btn link" href="#/customers" style="display:inline-flex;align-items:center">← Back to customers</a></span>`) + `
    <div class="card"><div class="status"><div class="led" style="background:${doc.recommendations.length ? "var(--green)" : "var(--faint)"}"></div><div><b>${doc.recommendations.length ? `${doc.recommendations.length} green suggestion${doc.recommendations.length > 1 ? "s" : ""}` : "No suggestion today"}</b><span>${doc.recommendations.length ? "Ranked by propensity × confidence × memory weight" : "See “Why not the others”"}</span></div></div>
      <div class="tiles">
        <div class="tile"><div class="k">Income band</div><div class="v">${label(who.declared_income_band)}</div></div>
        <div class="tile"><div class="k">Marketing consent</div><div class="v">${who.marketing_consent ? "yes" : "no"}</div></div>
        <div class="tile"><div class="k">Preferred channel</div><div class="v">${label(who.preferred_contact_channel)}</div></div>
        <div class="tile" style="${who.vulnerability_flag ? "background:var(--amber-soft);border-color:var(--amber-line)" : ""}"><div class="k">Vulnerability flag</div><div class="v" style="${who.vulnerability_flag ? "color:var(--amber)" : ""}">${who.vulnerability_flag ? "yes · human review" : "no"}</div></div>
      </div></div>
    <div class="tabs">${[["recs", "Suggestions"], ["why", "Why not the others"], ["c360", "Customer 360"], ["features", "Behaviour features"], ["memory", "Memory"]].map(([k, v]) => `<button data-tab="${k}" class="${tab === k ? "on" : ""}">${v}</button>`).join("")}</div>
    <div id="tab"></div>`;
  view.querySelectorAll(".tabs button").forEach((b) => b.onclick = () => renderCustomer(id, b.dataset.tab, doc));
  const t = $("#tab");
  if (tab === "recs") {
    t.innerHTML = doc.recommendations.length ? `<div class="grid g-${Math.min(doc.recommendations.length, 3) === 1 ? "2" : "2"}">${doc.recommendations.map(recCard).join("")}</div>` : `<div class="card"><div class="empty">Nothing is suggested to this customer today.</div></div>`;
    t.querySelectorAll("[data-rw]").forEach((b) => b.onclick = async () => {
      const product = b.dataset.rw;
      b.disabled = true; b.textContent = "Rewriting…";
      try {
        const r = await api(`/api/customer/${id}/explain`, { product });
        if (r.error) { toast(r.error); return; }
        if (r.passed) {
          t.querySelector(`[data-text="${product}"]`).textContent = r.text;
          t.querySelector(`[data-src="${product}"]`).innerHTML = `${pill("AI rewrite · checked", "blue")} data as of ${esc(doc.as_of)}`;
        } else {
          toast(`Kept the standard wording: the rewrite failed a check (${r.problems.join("; ")}).`);
        }
      } catch (e) { toast(e.message); } finally { b.disabled = false; b.textContent = "Rewrite"; }
    });
    t.querySelectorAll("[data-fb]").forEach((b) => b.onclick = async () => {
      const [product, response] = b.dataset.fb.split("|");
      const next = await api(`/api/customer/${id}/feedback`, { product, response });
      toast(`Stored “${response.replace(/_/g, " ")}” for ${names[product]} in customer memory; suggestions recomputed.`);
      renderCustomer(id, "recs", next);
    });
  } else if (tab === "why") {
    t.innerHTML = `<div class="card"><div class="table-wrap"><table><thead><tr><th>Product</th><th>Eligible</th><th class="num">Propensity</th><th class="num">Threshold</th><th class="num">Confidence</th><th class="num">CO2e/yr</th><th>Decision</th></tr></thead><tbody>
      ${doc.decisions.map((d) => `<tr><td>${esc(names[d.product])}</td><td>${d.eligible ? pill("yes", "green") : pill("no", "red")}</td><td class="num mono">${d.propensity === null ? "–" : pct(d.propensity)}</td><td class="num mono">${d.propensity_threshold == null ? "–" : pct(d.propensity_threshold)}</td><td class="num mono">${d.confidence === null ? "–" : num(d.confidence, 2)}</td><td class="num mono">${d.co2e_avoided_kg_per_year == null ? "–" : kg(d.co2e_avoided_kg_per_year)}</td><td>${decisionPill(d.decision)}</td></tr>`).join("")}
      </tbody></table></div></div>`;
  } else if (tab === "c360" || tab === "features") {
    const secs = tab === "c360" ? doc.customer_360 : doc.features;
    t.innerHTML = `<div class="card">${secs.map((x, i) => `<details class="sec" ${i < 2 ? "open" : ""}><summary><span>${label(x.section)}</span><span class="q">${esc(x.question)}<br><i>read by: ${esc(x.used_by)}</i></span></summary>
      <div class="kv">${Object.entries(x.fields).map(([k, v]) => `<div><span>${label(k)}</span><b>${fmtField(v)}</b></div>`).join("")}</div></details>`).join("")}</div>`;
  } else {
    t.innerHTML = `<div class="card"><div class="card-h"><h3>What we remember, and what it does today</h3><span class="sub">CRM campaigns and app responses, last 12 months · as of ${esc(doc.as_of)}</span></div>
      <div class="table-wrap"><table><thead><tr><th>Product</th><th class="num">Contacts</th><th class="num">Ignored in a row</th><th class="num">Memory weight</th><th>Latest response</th><th>Date</th><th>Source</th><th>Effect today</th><th>Until</th></tr></thead><tbody>
      ${(doc.memory || []).map((m) => `<tr><td>${esc(m.name)}</td><td class="num mono">${num(m.contacts)}</td><td class="num mono">${num(m.streak)}</td><td class="num mono">×${num(m.weight, 2)}</td><td>${!m.response || m.response === "none" ? "–" : pill(m.response.replace(/_/g, " "), "blue")}</td><td class="mono">${esc(m.date || "–")}</td><td>${esc(m.source || "–")}</td><td>${effectPill(m.effect)}</td><td class="mono">${esc(m.until || "")}</td></tr>`).join("")}
      </tbody></table></div></div>
      ${doc.memory_effect ? `<div class="card"><div class="card-h"><h3>What memory changed for this customer</h3><span class="sub">the same engine with and without memory</span></div>
      <div class="table-wrap"><table><thead><tr><th>Product</th><th>Without memory</th><th>With memory</th><th class="num">Weight</th><th>Change</th></tr></thead><tbody>
      ${doc.memory_effect.products.map((m) => `<tr><td>${esc(m.name)}</td><td>${m.rank_without ? pill(`suggestion ${m.rank_without}`, "green") : decisionPill(m.without)}</td><td>${m.rank_with ? pill(`suggestion ${m.rank_with}`, "green") : decisionPill(m.with)}</td><td class="num mono">×${num(m.weight, 2)}</td><td>${m.change === "no change" ? pill("no change", "grey") : pill(m.change, m.change === "hidden by memory" || m.change === "moved down" ? "amber" : "blue")}</td></tr>`).join("")}
      </tbody></table></div></div>
      <div class="card"><div class="card-h"><h3>Timeline</h3><span class="sub">every contact and response we hold</span></div>
      <div class="table-wrap"><table><thead><tr><th>Date</th><th>Product</th><th>Response</th><th>Source</th></tr></thead><tbody>
      ${doc.memory_effect.timeline.map((e) => `<tr><td class="mono">${esc(e.date)}</td><td>${esc(e.product)}</td><td>${pill(e.response.replace(/_/g, " "), "blue")}</td><td>${esc(e.source)}</td></tr>`).join("") || `<tr><td colspan="4" class="empty">No contacts yet.</td></tr>`}
      </tbody></table></div></div>` : ""}
      <div class="card"><div class="card-h"><h3>Responses given in the apps</h3></div>
      <div class="table-wrap"><table><thead><tr><th>Date</th><th>Product</th><th>Response</th><th>Source</th></tr></thead><tbody>
      ${(doc.memory_events || []).map((e) => `<tr><td class="mono">${esc(e.response_date)}</td><td>${esc(names[e.product])}</td><td>${pill(e.response.replace(/_/g, " "), "blue")}</td><td>${esc(e.source)}</td></tr>`).join("") || `<tr><td colspan="4" class="empty">None yet.</td></tr>`}
      </tbody></table></div></div>`;
  }
}
function evidenceWindow(asOf) {
  if (!asOf) return "";
  const end = new Date(asOf), start = new Date(end);
  start.setFullYear(end.getFullYear() - 1); start.setDate(start.getDate() + 1);
  const f = (d) => d.toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" });
  return `${f(start)} – ${f(end)}`;
}
function confidenceWord(score) {
  if (score == null) return "–";
  return score <= 2.5 ? "high" : score <= 3.5 ? "good" : score <= 4.5 ? "partial" : "low";
}
function effectPill(e) {
  if (!e || e === "no effect" || e === "no history") return pill(e || "none", "grey");
  if (e.includes("cool-down") || e.startsWith("no interest")) return pill(`paused · ${e}`, "amber");
  if (e === "already has it") return pill("stopped · already has it", "red");
  if (e === "priority raised") return pill(e, "green");
  if (e === "priority lowered") return pill(e, "amber");
  return pill(e, "grey");
}
function fmtField(v) {
  if (v === null || v === undefined) return "–";
  if (typeof v === "boolean") return v ? "yes" : "no";
  if (typeof v === "number") return Number.isInteger(v) ? num(v) : num(v, Math.abs(v) < 10 ? 3 : 1);
  return esc(String(v).replace(/_/g, " "));
}
function recCard(r) {
  const conf = r.confidence >= 0.7 ? ["high", "green"] : r.confidence >= 0.45 ? ["medium", "amber"] : ["low", "red"];
  const okImpact = r.impact_status === "ok";
  const money = r.running_cost_change_gbp_per_year;
  const moneyText = money === null || money === undefined || Math.abs(money) < 5 ? (r.product === "GP05" ? "Not estimated" : "No change") : `${gbp(Math.abs(money))} ${money < 0 ? "less" : "more"}`;
  const moneySmall = r.product === "GP05" ? "Returns are uncertain; information, not advice." : [r.upfront_cost_gbp ? `Upfront about ${gbp(r.upfront_cost_gbp)}` : "", r.payback_years && r.payback_years <= 30 ? `pays back in about ${num(r.payback_years)} years` : (r.upfront_cost_gbp ? "payback over 30 years" : "")].filter(Boolean).join(", ") || "per year at today's prices";
  const reasons = (r.reason_codes || []).map((c, i) => `<div class="reason"><span class="mono code">${esc(c)}</span><span>${esc((r.evidence || [])[i] || "")}</span></div>`).join("");
  const ce = r.confidence_of_estimate;
  const record = [["customer_id", r.customer_id], ["opportunity", r.opportunity], ["propensity", num(r.propensity, 2)], ["confidence", num(r.confidence, 2)],
    ["reason_codes", (r.reason_codes || []).join(", ") || "–"], ["evidence", (r.evidence || []).join("; ") || "–"],
    ["impact", r.impact_status === "ok" ? `${num(r.co2e_avoided_kg_per_year)} kg CO2e/yr (${r.method})` : "insufficient data"],
    ["evidence_window", evidenceWindow(r.data_timestamp)], ["data_timestamp", r.data_timestamp],
    ...(r.value ? [["customer_npv", gbp(r.value.customer_npv_gbp)], ["abatement_cost", r.value.abatement_cost_gbp_per_t == null ? "–" : `${gbp(r.value.abatement_cost_gbp_per_t)} per t CO2e`],
      ["bank_value_if_taken", gbp(r.value.bank_value_if_taken_gbp)], ["bank_expected_value", gbp(r.value.bank_expected_value_gbp)]] : [])];
  return `<div class="card rec" style="margin-bottom:0">
    <div class="rec-top"><div><div class="rank">Suggestion ${r.rank} · <span class="mono">${esc(r.opportunity || r.product)}</span></div><h4>${esc(r.product_name)}</h4></div>${r.status === "human_review" ? pill("human review", "amber") : pill("recommended", "green")}</div>
    <div class="meter"><span>Propensity <b>${pct(r.propensity)}</b></span><span>Confidence <b>${num(r.confidence, 2)}</b> ${pill(conf[0], conf[1])}</span><span title="${esc(ce ? `${ce.based_on || ""} (PCAF data-quality score ${ce.pcaf_score}: 1 best, 5 worst)` : "")}">Estimate confidence <b class="dots">${ce ? "●".repeat(ce.dots) + "○".repeat(ce.of - ce.dots) : "–"}</b> ${ce ? esc(ce.level) : ""}</span></div>
    <div class="statements">
      <div class="statement carbon"><div class="k">Sustainability gain</div><div class="big">${okImpact ? kg(r.co2e_avoided_kg_per_year) + " CO2e/yr" : "Insufficient data"}</div><div class="small">${okImpact ? `${esc(r.measure || "")}${r.horizon_years > 1 ? ` · about ${kg(r.co2e_avoided_kg_horizon)} over ${num(r.horizon_years)} years` : ""}` : "Inputs the method needs are missing; no number is given."}</div></div>
      <div class="statement money"><div class="k">Financial impact</div><div class="big">${moneyText}${moneyText.endsWith("less") || moneyText.endsWith("more") ? "/yr" : ""}</div><div class="small">${esc(moneySmall)}</div></div>
    </div>
    <div><div class="eyebrow" style="margin-bottom:6px">Why (reason codes and evidence) · data ${esc(evidenceWindow(r.data_timestamp))}</div>${reasons || '<span class="chip">overall fit</span>'}</div>
    <details class="record"><summary>Recommendation record</summary><pre class="mono">${record.map(([k, v]) => `${k}: ${esc(v)}`).join("\n")}</pre>${ce && ce.based_on ? `<div class="note">Estimate confidence ${esc(ce.level)}: ${esc(ce.based_on)} (PCAF data-quality score ${ce.pcaf_score}, where 1 is best and 5 is worst).</div>` : ""}${r.method_label ? `<div class="note">Method: ${esc(r.method_label)}.</div>` : ""}</details>
    <div class="explanation"><span data-text="${r.product}">${esc(r.explanation || "")}</span><div class="src"><span data-src="${r.product}">${r.source === "llm" ? pill("AI rewrite · checked", "blue") : pill("standard wording", "grey")} data as of ${esc(r.data_timestamp)}</span><span>${esc(r.customer_action || "")} <button class="btn small" data-rw="${r.product}">Rewrite</button></span></div></div>
    <div class="feedback"><span>Customer response:</span>${[["clicked", "Interested"], ["no_response", "No response"], ["dismissed", "Dismiss"], ["not_interested", "Not interested"], ["already_have", "Already have it"]].map(([k, v]) => `<button class="btn small" data-fb="${r.product}|${k}">${v}</button>`).join("")}<span class="note" style="margin-left:6px">memory weight ×${num(r.memory_weight ?? 1, 2)}</span></div>
  </div>`;
}
/* ---------- products ---------- */
function renderProducts() {
  const ins = state.overview.insights, P = ins.products, names = P.map((p) => p.name);
  view.innerHTML = pageHead("Products", `${P.length} green products · live snapshot ${esc(state.overview.dates?.live || "")}`) + `
    <div class="grid g-2" style="margin-bottom:24px">${chartBox("p-shown", "Suggestions per product", "shown to customers and sent to human review")}${chartBox("p-takeups", "Expected take-ups per product", "sum of calibrated propensities of the suggestions")}</div>
    <div class="grid g-2" style="margin-bottom:24px">${chartBox("p-conf", "Estimate confidence of the suggestions", "how much each impact estimate rests on the customer's own data")}${chartBox("p-elig", "Eligible customers per product", "after the eligibility rules")}</div>
    <div class="grid g-2" style="margin-bottom:24px">${chartBox("p-co2", "CO2e avoided per take-up", "median, kg a year")}${chartBox("p-money", "Running-cost change per take-up", "median, £ a year (negative = saving)")}</div>
    <div class="card"><div class="card-h"><h3>Product schedule</h3><span class="sub">medians over the customers shown each product</span></div><div class="card-b grid g-3">
    ${P.map((p) => `<div class="box"><div class="k" style="color:${PRODUCT_COLOURS[p.product]}">${esc(p.name)}</div><div class="v">${num(p.expected_co2e_tonnes_per_year, p.expected_co2e_tonnes_per_year < 10 ? 1 : 0)} t CO2e/yr expected</div>
      <div class="d">Eligible ${num(p.eligible)} · shown ${num(p.shown)} · review ${num(p.human_review)}<br>Expected take-ups ${num(p.expected_takeups, 0)} (mean propensity ${pct(p.mean_propensity_shown)})<br>
      Per take-up: ${p.co2e_kg_per_year_if_taken_up_median !== null ? kg(p.co2e_kg_per_year_if_taken_up_median) + " CO2e/yr" : "–"}${p.money_gbp_per_year_median !== null ? ` · ${gbp(p.money_gbp_per_year_median)}/yr running cost` : ""}${p.upfront_gbp_median ? ` · upfront ${gbp(p.upfront_gbp_median)}` : ""}${p.payback_years_median ? ` · payback ${num(p.payback_years_median, 0)} yrs` : ""}<br>
      Impact not computable for ${pct(p.insufficient_data_share, 0)} · estimate confidence ${confidenceWord(p.mean_data_quality_shown)}</div></div>`).join("")}
    </div></div>
    <div class="card"><div class="card-h"><h3>Decision funnel</h3><span class="sub">every customer × product, by the first rule that decided it</span></div>
    <div class="table-wrap"><table><thead><tr><th>Product</th>${Object.keys(FUNNEL_COLOURS).map((k) => `<th class="num">${esc(k)}</th>`).join("")}</tr></thead><tbody>
    ${ins.funnel.map((f) => `<tr><td>${esc(state.overview.products[f.product])}</td>${Object.keys(FUNNEL_COLOURS).map((k) => `<td class="num mono">${num(f[k] || 0)}</td>`).join("")}</tr>`).join("")}
    </tbody></table></div></div>`;
  plot("p-shown", "bar", names, [{ label: "shown", data: P.map((p) => p.shown), color: "#15803d" }, { label: "human review", data: P.map((p) => p.human_review), color: "#b45309" }], { stacked: true });
  plot("p-takeups", "bar", names, [{ label: "expected take-ups", data: P.map((p) => p.expected_takeups), color: "#1d4ed8" }]);
  const lv = [["High", "#15803d"], ["Good", "#65a30d"], ["Partial", "#d97706"], ["Low", "#b91c1c"]];
  plot("p-conf", "bar", names, lv.map(([k, c]) => ({ label: k, data: P.map((p) => (p.confidence_mix || {})[k] || 0), color: c })), { stacked: true, horizontal: true });
  plot("p-elig", "bar", names, [{ label: "eligible customers", data: P.map((p) => p.eligible), color: "#64748b" }]);
  plot("p-co2", "bar", names, [{ label: "kg CO2e a year", data: P.map((p) => p.co2e_kg_per_year_if_taken_up_median), color: "#15803d" }], { horizontal: true, fmt: (v) => num(v) + " kg" });
  plot("p-money", "bar", names, [{ label: "£ a year", data: P.map((p) => p.money_gbp_per_year_median), color: "#1d4ed8" }], { horizontal: true, fmt: moneyTick });
}

/* ---------- regions ---------- */
async function renderRegions(region = "") {
  const ins = state.overview.insights, tr = ins.trends;
  const t = await api(`/api/trends?region=${encodeURIComponent(region)}`);
  const fp = ins.footprint, months = tr.months.map(monthLabel);
  const R = [...ins.regions].sort((a, b) => b.expected_co2e_kg_per_customer - a.expected_co2e_kg_per_customer);
  view.innerHTML = pageHead("Regions & trends", "Spending trends from categorised transactions, footprint and opportunity by region") + `
    <div class="card"><div class="card-b"><div class="chips-row" style="margin-bottom:14px"><b>Region:</b><button class="qchip ${region ? "" : "on"}" data-r="">All regions</button>${state.overview.regions.map((r) => `<button class="qchip ${r === region ? "on" : ""}" data-r="${esc(r)}">${esc(r)}</button>`).join("")}</div>
      <div class="eyebrow" style="margin-bottom:8px">Monthly spend by category · ${esc(t.region)}</div><div class="chart-box"><canvas id="trend"></canvas></div></div></div>
    <div class="eyebrow" style="margin:6px 0 10px">Green signals over time · all regions</div>
    <div class="grid g-2" style="margin-bottom:24px">${chartBox("t-ev", "Customers using public EV charging", "customers with an EV charging payment each month")}${chartBox("t-green", "Spend on green travel and home upgrades", "£ a month")}</div>
    <div class="grid g-2" style="margin-bottom:24px">${chartBox("t-fuel", "Road fuel: spend vs litres", "litres = spend ÷ that month's average pump price")}${chartBox("t-energy", "Home energy bills", "£ a month")}</div>
    ${tr.takeups ? chartBox("t-takeups", "Green product take-ups", `new green products opened per month · observed window ${windowText()}`, 240) : ""}
    <div class="eyebrow" style="margin:6px 0 10px">Regions</div>
    <div class="grid g-2" style="margin-bottom:24px">${chartBox("r-opp", "Expected CO2e avoided per customer", "kg a year, by region", 300)}${chartBox("r-fp", "Median footprint by region", "travel and home energy, CO2e a year", 300)}</div>
    <div class="grid g-2" style="margin-bottom:24px">${chartBox("r-income", "Footprint by income band", "median CO2e a year")}${chartBox("r-dec", "Footprint distribution", "deciles, CO2e a year")}</div>
    <div class="card"><div class="card-h"><h3>Regions</h3><span class="sub">sorted by expected CO2e avoided per customer</span></div><div class="table-wrap"><table><thead><tr><th>Region</th><th class="num">Customers</th><th class="num">Median footprint</th><th class="num">Transport</th><th class="num">Home</th><th class="num">Suggestions</th><th class="num">Expected CO2e/yr</th><th class="num">Per customer</th><th>Top product</th></tr></thead><tbody>
      ${R.map((r) => `<tr><td>${esc(r.region)}</td><td class="num mono">${num(r.customers)}</td><td class="num mono">${kg(r.footprint_kg_median)}</td><td class="num mono">${kg(r.transport_kg_median)}</td><td class="num mono">${kg(r.home_kg_median)}</td><td class="num mono">${num(r.recommendations)}</td><td class="num mono">${num(r.expected_co2e_tonnes_per_year, 1)} t</td><td class="num mono">${num(r.expected_co2e_kg_per_customer, 1)} kg</td><td>${esc(r.top_product)}</td></tr>`).join("")}
    </tbody></table></div></div>`;
  view.querySelectorAll("[data-r]").forEach((b) => b.onclick = () => { clearCharts(); renderRegions(b.dataset.r); });
  const keep = ["fuel", "electricity", "gas", "public_transport", "home_improvement", "automotive"];
  plot("trend", "line", t.months.map(monthLabel), keep.filter((c) => t.series[c]).map((c) => ({ label: c.replace(/_/g, " "), data: t.series[c], color: CATEGORY_COLOURS[c] })), { fmt: moneyTick });
  const gs = tr.green_signals || {};
  if (gs["EV public charging"]) plot("t-ev", "bar", months, [{ type: "line", label: "customers", data: gs["EV public charging"].customers, color: "#1d4ed8" }, { label: "spend", data: gs["EV public charging"].spend, color: "#bfdbfe", yAxisID: "y2" }], { y2: true, fmt2: moneyTick });
  plot("t-green", "line", months, ["Energy installers", "Bike and scooter hire", "Rail", "Bus and coach", "Heating oil and LPG"].filter((k) => gs[k]).map((k, i) => ({ label: k, data: gs[k].spend, color: PALETTE[i + 1] })), { fmt: moneyTick });
  if (tr.fuel) plot("t-fuel", "bar", months, [{ label: "spend", data: tr.fuel.spend, color: "#fecaca" }, { type: "line", label: "litres", data: tr.fuel.litres, color: "#b91c1c", yAxisID: "y2" }], { y2: true, fmt: moneyTick, fmt2: (v) => num(v / 1000, 0) + "k l" });
  if (tr.energy_bills) plot("t-energy", "bar", months, Object.entries(tr.energy_bills).map(([k, v], i) => ({ label: k, data: v, color: ["#1d4ed8", "#b45309", "#0f766e"][i] })), { stacked: true, fmt: moneyTick });
  const win = tr.months.map((m, i) => [m, i]).filter(([m]) => afterCutoff(m));
  if (tr.takeups) plot("t-takeups", "bar", win.map(([m]) => monthLabel(m)), Object.entries(tr.takeups).map(([k, v], i) => ({ label: k, data: win.map(([, j]) => v[j]), color: Object.values(PRODUCT_COLOURS)[i] })), { stacked: true });
  plot("r-opp", "bar", R.map((r) => r.region), [{ label: "kg CO2e a year per customer", data: R.map((r) => r.expected_co2e_kg_per_customer), color: "#15803d" }], { horizontal: true, fmt: (v) => num(v) + " kg" });
  const RF = [...ins.regions].sort((a, b) => b.footprint_kg_median - a.footprint_kg_median);
  plot("r-fp", "bar", RF.map((r) => r.region), [{ label: "travel", data: RF.map((r) => r.transport_kg_median), color: "#b91c1c" }, { label: "home energy", data: RF.map((r) => r.home_kg_median), color: "#b45309" }], { horizontal: true, stacked: true, fmt: (v) => num(v / 1000, 1) + " t" });
  plot("r-income", "bar", fp.by_income_band.map((b) => b.income_band.replace(/_/g, " ")), [{ label: "median kg CO2e", data: fp.by_income_band.map((b) => b.footprint_kg_median), color: "#334155" }], { fmt: (v) => num(v / 1000, 1) + " t" });
  plot("r-dec", "bar", fp.deciles_kg.map((_, i) => `P${(i + 1) * 10}`), [{ label: "kg CO2e a year", data: fp.deciles_kg, color: "#15803d" }], { fmt: (v) => num(v / 1000, 1) + " t" });
}

/* ---------- business value ---------- */
function renderValue() {
  const B = state.overview.business, A = state.overview.business_assumptions || {};
  if (!B) { view.innerHTML = pageHead("Business value", "Not computed for this run") ; return; }
  const c = B.portfolio.customer, b = B.portfolio.bank, P = B.products, names = P.map((p) => p.name), bt = B.backtest || {};
  const G = B.grounded, U = B.measured_uplift;
  const signed = (x) => (x < 0 ? "−" : "") + gbp(Math.abs(x)).replace("−", "");
  view.innerHTML = pageHead("Business value", "Financial and carbon return of today's suggestions · expected = propensity × value if taken up") + `
    <div class="eyebrow" style="margin:0 0 10px">For customers</div>
    <div class="grid g-4" style="margin-bottom:10px">
      ${kpi("Expected savings", gbp(c.expected_saving_gbp_per_year) + "/yr", "running costs at today's prices", "good")}
      ${kpi("Expected net benefit", gbp(c.expected_npv_gbp), `net present value at ${pct(A.discount_rate || 0.035, 1)} over each product's life`, "blue")}
      ${kpi("Suggestions that pay back", pct(c.share_with_positive_npv, 0), "positive net present value for the customer")}
      ${kpi("Cost of carbon abatement", (c.abatement_cost_gbp_per_t < 0 ? "−" : "") + gbp(Math.abs(c.abatement_cost_gbp_per_t)) + "/t", c.abatement_cost_gbp_per_t < 0 ? "negative: customers save money per tonne cut" : "per tonne CO2e avoided", c.abatement_cost_gbp_per_t < 0 ? "good" : "")}
    </div>
    <div class="grid g-3" style="margin-bottom:24px">
      ${kpi("Expected CO2e avoided", num(c.expected_co2e_t_per_year, 0) + " t/yr", `${num(c.expected_co2e_t_lifetime)} t over the products' lives`, "good")}
      ${kpi("Median payback", c.median_payback_years == null ? "–" : num(c.median_payback_years, 1) + " years", "where there is an upfront cost")}
      ${kpi("Median net present value per take-up", signed(c.median_npv_per_takeup_gbp), "for the customer")}
    </div>
    <div class="eyebrow" style="margin:0 0 10px">For the bank · grounded</div>
    ${G ? `<div class="grid g-4" style="margin-bottom:10px">
      ${kpi("Value at stake", gbp(G.value_at_stake_gbp), `bank value of the ${num(G.takeups_at_stake, 0)} take-ups expected among the customers suggested to; most would happen without us`)}
      ${kpi("Campaign cost", gbp(G.campaign_cost_gbp), "contact by channel and human reviews")}
      ${kpi("Break-even uplift", pct(G.break_even_share_caused, 1), "share of those take-ups the suggestions must cause to pay for themselves", "good")}
      ${kpi("New green lending at stake", gbp(b.expected_new_lending_gbp), "EV, home, solar and green loans")}
    </div>` : ""}
    <div class="grid g-4" style="margin-bottom:24px">
      ${kpi("Net interest income at stake", gbp(b.expected_nii_gbp), `over the loans' lives, margin over ${pct(A.cost_of_funds || 0.04, 1)} funding`)}
      ${kpi("Expected credit losses", gbp(b.expected_credit_loss_gbp), "probability of default × loss given default", "alert")}
      ${kpi("CO2e at stake per £1,000 spent", num(b.t_co2e_lifetime_per_1k_campaign, 1) + " t", "over the products' lives", "good")}
      ${kpi("Financed emissions avoided", num(b.expected_financed_co2e_t_per_year, 0) + " t/yr", `${num(b.financed_t_per_year_per_m_lending, 1)} t a year per £1m of new lending (PCAF)`)}
    </div>
    ${G ? `<div class="grid g-2" style="margin-bottom:24px">
      <div class="card" style="margin-bottom:0"><div class="card-h"><h3>If the suggestions cause this share of take-ups</h3><span class="sub">scenarios until the control group measures it</span></div>
      <div class="table-wrap"><table><thead><tr><th>Share caused</th><th class="num">Take-ups caused</th><th class="num">Net value</th><th class="num">Return on cost</th><th class="num">CO2e caused</th></tr></thead><tbody>
      ${G.scenarios.map((x) => `<tr><td class="mono">${pct(x.share_caused, 0)}</td><td class="num mono">${num(x.incremental_takeups, 0)}</td><td class="num mono">${signed(x.incremental_net_value_gbp)}</td><td class="num mono">${x.roi == null ? "–" : num(x.roi, 1) + "×"}</td><td class="num mono">${num(x.incremental_co2e_t_lifetime)} t</td></tr>`).join("")}
      </tbody></table></div></div>
      <div class="card" style="margin-bottom:0"><div class="card-h"><h3>How a take-up is linked to a suggestion</h3></div><div class="card-b">
        <ol class="steps">
          <li><b>Logged:</b> every suggestion has an id, a shown date and a channel.</li>
          <li><b>Direct:</b> the call to action (e.g. "Explore EV options") carries the id into the application, so that take-up is linked for certain.</li>
          <li><b>Window:</b> otherwise, the same product opened in the product system within ${num((A.attribution || {}).window_days)} days of being shown is linked to it.</li>
          <li><b>Caused:</b> ${pct((A.attribution || {}).control_share, 0)} of selected customers are held back at random; take-ups caused = treated rate − held-back rate.</li>
        </ol>
        ${U ? `<div class="note" style="margin-top:8px"><b>Measured on the backtest</b> (${esc(U.note)}): shown ${num(U.treated.customers)} customers, ${pct(U.treated.rate, 1)} took up; held back ${num(U.control.customers)}, ${pct(U.control.rate, 1)} took up; uplift ${num(U.uplift.points, 1)} points (95% interval ${num(U.uplift.ci95_points[0], 1)} to ${num(U.uplift.ci95_points[1], 1)}).</div>` : ""}
      </div></div>
    </div>` : ""}
    <div class="grid g-2" style="margin-bottom:24px">${chartBox("v-bank", "Bank value at stake by product", "£, propensity-weighted, after costs; not all caused by the suggestion")}${chartBox("v-abate", "Customer cost of carbon abatement", "£ per tonne CO2e, negative = saves money", 260)}</div>
    <div class="grid g-2" style="margin-bottom:24px">${chartBox("v-cust", "Expected customer savings by product", "£ a year, propensity-weighted")}${chartBox("v-carbon", "Expected CO2e avoided by product", "tonnes over the products' lives")}</div>
    ${bt.total ? `<div class="eyebrow" style="margin:6px 0 10px">Backtest: who should we contact? · held-out test customers, what really happened</div>
    <div class="grid g-3" style="margin-bottom:10px">${chartBox("bt-take", "Take-ups", "same number contacted per product", 220)}${chartBox("bt-co2", "CO2e avoided", "tonnes over the products' lives", 220)}${chartBox("bt-value", "Bank value", "£, after contact costs", 220)}</div>
    <div class="card"><div class="card-b note">${esc(bt.note || "")}. Model vs random: ${num(bt.uplift_vs_random?.takeups, 1)}× the take-ups, ${num(bt.uplift_vs_random?.co2e_t_lifetime, 1)}× the CO2e. Model vs rules baseline: ${num(bt.uplift_vs_rules?.takeups, 1)}× the take-ups, ${num(bt.uplift_vs_rules?.co2e_t_lifetime, 1)}× the CO2e.</div></div>` : ""}
    <div class="card"><div class="card-h"><h3>By product</h3><span class="sub">expected values over today's suggestions</span></div><div class="table-wrap"><table><thead><tr><th>Product</th><th class="num">Suggestions</th><th class="num">Take-ups</th><th class="num">Customer saving/yr</th><th class="num">Customer NPV</th><th class="num">Abatement £/t</th><th class="num">New lending</th><th class="num">Bank value at stake</th><th class="num">(At stake − cost) ÷ cost</th><th class="num">t CO2e per £1k</th></tr></thead><tbody>
    ${P.map((p) => `<tr><td>${esc(p.name)}</td><td class="num mono">${num(p.suggestions)}</td><td class="num mono">${num(p.expected_takeups, 0)}</td><td class="num mono">${gbp(p.customer.expected_saving_gbp_per_year)}</td><td class="num mono">${signed(p.customer.expected_npv_gbp)}</td><td class="num mono">${p.customer.abatement_cost_gbp_per_t == null ? "–" : signed(p.customer.abatement_cost_gbp_per_t)}</td><td class="num mono">${gbp(p.bank.expected_new_lending_gbp)}</td><td class="num mono">${signed(p.bank.expected_net_value_gbp)}</td><td class="num mono">${p.bank.roi == null ? "–" : num(p.bank.roi, 0) + "×"}</td><td class="num mono">${num(p.bank.t_co2e_lifetime_per_1k_campaign, 1)}</td></tr>`).join("")}
    </tbody></table></div></div>
    <div class="card"><div class="card-h"><h3>Assumptions</h3><span class="sub">configs/business.yaml · illustrative, plausible for a UK retail bank in 2026</span></div><div class="card-b chips-row">
      <span class="chip">discount rate ${pct(A.discount_rate, 1)} (Green Book)</span><span class="chip">cost of funds ${pct(A.cost_of_funds, 1)}</span><span class="chip">EV price ${gbp(A.ev_price_gbp)}</span>
      ${Object.entries(A.lending || {}).map(([k, v]) => `<span class="chip">${esc(state.overview.products[k])}: APR ${pct(v.rate, 1)}, ${v.term_years} yrs, PD ${pct(v.pd, 1)}, LGD ${pct(v.lgd, 0)}</span>`).join("")}
      <span class="chip">contact £${num((A.contact_cost_gbp || {}).app, 2)} app · £${num((A.contact_cost_gbp || {}).email, 2)} email · £${num((A.contact_cost_gbp || {}).post, 2)} post</span><span class="chip">human review £${num(A.human_review_cost_gbp)}</span>
    </div></div>`;
  plot("v-bank", "bar", names, [{ label: "expected net value", data: P.map((p) => p.bank.expected_net_value_gbp), color: "#1d4ed8" }], { fmt: moneyTick });
  const ab = P.filter((p) => p.customer.abatement_cost_gbp_per_t != null);
  plot("v-abate", "bar", ab.map((p) => p.name), [{ label: "£ per tonne", data: ab.map((p) => p.customer.abatement_cost_gbp_per_t), backgroundColor: ab.map((p) => p.customer.abatement_cost_gbp_per_t < 0 ? "#15803d" : "#d97706") }], { horizontal: true, fmt: moneyTick });
  plot("v-cust", "bar", names, [{ label: "£ a year", data: P.map((p) => p.customer.expected_saving_gbp_per_year), color: "#15803d" }], { fmt: moneyTick });
  plot("v-carbon", "bar", names, [{ label: "t CO2e", data: P.map((p) => p.customer.expected_co2e_t_lifetime), color: "#15803d" }], { fmt: (v) => num(v) + " t" });
  if (bt.total) {
    const S = ["model", "rules", "random"].filter((k) => bt.total[k]), lab = { model: "Our model", rules: "Rules baseline", random: "Random" }, col = { model: "#1d4ed8", rules: "#94a3b8", random: "#cbd5e1" };
    const one = (id, key, fmt) => plot(id, "bar", S.map((k) => lab[k]), [{ label: key, data: S.map((k) => bt.total[k][key]), backgroundColor: S.map((k) => col[k]) }], { fmt });
    one("bt-take", "takeups"); one("bt-co2", "co2e_t_lifetime", (v) => num(v) + " t"); one("bt-value", "bank_value_gbp", moneyTick);
  }
}

/* ---------- memory ---------- */
async function renderMemory() {
  const mem = await api("/api/memory"), m = mem.replay;
  view.innerHTML = pageHead("Customer memory", `Responses and their effect on ${esc(mem.as_of)} · dismissed pauses ${mem.cooldown_days.dismissed} days, not interested ${mem.cooldown_days.not_interested} days`,
    `<button class="btn" id="reset">Forget app responses (${num(mem.app_responses)})</button>`) + `
    ${mem.effect ? `<div class="grid g-4" style="margin-bottom:24px">
      ${kpi("Suggestions without memory", num(mem.effect.suggestions_without_memory), "same engine, empty memory")}
      ${kpi("Hidden by memory", num(mem.effect.hidden_by_memory), "paused, stopped or outranked", "alert")}
      ${kpi("Shown because of memory", num(mem.effect.shown_because_of_memory), `moved into a customer's top ${num(state.overview.config?.max_per_customer)}`, "blue")}
      ${kpi("Reordered", num(mem.effect.reordered), "same suggestion, different rank")}
    </div>` : ""}
    <div class="grid g-2" style="margin-bottom:24px">${chartBox("mem-state", "Memory in force by product", "customers per state")}${chartBox("mem-replay", `Replay: suggestions over ${m ? m.rounds.length : ""} months`, "shown vs held back as memory builds up")}</div>
    <div class="card"><div class="card-h"><h3>Memory in force today</h3><span class="sub">every contact and response in the last 12 months · ${num(mem.fatigue_contacts)} ignored contacts in a row pause a product</span></div>
    <div class="table-wrap"><table><thead><tr><th>Product</th><th class="num">Customers with history</th><th class="num">Paused · cool-down</th><th class="num">Paused · no interest</th><th class="num">Stopped · already has it</th><th class="num">Priority raised</th><th class="num">Priority lowered</th></tr></thead><tbody>
    ${mem.products.map((r) => `<tr><td>${esc(r.name)}</td><td class="num mono">${num(r.customers_with_history)}</td><td class="num mono">${num(r.paused_cooldown)}</td><td class="num mono">${num(r.paused_no_interest)}</td><td class="num mono">${num(r.stopped_already_have)}</td><td class="num mono">${num(r.priority_raised)}</td><td class="num mono">${num(r.priority_lowered)}</td></tr>`).join("")}
    </tbody></table></div>
    <div class="card-b"><div class="chips-row"><b>Memory weight per response:</b>${Object.entries(mem.weights).map(([k, v]) => `<span class="chip">${label(k)} ×${num(v, 2)}</span>`).join("")}</div></div></div>
    <div class="card"><div class="card-h"><h3>Latest responses in the apps</h3></div>
    <div class="table-wrap"><table><thead><tr><th>Customer</th><th>Product</th><th>Response</th><th>Source</th></tr></thead><tbody>
    ${mem.recent.map((e) => `<tr class="click" data-id="${esc(e.customer_id)}"><td class="mono">${esc(e.customer_id)}</td><td>${esc(e.product)}</td><td>${pill(e.response.replace(/_/g, " "), "blue")}</td><td>${esc(e.source)}</td></tr>`).join("") || `<tr><td colspan="4" class="empty">None yet: respond to a suggestion in the customer app or on a customer page.</td></tr>`}
    </tbody></table></div></div>
    ${m ? `<div class="card"><div class="card-h"><h3>Check: replay of ${num(m.customers)} customers over ${m.rounds.length} months</h3><span class="sub">simulated responses · a paused product must never come back early</span></div>
    <div class="table-wrap"><table><thead><tr><th>Round</th><th>Date</th><th class="num">Shown</th><th class="num">Held back by memory</th><th class="num">Paused · no interest</th><th class="num">Re-shown inside a pause</th><th>Responses</th></tr></thead><tbody>
    ${m.rounds.map((r) => `<tr><td class="mono">${r.round}</td><td class="mono">${esc(r.date)}</td><td class="num mono">${num(r.shown)}</td><td class="num mono">${num(r.suppressed_by_memory)}</td><td class="num mono">${num(r.paused_no_interest ?? 0)}</td><td class="num">${r.reshown_within_cooldown === undefined ? "–" : pill(num(r.reshown_within_cooldown), r.reshown_within_cooldown === 0 ? "green" : "red")}</td><td>${Object.entries(r.responses).map(([k, v]) => `<span class="chip">${label(k)} ${num(v)}</span>`).join("")}</td></tr>`).join("")}
    </tbody></table></div></div>` : ""}`;
  $("#reset").onclick = async () => { await api("/api/memory/reset", {}); toast("App responses forgotten; suggestions restored."); renderMemory(); };
  view.querySelectorAll("tr.click").forEach((tr) => tr.onclick = () => location.hash = `#/customer/${tr.dataset.id}`);
  const st = [["paused_cooldown", "paused · cool-down", "#d97706"], ["paused_no_interest", "paused · no interest", "#b45309"], ["stopped_already_have", "stopped · already has it", "#b91c1c"], ["priority_raised", "priority raised", "#15803d"], ["priority_lowered", "priority lowered", "#94a3b8"]];
  plot("mem-state", "bar", mem.products.map((r) => r.name), st.map(([k, l, c]) => ({ label: l, data: mem.products.map((r) => r[k]), color: c })), { stacked: true, horizontal: true });
  if (m) plot("mem-replay", "bar", m.rounds.map((r) => r.date), [{ label: "shown", data: m.rounds.map((r) => r.shown), color: "#15803d" }, { label: "held back by memory", data: m.rounds.map((r) => r.suppressed_by_memory), color: "#d97706" }, { type: "line", label: "paused · no interest", data: m.rounds.map((r) => r.paused_no_interest || 0), color: "#b91c1c" }]);
}

/* ---------- models & checks ---------- */
function renderModels(product = "GP01") {
  const o = state.overview, names = o.products, ex = o.explanations || {}, tr = o.impact_truth_recovery || {};
  const models = o.models || {};
  const metricRow = (p) => {
    const t = models[p].test, c = t.lightgbm_calibrated, b = t.rules_baseline, l = t.logistic_regression, or = t.oracle_p_true;
    return `<tr><td>${esc(names[p])}</td><td class="num mono">${num(models[p].positives.test)}</td><td class="num mono">${num(b.roc_auc, 3)}</td><td class="num mono">${num(l.roc_auc, 3)}</td><td class="num mono">${num((t.xgboost_raw || {}).roc_auc, 3)}</td><td class="num mono">${num((t.catboost_raw || {}).roc_auc, 3)}</td><td class="num mono"><b>${num(c.roc_auc, 3)}</b></td><td class="num mono">${num(or.roc_auc, 3)}</td><td class="num mono">${num(c.pr_auc, 3)}</td><td class="num mono">${num(c.lift_top10, 2)}×</td><td class="num mono">${num(c.ece, 3)}</td><td class="num mono">${num(models[p].mean_confidence_test, 2)}</td></tr>`;
  };
  const ab = (o.ablations || {})[product] || {};
  const fair = models[product]?.fairness_test || {};
  const llm = ex.llm_eval || {};
  const recovery = ["fuel_litres", "gas_kwh", "electricity_kwh", "transport_co2e_kg", "home_energy_co2e_kg"].filter((k) => tr[k]);
  const sens = (o.impact_sensitivity || []).filter((r) => r.co2e_swing_pct > 0 || r.money_swing_pct > 0).sort((a, b) => (b.co2e_swing_pct || 0) - (a.co2e_swing_pct || 0)).slice(0, 12);
  view.innerHTML = pageHead("Models & checks", "Held-out test results") + `
    <div class="grid g-2" style="margin-bottom:24px">${chartBox("m-auc", "Test AUC by product", "ranking quality on customers never seen in training")}${chartBox("m-lift", "Lift in the top 10%", "take-up rate of the top 10% vs average")}</div>
    <div class="card"><div class="card-h"><h3>FR3 propensity · test set (customers never seen in training)</h3><span class="sub">AUC: ranking · lift: take-up rate in the top 10% vs average · ECE: calibration error</span></div>
    <div class="table-wrap"><table><thead><tr><th>Product</th><th class="num">Test take-ups</th><th class="num">Rules AUC</th><th class="num">Logistic AUC</th><th class="num">XGBoost AUC</th><th class="num">CatBoost AUC</th><th class="num">LightGBM AUC</th><th class="num">Oracle AUC</th><th class="num">PR-AUC</th><th class="num">Lift top 10%</th><th class="num">ECE</th><th class="num">Confidence</th></tr></thead><tbody>
    ${Object.keys(models).map(metricRow).join("")}</tbody></table></div></div>
    <div class="card"><div class="card-h"><h3>Suggestion thresholds</h3><span class="sub">propensity ≥ ${num((Object.values(models)[0] || {}).threshold?.min_lift, 1)} × take-up rate · test set</span></div>
    <div class="table-wrap"><table><thead><tr><th>Product</th><th class="num">Take-up rate</th><th class="num">Threshold</th><th class="num">Customers above</th><th class="num">Their take-up rate</th><th class="num">Lift</th><th class="num">Take-ups captured</th></tr></thead><tbody>
    ${Object.keys(models).filter((p) => models[p].threshold).map((p) => { const t = models[p].threshold, a = t.at_threshold.test; return `<tr><td>${esc(names[p])}</td><td class="num mono">${pct(t.base_rate)}</td><td class="num mono">${pct(t.min_propensity)}</td><td class="num mono">${pct(a.selected_share)}</td><td class="num mono">${a.takeup_rate_selected == null ? "–" : pct(a.takeup_rate_selected)}</td><td class="num mono">${a.lift == null ? "–" : num(a.lift, 1) + "×"}</td><td class="num mono">${pct(a.recall, 0)}</td></tr>`; }).join("")}
    </tbody></table></div></div>
    <div class="chips-row" style="margin-bottom:12px"><b>Product:</b>${Object.keys(models).map((p) => `<button class="qchip ${p === product ? "on" : ""}" data-p="${p}">${esc(names[p])}</button>`).join("")}</div>
    <div class="grid g-2" style="margin-bottom:24px">
      <div class="card"><div class="card-h"><h3>Ablations · ${esc(names[product])}</h3><span class="sub">retrained without one ingredient</span></div><div class="table-wrap"><table><thead><tr><th>Variant</th><th class="num">AUC</th><th class="num">PR-AUC</th><th class="num">Lift</th><th class="num">ECE</th><th class="num">Inputs</th></tr></thead><tbody>
        ${Object.entries(ab).map(([k, v]) => `<tr><td>${label(k)}</td><td class="num mono">${num(v.roc_auc, 3)}</td><td class="num mono">${num(v.pr_auc, 3)}</td><td class="num mono">${num(v.lift_top10, 2)}</td><td class="num mono">${num(v.ece, 3)}</td><td class="num mono">${num(v.features)}</td></tr>`).join("")}</tbody></table></div></div>
      <div class="card"><div class="card-h"><h3>Fairness · ${esc(names[product])}</h3><span class="sub">test set by income band</span></div><div class="table-wrap"><table><thead><tr><th>Band</th><th class="num">Customers</th><th class="num">Take-up rate</th><th class="num">Mean propensity</th><th class="num">Selected</th></tr></thead><tbody>
        ${Object.entries(fair.income_band || {}).map(([k, v]) => `<tr><td>${label(k)}</td><td class="num mono">${num(v.n)}</td><td class="num mono">${pct(v.take_up_rate)}</td><td class="num mono">${pct(v.mean_propensity)}</td><td class="num mono">${pct(v.selected_rate)}</td></tr>`).join("")}</tbody></table></div></div>
    </div>
    <div class="card"><div class="card-h"><h3>FR5 explanations</h3><span class="sub">LLM vs standard wording</span></div><div class="card-b grid g-${o.gemini_sample ? 5 : 4}">
      ${kpi("Template guard pass", `${num(ex.template_guard_pass)}/${num(ex.explanations)}`, "numbers, date, disclaimer, banned claims", "good")}
      ${kpi("LLM guard pass", llm.guard_pass_rate !== undefined ? pct(llm.guard_pass_rate, 0) : "–", llm.items ? `${num(llm.items)} rewrites · ${esc(llm.model)} (local)` : esc(llm.skipped || "not run yet"))}
      ${o.gemini_sample ? kpi("Gemini guard pass", `${num(o.gemini_sample.passed_guards)}/${num(o.gemini_sample.replies)}`, `${esc(Object.keys(o.gemini_sample.models_used || { [o.gemini_sample.model]: 1 }).join(", "))} · used for Rewrite · measured ${esc(o.gemini_sample.measured || "")}`, "good") : ""}
      ${kpi("Reading ease", llm.reading_ease_mean ? `${num(llm.reading_ease_mean.llm, 0)} vs ${num(llm.reading_ease_mean.template, 0)}` : num(ex.reading_ease_mean, 0), "Flesch: LLM vs template (higher is easier)")}
      ${kpi("Length", llm.words_mean ? `${num(llm.words_mean.llm, 0)} vs ${num(llm.words_mean.template, 0)}` : num(ex.words_mean, 0) + " words", "words: LLM vs template")}
    </div></div>
    <div class="card"><div class="card-h"><h3>FR4 impact inputs vs true usage</h3><span class="sub">${fmtDay(o.dates?.training_cutoff)} · visible spend only</span></div><div class="card-b grid g-${Math.min(recovery.length, 5) || 1}" style="grid-template-columns:repeat(${recovery.length || 1},1fr)">
      ${recovery.map((k) => kpi(label(k), pct(tr[k].within_20pct, 0), `within ±20% · median error ${num(tr[k].median_abs_pct_error, 1)}% · r = ${num(tr[k].correlation, 2)} · n = ${num(tr[k].customers)}`, tr[k].within_20pct >= 0.7 ? "good" : "")).join("")}
    </div></div>
    <div class="card"><div class="card-h"><h3>FR4 sensitivity</h3><span class="sub">mean per customer · low · base · high</span></div><div class="table-wrap"><table><thead><tr><th>Assumption</th><th>Range</th><th>Product</th><th class="num">CO2e low · base · high (kg/yr)</th><th class="num">Swing</th><th class="num">Money low · base · high (£/yr)</th><th class="num">Swing</th></tr></thead><tbody>
      ${sens.map((r) => `<tr><td>${label(r.assumption)}</td><td class="mono">${esc(r.range)}</td><td>${esc(names[r.product])}</td><td class="num mono">${num(r.co2e_low)} · ${num(r.co2e_base)} · ${num(r.co2e_high)}</td><td class="num mono">${num(r.co2e_swing_pct, 0)}%</td><td class="num mono">${num(r.money_low)} · ${num(r.money_base)} · ${num(r.money_high)}</td><td class="num mono">${num(r.money_swing_pct, 0)}%</td></tr>`).join("")}
    </tbody></table></div></div>`;
  view.querySelectorAll("[data-p]").forEach((b) => b.onclick = () => { clearCharts(); renderModels(b.dataset.p); });
  const ks = Object.keys(models), mn = ks.map((p) => names[p]);
  const series = (key, metric) => ks.map((p) => (models[p].test[key] || {})[metric]);
  const hasBench = (k) => ks.some((p) => models[p].test[k]);
  plot("m-auc", "bar", mn, [["Rules baseline", "rules_baseline", "#cbd5e1"], ["Logistic regression", "logistic_regression", "#94a3b8"],
    ...(hasBench("xgboost_raw") ? [["XGBoost", "xgboost_raw", "#7c3aed"]] : []), ...(hasBench("catboost_raw") ? [["CatBoost", "catboost_raw", "#db2777"]] : []),
    ["LightGBM (ours)", "lightgbm_calibrated", "#1d4ed8"], ["Oracle (ceiling)", "oracle_p_true", "#15803d"]]
    .map(([l, k, c]) => ({ label: l, data: series(k, "roc_auc"), color: c })), { fmt: (v) => num(v, 2) });
  plot("m-lift", "bar", mn, [["Rules baseline", "rules_baseline", "#cbd5e1"], ["LightGBM (ours)", "lightgbm_calibrated", "#1d4ed8"], ["Oracle (ceiling)", "oracle_p_true", "#15803d"]]
    .map(([l, k, c]) => ({ label: l, data: series(k, "lift_top10"), color: c })), { fmt: (v) => num(v, 1) + "×" });
}

route();
