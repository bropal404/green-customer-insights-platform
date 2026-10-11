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
const PAGES = { overview: renderOverview, customers: renderCustomers, customer: renderCustomer, products: renderProducts, regions: renderRegions, memory: renderMemory, models: renderModels };
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
  ["Ingest", "FR1 · 7 source systems", "data"], ["Categorise", "FR2 · 8 categories", "data"], ["Features", "Customer 360 + features", "data"],
  ["Propensity", "FR3 · LightGBM", "propensity"], ["Impact", "FR4 · carbon + money", "impact"], ["Recommend", "FR5–7 · top 3", "recommend"],
  ["Explain", "FR5 · template + local LLM", "explain"], ["Insights", "FR6 · bank view", "insights"],
];
function stepper(done) {
  return `<div class="stepper">${STEPS.map(([name, what, key], i) => {
    const ok = key === "data" || done.includes(key);
    return `<div class="step"><div class="dot ${ok ? "" : "todo"}">${ok && i === STEPS.length - 1 ? "✓" : i + 1}</div><div class="name" style="${ok ? "" : "color:var(--faint)"}">${name}</div><div class="what">${what}</div></div>`;
  }).join("")}</div>`;
}
function renderOverview() {
  const o = state.overview, ins = o.insights, ov = ins.overview, d = o.data || {};
  const products = ins.products;
  const maxT = Math.max(...products.map((p) => p.expected_co2e_tonnes_per_year || 0), 1);
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
      ${kpi("Customers with a suggestion", num(ov.customers_with_a_recommendation), `${num(ov.recommendations)} suggestions, at most 3 each`)}
      ${kpi("Expected take-ups", num(ov.expected_takeups), "sum of calibrated propensities", "blue")}
      ${kpi("Expected CO2e avoided", num(ov.expected_co2e_tonnes_per_year, 0) + " t/yr", `${num(ov.co2e_tonnes_per_year_if_all_shown_taken_up)} t/yr if every suggestion were taken up`, "good")}
      ${kpi("Expected customer savings", gbp(ov.customer_savings_gbp_per_year_expected) + "/yr", "running costs at today's prices")}
    </div>
    <div class="grid g-21" style="margin-bottom:24px">
      <div class="card"><div class="card-h"><h3>Expected CO2e avoided by product</h3><span class="sub">tonnes a year · propensity × avoided emissions</span></div>
        <div class="card-b">${products.map((p) => `<div class="bar-row"><span>${esc(p.name)}</span><div class="bar green"><i style="width:${(100 * (p.expected_co2e_tonnes_per_year || 0) / maxT).toFixed(1)}%"></i></div><span class="mono" style="text-align:right">${num(p.expected_co2e_tonnes_per_year, p.expected_co2e_tonnes_per_year < 10 ? 1 : 0)} t</span></div>`).join("")}</div></div>
      <div class="card"><div class="card-h"><h3>Needs attention</h3></div><div class="card-b grid" style="gap:10px">
        ${kpi("Human review", num(ov.human_review), "credit offers to customers with a vulnerability flag", "alert")}
        ${kpi("Median footprint", kg(ov.footprint_kg_median) + "/yr", "visible fuel and home energy per customer")}
      </div></div>
    </div>
    <div class="card"><div class="card-h"><h3>Why products were or were not shown</h3><span class="sub">audited decision for every customer × product</span></div><div class="card-b">${funnelBars(ins.funnel)}</div></div>
    <div class="section-head eyebrow" style="margin:6px 0 10px">Customer segments</div>
    <div class="grid g-3">${ins.segments.map(segmentCard).join("")}</div>`;
}
const FUNNEL_COLOURS = { recommended: "#15803d", "human review": "#b45309", blocked: "#b91c1c", "no marketing consent": "#94a3b8", "memory cool-down or already has it": "#1d4ed8", "impact cannot be estimated": "#fca5a5", "below the propensity threshold": "#cbd5e1", "low confidence": "#fde68a", "impact too small": "#a7f3d0", "routed to a more specific product": "#7c3aed", "lower priority than three other products": "#334155" };
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
    <div class="card"><div class="status"><div class="led" style="background:${doc.recommendations.length ? "var(--green)" : "var(--faint)"}"></div><div><b>${doc.recommendations.length ? `${doc.recommendations.length} green suggestion${doc.recommendations.length > 1 ? "s" : ""}` : "No suggestion today"}</b><span>${doc.recommendations.length ? "Ranked by propensity × confidence" : "See “Why not the others”"}</span></div></div>
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
    t.innerHTML = `<div class="card"><div class="card-h"><h3>What we remember, and what it does today</h3><span class="sub">latest response per product · as of ${esc(doc.as_of)}</span></div>
      <div class="table-wrap"><table><thead><tr><th>Product</th><th>Latest response</th><th>Date</th><th>Source</th><th>Effect today</th><th>Until</th></tr></thead><tbody>
      ${(doc.memory || []).map((m) => `<tr><td>${esc(m.name)}</td><td>${m.response === "none" ? "–" : pill(m.response.replace(/_/g, " "), "blue")}</td><td class="mono">${esc(m.date || "–")}</td><td>${esc(m.source || "–")}</td><td>${effectPill(m.effect)}</td><td class="mono">${esc(m.until || "")}</td></tr>`).join("")}
      </tbody></table></div></div>
      <div class="card"><div class="card-h"><h3>Responses given in the apps</h3></div>
      <div class="table-wrap"><table><thead><tr><th>Date</th><th>Product</th><th>Response</th><th>Source</th></tr></thead><tbody>
      ${(doc.memory_events || []).map((e) => `<tr><td class="mono">${esc(e.response_date)}</td><td>${esc(names[e.product])}</td><td>${pill(e.response.replace(/_/g, " "), "blue")}</td><td>${esc(e.source)}</td></tr>`).join("") || `<tr><td colspan="4" class="empty">None yet.</td></tr>`}
      </tbody></table></div></div>`;
  }
}
function confidenceWord(score) {
  if (score == null) return "–";
  return score <= 2.5 ? "high" : score <= 3.5 ? "good" : score <= 4.5 ? "partial" : "low";
}
function effectPill(e) {
  if (!e || e === "no effect") return pill("none", "grey");
  if (e.startsWith("paused")) return pill(e, "amber");
  if (e.startsWith("stopped")) return pill(e, "red");
  if (e.startsWith("priority")) return pill(e, "green");
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
    ["data_timestamp", r.data_timestamp]];
  return `<div class="card rec" style="margin-bottom:0">
    <div class="rec-top"><div><div class="rank">Suggestion ${r.rank} · <span class="mono">${esc(r.opportunity || r.product)}</span></div><h4>${esc(r.product_name)}</h4></div>${r.status === "human_review" ? pill("human review", "amber") : pill("recommended", "green")}</div>
    <div class="meter"><span>Propensity <b>${pct(r.propensity)}</b></span><span>Confidence <b>${num(r.confidence, 2)}</b> ${pill(conf[0], conf[1])}</span><span title="${esc(ce ? `${ce.based_on || ""} (PCAF data-quality score ${ce.pcaf_score}: 1 best, 5 worst)` : "")}">Estimate confidence <b class="dots">${ce ? "●".repeat(ce.dots) + "○".repeat(ce.of - ce.dots) : "–"}</b> ${ce ? esc(ce.level) : ""}</span></div>
    <div class="statements">
      <div class="statement carbon"><div class="k">Sustainability gain</div><div class="big">${okImpact ? kg(r.co2e_avoided_kg_per_year) + " CO2e/yr" : "Insufficient data"}</div><div class="small">${okImpact ? `${esc(r.measure || "")}${r.horizon_years > 1 ? ` · about ${kg(r.co2e_avoided_kg_horizon)} over ${num(r.horizon_years)} years` : ""}` : "Inputs the method needs are missing; no number is given."}</div></div>
      <div class="statement money"><div class="k">Financial impact</div><div class="big">${moneyText}${moneyText.endsWith("less") || moneyText.endsWith("more") ? "/yr" : ""}</div><div class="small">${esc(moneySmall)}</div></div>
    </div>
    <div><div class="eyebrow" style="margin-bottom:6px">Why (reason codes and evidence)</div>${reasons || '<span class="chip">overall fit</span>'}</div>
    <details class="record"><summary>Recommendation record</summary><pre class="mono">${record.map(([k, v]) => `${k}: ${esc(v)}`).join("\n")}</pre>${ce && ce.based_on ? `<div class="note">Estimate confidence ${esc(ce.level)}: ${esc(ce.based_on)} (PCAF data-quality score ${ce.pcaf_score}, where 1 is best and 5 is worst).</div>` : ""}${r.method_label ? `<div class="note">Method: ${esc(r.method_label)}.</div>` : ""}</details>
    <div class="explanation"><span data-text="${r.product}">${esc(r.explanation || "")}</span><div class="src"><span data-src="${r.product}">${r.source === "llm" ? pill("AI rewrite · checked", "blue") : pill("standard wording", "grey")} data as of ${esc(r.data_timestamp)}</span><span>${esc(r.customer_action || "")} <button class="btn small" data-rw="${r.product}">Rewrite</button></span></div></div>
    <div class="feedback"><span>Customer response:</span>${[["clicked", "Interested"], ["dismissed", "Dismiss"], ["not_interested", "Not interested"], ["already_have", "Already have it"]].map(([k, v]) => `<button class="btn small" data-fb="${r.product}|${k}">${v}</button>`).join("")}</div>
  </div>`;
}
/* ---------- products ---------- */
function renderProducts() {
  const ins = state.overview.insights;
  view.innerHTML = pageHead("Products", "6 green products · live snapshot") + `
    <div class="card"><div class="card-h"><h3>Product schedule</h3><span class="sub">medians over the customers shown each product</span></div><div class="card-b grid g-3">
    ${ins.products.map((p) => `<div class="box"><div class="k" style="color:${PRODUCT_COLOURS[p.product]}">${esc(p.name)}</div><div class="v">${num(p.expected_co2e_tonnes_per_year, 0)} t CO2e/yr expected</div>
      <div class="d">Eligible ${num(p.eligible)} · shown ${num(p.shown)} · review ${num(p.human_review)}<br>Expected take-ups ${num(p.expected_takeups, 0)} (mean propensity ${pct(p.mean_propensity_shown)})<br>
      Per take-up: ${p.co2e_kg_per_year_if_taken_up_median !== null ? kg(p.co2e_kg_per_year_if_taken_up_median) + " CO2e/yr" : "–"}${p.money_gbp_per_year_median !== null ? ` · ${gbp(p.money_gbp_per_year_median)}/yr running cost` : ""}${p.upfront_gbp_median ? ` · upfront ${gbp(p.upfront_gbp_median)}` : ""}${p.payback_years_median ? ` · payback ${num(p.payback_years_median, 0)} yrs` : ""}<br>
      Impact not computable for ${pct(p.insufficient_data_share, 0)} · estimate confidence ${confidenceWord(p.mean_data_quality_shown)} (mean PCAF score ${num(p.mean_data_quality_shown, 1)}, 1 best)</div></div>`).join("")}
    </div></div>
    <div class="card"><div class="card-h"><h3>Decision funnel</h3><span class="sub">every customer × product, by the first rule that decided it</span></div><div class="card-b">${funnelBars(ins.funnel)}</div>
    <div class="table-wrap"><table><thead><tr><th>Product</th>${Object.keys(FUNNEL_COLOURS).map((k) => `<th class="num">${esc(k)}</th>`).join("")}</tr></thead><tbody>
    ${ins.funnel.map((f) => `<tr><td>${esc(state.overview.products[f.product])}</td>${Object.keys(FUNNEL_COLOURS).map((k) => `<td class="num mono">${num(f[k] || 0)}</td>`).join("")}</tr>`).join("")}
    </tbody></table></div></div>`;
}

/* ---------- regions ---------- */
async function renderRegions(region = "") {
  const ins = state.overview.insights;
  const t = await api(`/api/trends?region=${encodeURIComponent(region)}`);
  const fp = ins.footprint;
  view.innerHTML = pageHead("Regions & trends", "Footprint, opportunity and monthly spend by region") + `
    <div class="card"><div class="card-b"><div class="chips-row" style="margin-bottom:14px"><b>Region:</b><button class="qchip ${region ? "" : "on"}" data-r="">All regions</button>${state.overview.regions.map((r) => `<button class="qchip ${r === region ? "on" : ""}" data-r="${esc(r)}">${esc(r)}</button>`).join("")}</div>
      <div class="eyebrow" style="margin-bottom:8px">Monthly spend by category · ${esc(t.region)}</div><div class="chart-box"><canvas id="trend"></canvas></div></div></div>
    <div class="card"><div class="card-h"><h3>Regions</h3><span class="sub">sorted by expected CO2e avoided per customer</span></div><div class="table-wrap"><table><thead><tr><th>Region</th><th class="num">Customers</th><th class="num">Median footprint</th><th class="num">Transport</th><th class="num">Home</th><th class="num">Suggestions</th><th class="num">Expected CO2e/yr</th><th class="num">Per customer</th><th>Top product</th></tr></thead><tbody>
      ${ins.regions.map((r) => `<tr><td>${esc(r.region)}</td><td class="num mono">${num(r.customers)}</td><td class="num mono">${kg(r.footprint_kg_median)}</td><td class="num mono">${kg(r.transport_kg_median)}</td><td class="num mono">${kg(r.home_kg_median)}</td><td class="num mono">${num(r.recommendations)}</td><td class="num mono">${num(r.expected_co2e_tonnes_per_year, 1)} t</td><td class="num mono">${num(r.expected_co2e_kg_per_customer, 1)} kg</td><td>${esc(r.top_product)}</td></tr>`).join("")}
    </tbody></table></div></div>
    <div class="grid g-2"><div class="card"><div class="card-h"><h3>Footprint by income band</h3><span class="sub">median kg CO2e a year</span></div><div class="card-b">
      ${(() => { const m = Math.max(...fp.by_income_band.map((b) => b.footprint_kg_median)); return fp.by_income_band.map((b) => `<div class="bar-row" style="grid-template-columns:110px 1fr 70px"><span>${label(b.income_band)}</span><div class="bar"><i style="width:${(100 * b.footprint_kg_median / m).toFixed(1)}%"></i></div><span class="mono" style="text-align:right">${kg(b.footprint_kg_median)}</span></div>`).join(""); })()}
      <div class="facts" style="margin-top:10px"><span>Transport share <b>${pct(fp.share_transport, 0)}</b></span></div></div></div>
      <div class="card"><div class="card-h"><h3>Footprint distribution</h3><span class="sub">deciles, kg CO2e a year</span></div><div class="card-b">
      ${fp.deciles_kg.map((v, i) => `<div class="bar-row" style="grid-template-columns:60px 1fr 70px"><span>P${(i + 1) * 10}</span><div class="bar green"><i style="width:${(100 * v / fp.deciles_kg[8]).toFixed(1)}%"></i></div><span class="mono" style="text-align:right">${kg(v)}</span></div>`).join("")}</div></div></div>`;
  view.querySelectorAll("[data-r]").forEach((b) => b.onclick = () => { clearCharts(); renderRegions(b.dataset.r); });
  const keep = ["fuel", "electricity", "gas", "public_transport", "home_improvement", "automotive"];
  chart($("#trend"), { type: "line", data: { labels: t.months, datasets: keep.filter((c) => t.series[c]).map((c) => ({ label: c.replace(/_/g, " "), data: t.series[c], borderColor: CATEGORY_COLOURS[c], backgroundColor: CATEGORY_COLOURS[c], borderWidth: 2, pointRadius: 0, tension: 0.25 })) },
    options: { maintainAspectRatio: false, interaction: { mode: "index", intersect: false }, plugins: { legend: { position: "bottom", labels: { boxWidth: 10 } }, tooltip: { callbacks: { label: (c) => `${c.dataset.label}: ${gbp(c.parsed.y)}` } } }, scales: { y: { ticks: { callback: (v) => "£" + num(v / 1000) + "k" }, grid: { color: "#f1f5f9" } }, x: { grid: { display: false } } } } });
}

/* ---------- memory ---------- */
async function renderMemory() {
  const mem = await api("/api/memory"), m = mem.replay;
  view.innerHTML = pageHead("Customer memory", `Responses and their effect on ${esc(mem.as_of)} · dismissed pauses ${mem.cooldown_days.dismissed} days, not interested ${mem.cooldown_days.not_interested} days`,
    `<button class="btn" id="reset">Forget app responses (${num(mem.app_responses)})</button>`) + `
    <div class="card"><div class="card-h"><h3>Memory in force today</h3><span class="sub">latest response per customer × product: CRM campaigns and the apps</span></div>
    <div class="table-wrap"><table><thead><tr><th>Product</th><th class="num">Customers contacted</th><th class="num">Paused · dismissed</th><th class="num">Paused · not interested</th><th class="num">Stopped · already has it</th><th class="num">Priority raised · interested</th></tr></thead><tbody>
    ${mem.products.map((r) => `<tr><td>${esc(r.name)}</td><td class="num mono">${num(r.responses)}</td><td class="num mono">${num(r.paused_dismissed)}</td><td class="num mono">${num(r.paused_not_interested)}</td><td class="num mono">${num(r.stopped_already_have)}</td><td class="num mono">${num(r.interested)}</td></tr>`).join("")}
    </tbody></table></div></div>
    <div class="card"><div class="card-h"><h3>Latest responses in the apps</h3></div>
    <div class="table-wrap"><table><thead><tr><th>Customer</th><th>Product</th><th>Response</th><th>Source</th></tr></thead><tbody>
    ${mem.recent.map((e) => `<tr class="click" data-id="${esc(e.customer_id)}"><td class="mono">${esc(e.customer_id)}</td><td>${esc(e.product)}</td><td>${pill(e.response.replace(/_/g, " "), "blue")}</td><td>${esc(e.source)}</td></tr>`).join("") || `<tr><td colspan="4" class="empty">None yet: respond to a suggestion in the customer app or on a customer page.</td></tr>`}
    </tbody></table></div></div>
    ${m ? `<div class="card"><div class="card-h"><h3>Check: replay of ${num(m.customers)} customers over ${m.rounds.length} months</h3><span class="sub">simulated responses · a paused product must never come back early</span></div>
    <div class="table-wrap"><table><thead><tr><th>Round</th><th>Date</th><th class="num">Shown</th><th class="num">Held back by memory</th><th class="num">Re-shown inside a pause</th><th>Responses</th></tr></thead><tbody>
    ${m.rounds.map((r) => `<tr><td class="mono">${r.round}</td><td class="mono">${esc(r.date)}</td><td class="num mono">${num(r.shown)}</td><td class="num mono">${num(r.suppressed_by_memory)}</td><td class="num">${r.reshown_within_cooldown === undefined ? "–" : pill(num(r.reshown_within_cooldown), r.reshown_within_cooldown === 0 ? "green" : "red")}</td><td>${Object.entries(r.responses).map(([k, v]) => `<span class="chip">${label(k)} ${num(v)}</span>`).join("")}</td></tr>`).join("")}
    </tbody></table></div></div>` : ""}`;
  $("#reset").onclick = async () => { await api("/api/memory/reset", {}); toast("App responses forgotten; suggestions restored."); renderMemory(); };
  view.querySelectorAll("tr.click").forEach((tr) => tr.onclick = () => location.hash = `#/customer/${tr.dataset.id}`);
}

/* ---------- models & checks ---------- */
function renderModels(product = "GP01") {
  const o = state.overview, names = o.products, ex = o.explanations || {}, tr = o.impact_truth_recovery || {};
  const models = o.models || {};
  const metricRow = (p) => {
    const t = models[p].test, c = t.lightgbm_calibrated, b = t.rules_baseline, l = t.logistic_regression, or = t.oracle_p_true;
    return `<tr><td>${esc(names[p])}</td><td class="num mono">${num(models[p].positives.test)}</td><td class="num mono">${num(b.roc_auc, 3)}</td><td class="num mono">${num(l.roc_auc, 3)}</td><td class="num mono"><b>${num(c.roc_auc, 3)}</b></td><td class="num mono">${num(or.roc_auc, 3)}</td><td class="num mono">${num(c.pr_auc, 3)}</td><td class="num mono">${num(c.lift_top10, 2)}×</td><td class="num mono">${num(c.ece, 3)}</td><td class="num mono">${num(models[p].mean_confidence_test, 2)}</td></tr>`;
  };
  const ab = (o.ablations || {})[product] || {};
  const fair = models[product]?.fairness_test || {};
  const llm = ex.llm_eval || {};
  const recovery = ["fuel_litres", "gas_kwh", "electricity_kwh", "transport_co2e_kg", "home_energy_co2e_kg"].filter((k) => tr[k]);
  const sens = (o.impact_sensitivity || []).filter((r) => r.co2e_swing_pct > 0 || r.money_swing_pct > 0).sort((a, b) => (b.co2e_swing_pct || 0) - (a.co2e_swing_pct || 0)).slice(0, 12);
  view.innerHTML = pageHead("Models & checks", "Held-out test results") + `
    <div class="card"><div class="card-h"><h3>FR3 propensity · test set (customers never seen in training)</h3><span class="sub">AUC: ranking · lift: take-up rate in the top 10% vs average · ECE: calibration error</span></div>
    <div class="table-wrap"><table><thead><tr><th>Product</th><th class="num">Test take-ups</th><th class="num">Rules AUC</th><th class="num">Logistic AUC</th><th class="num">LightGBM AUC</th><th class="num">Oracle AUC</th><th class="num">PR-AUC</th><th class="num">Lift top 10%</th><th class="num">ECE</th><th class="num">Confidence</th></tr></thead><tbody>
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
      ${o.gemini_sample ? kpi("Gemini guard pass", `${num(o.gemini_sample.passed_guards)}/${num(o.gemini_sample.replies)}`, `${esc(o.gemini_sample.model)} · used for Rewrite`, "good") : ""}
      ${kpi("Reading ease", llm.reading_ease_mean ? `${num(llm.reading_ease_mean.llm, 0)} vs ${num(llm.reading_ease_mean.template, 0)}` : num(ex.reading_ease_mean, 0), "Flesch: LLM vs template (higher is easier)")}
      ${kpi("Length", llm.words_mean ? `${num(llm.words_mean.llm, 0)} vs ${num(llm.words_mean.template, 0)}` : num(ex.words_mean, 0) + " words", "words: LLM vs template")}
    </div></div>
    <div class="card"><div class="card-h"><h3>FR4 impact inputs vs true usage</h3><span class="sub">31 Mar 2026 · visible spend only</span></div><div class="card-b grid g-${Math.min(recovery.length, 5) || 1}" style="grid-template-columns:repeat(${recovery.length || 1},1fr)">
      ${recovery.map((k) => kpi(label(k), pct(tr[k].within_20pct, 0), `within ±20% · median error ${num(tr[k].median_abs_pct_error, 1)}% · r = ${num(tr[k].correlation, 2)} · n = ${num(tr[k].customers)}`, tr[k].within_20pct >= 0.7 ? "good" : "")).join("")}
    </div></div>
    <div class="card"><div class="card-h"><h3>FR4 sensitivity</h3><span class="sub">mean per customer · low · base · high</span></div><div class="table-wrap"><table><thead><tr><th>Assumption</th><th>Range</th><th>Product</th><th class="num">CO2e low · base · high (kg/yr)</th><th class="num">Swing</th><th class="num">Money low · base · high (£/yr)</th><th class="num">Swing</th></tr></thead><tbody>
      ${sens.map((r) => `<tr><td>${label(r.assumption)}</td><td class="mono">${esc(r.range)}</td><td>${esc(names[r.product])}</td><td class="num mono">${num(r.co2e_low)} · ${num(r.co2e_base)} · ${num(r.co2e_high)}</td><td class="num mono">${num(r.co2e_swing_pct, 0)}%</td><td class="num mono">${num(r.money_low)} · ${num(r.money_base)} · ${num(r.money_high)}</td><td class="num mono">${num(r.money_swing_pct, 0)}%</td></tr>`).join("")}
    </tbody></table></div></div>`;
  view.querySelectorAll("[data-p]").forEach((b) => b.onclick = () => renderModels(b.dataset.p));
}

route();
