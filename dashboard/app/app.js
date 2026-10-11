/* Customer app: what one customer sees in their banking app. Reads /api/me/<id> (no scores, codes or internal flags). */

const $ = (sel) => document.querySelector(sel);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const num = (x, d = 0) => (x === null || x === undefined || Number.isNaN(x)) ? "–" : Number(x).toLocaleString("en-GB", { maximumFractionDigits: d, minimumFractionDigits: d });
const gbp = (x) => "£" + num(Math.abs(x));
const fmtDate = (iso) => iso ? new Date(iso).toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" }) : "";
const kg = (x) => (Math.abs(x) >= 1000 ? num(x / 1000, 1) + " t" : num(x, x < 10 ? 1 : 0) + " kg");
const NAMES = { fuel: "Fuel", electricity: "Electricity", gas: "Gas", public_transport: "Public transport", home_improvement: "Home improvement", automotive: "Car costs", utilities: "Utilities", other: "Everything else" };
const COLOURS = { fuel: "#b91c1c", electricity: "#1d4ed8", gas: "#b45309", public_transport: "#15803d", home_improvement: "#7c3aed", automotive: "#334155", utilities: "#0f766e", other: "#cbd5e1" };
let chart = null;

function toast(msg) {
  const t = $("#toast"); t.textContent = msg; t.classList.add("show");
  clearTimeout(t._h); t._h = setTimeout(() => t.classList.remove("show"), 3200);
}
async function api(path, body) {
  const res = await fetch(path, body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {});
  if (!res.ok) throw new Error((await res.json().catch(() => ({}))).error || res.statusText);
  return res.json();
}

function money(s) {
  const m = s.money_gbp_per_year;
  if (s.product === "GP05") return { v: "Not estimated", s: "Investments can go down as well as up." };
  if (m === null || m === undefined || Math.abs(m) < 5) return { v: "No change", s: "to your running costs" };
  const extra = [s.upfront_gbp ? `about ${gbp(s.upfront_gbp)} upfront` : "", s.payback_years && s.payback_years <= 30 ? `pays back in about ${num(s.payback_years)} years` : ""].filter(Boolean).join(", ");
  return { v: `${gbp(m)} ${m < 0 ? "less" : "more"} a year`, s: extra || "at today's prices" };
}

function window12(asOf) {
  if (!asOf) return "the last 12 months";
  const end = new Date(asOf), start = new Date(end);
  start.setFullYear(end.getFullYear() - 1); start.setDate(start.getDate() + 1);
  return `${fmtDate(start)} to ${fmtDate(end)}`;
}

function dots(c) {
  if (!c) return "";
  return `<span class="dots" title="${esc(c.based_on || "")}">${"●".repeat(c.dots)}${"○".repeat(c.of - c.dots)}</span> ${esc(c.level)}`;
}

function suggestion(s, i) {
  const m = money(s), c = s.confidence_of_estimate;
  const asOf = s.data_as_of ? fmtDate(s.data_as_of) : "";
  return `<div class="sugg" data-card="${s.product}">
    <div class="sugg-head"><div class="suggestion-topline"><span class="k">INSIGHT ${String(i + 1).padStart(2, "0")}</span><span class="product-name">${esc(s.name)}</span></div><h3>${esc(s.headline || s.name)}</h3></div>
    <div class="why-line"><b>Why you're seeing this:</b> ${esc((s.why_labels || []).join(", ") || "how you bank with us overall")}</div>
    <div class="tiles2">
      <div class="t2 carbon"><div class="k">Potential annual impact</div><div class="v">−${kg(s.co2e_kg_per_year)} CO₂e</div><div class="s">estimate${s.method_short ? ` · ${esc(s.method_short)}` : ""}${s.horizon_years > 1 ? ` · about ${kg(s.co2e_kg_horizon)} over ${num(s.horizon_years)} years` : ""}</div></div>
      <div class="t2 money"><div class="k">Estimated cost change</div><div class="v">${esc(m.v)}</div><div class="s">${esc(m.s)}</div></div>
    </div>
    <div class="meta-line">Data as of ${esc(asOf)}${c ? ` · Estimate confidence ${dots(c)}` : ""}</div>
    <div class="sugg-text">${esc(s.explanation)}</div>
    <details><summary>More detail</summary>
      <div class="meta" style="margin-top:6px">From your accounts with us, ${esc(window12(s.data_as_of))}:</div>
      <ul>${(s.why || []).map((w) => `<li>${esc(w.charAt(0).toUpperCase() + w.slice(1))}</li>`).join("") || "<li>How you bank with us overall</li>"}</ul>
      <div class="meta">${s.method_label ? `How the impact was estimated: ${esc(s.method_label)}. ` : ""}${c && c.based_on ? `Estimate confidence is ${esc(c.level.toLowerCase())} because it uses: ${esc(c.based_on)}. ` : ""}Figures are estimates, not offers.${s.disclaimer ? " " + esc(s.disclaimer) : ""}</div>
    </details>
    <div class="sugg-actions">
      <button class="btn primary" data-fb="${s.product}|clicked">${esc(s.action || "Find out more")}</button>
      <button class="btn" data-fb="${s.product}|dismissed">Remind me later</button>
      <button class="btn" data-fb="${s.product}|not_interested">Not for me</button>
      <button class="btn" data-fb="${s.product}|already_have">I already have this</button>
    </div>
  </div>`;
}

let current = null;
const DONE = {
  clicked: () => "Thanks, we've noted your interest and will send you the details in the app. This is not an application and nothing has been agreed.",
  dismissed: (m) => `OK, we'll remind you after ${fmtDate(m.until)}.`,
  not_interested: (m) => `Got it. We won't suggest this again before ${fmtDate(m.until)}.`,
  already_have: () => "Thanks for telling us. We won't suggest this again.",
};

function markDone(card, id, response, m) {
  const product = card.dataset.card;
  card.classList.toggle("closed", response !== "clicked");
  card.querySelector(".sugg-actions").innerHTML = `<div class="done">${esc(DONE[response](m || {}))} <button class="undo">Undo</button></div>`;
  card.querySelector(".undo").onclick = async () => {
    await api(`/api/customer/${id}/undo`, { product });
    const i = current.suggestions.findIndex((x) => x.product === product);
    const fresh = document.createElement("div");
    fresh.innerHTML = suggestion(current.suggestions[i], i);
    card.replaceWith(fresh.firstElementChild);
    bind(document.querySelector(`[data-card="${product}"]`), id);
  };
}

function bind(card, id) {
  const product = card.dataset.card;
  card.querySelectorAll("[data-fb]").forEach((b) => b.onclick = async () => {
    const response = b.dataset.fb.split("|")[1];
    card.querySelectorAll("button").forEach((x) => { x.disabled = true; });
    try {
      const doc = await api(`/api/customer/${id}/feedback`, { product, response });
      markDone(card, id, response, (doc.memory || []).find((x) => x.product === product));
    } catch (e) { toast(e.message); card.querySelectorAll("button").forEach((x) => { x.disabled = false; }); }
  });
}

async function load(id) {
  const d = await api(`/api/me/${id}`);
  $("#asof").textContent = `Up to ${d.as_of}`;
  const fp = d.footprint, total = Math.max(fp.total_kg, 1);
  const spendRows = Object.entries(d.spend_12m).filter(([, v]) => v > 0).sort((a, b) => b[1] - a[1]);
  const maxSpend = Math.max(...spendRows.map(([, v]) => v), 1);
  $("#screen").innerHTML = `
    <div class="hello">
      <p>Personalised insights from your banking activity, up to ${esc(d.as_of)}.</p>
    </div>
    <div class="panel footprint-panel"><div class="panel-heading"><span class="panel-icon">↗</span><h2>Your estimated footprint</h2><span class="period-pill">YEARLY</span></div>
      <div class="big">${kg(fp.total_kg)} <small>CO2e a year</small></div>
      ${fp.region_median_kg ? `<div class="compare">Similar households in ${esc(d.region)}: ${kg(fp.region_median_kg)}</div>` : ""}
      <div class="split"><i style="width:${(100 * fp.transport_kg / total).toFixed(1)}%;background:#b91c1c"></i><i style="width:${(100 * fp.home_kg / total).toFixed(1)}%;background:#b45309"></i></div>
      <div class="split-legend"><span>Travel <b>${kg(fp.transport_kg)}</b></span><span>Home energy <b>${kg(fp.home_kg)}</b></span></div>
    </div>
    <div class="section-label eyebrow section-heading"><span>Picked for you</span><span class="section-count">${d.suggestions.length} ${d.suggestions.length === 1 ? "insight" : "insights"}</span></div>
    ${d.suggestions.length ? d.suggestions.map(suggestion).join("") : `<div class="panel none">Nothing to suggest right now. We'll let you know if something could help you save energy or money.</div>`}
    <div class="panel spending-panel"><div class="panel-heading"><span class="panel-icon">◷</span><h2>Your spending patterns</h2></div><p class="panel-subtitle">A view of the last 12 months</p>
      ${spendRows.map(([c, v]) => `<div class="spend-row"><span>${NAMES[c] || c}</span><div class="bar"><i style="width:${(100 * v / maxSpend).toFixed(1)}%;background:${COLOURS[c]}"></i></div><span class="mono" style="text-align:right">${gbp(v)}</span></div>`).join("")}
      <div class="chart-sm"><canvas id="monthly"></canvas></div>
    </div>
    <div class="foot">Estimates use your account data and published UK emission factors and prices. Spending from other banks isn't included.</div>`;
  current = d;
  document.querySelectorAll(".sugg").forEach((card) => bind(card, id));
  (d.memory || []).filter((m) => m.response === "clicked" && m.source === "app").forEach((m) => {
    const card = document.querySelector(`[data-card="${m.product}"]`);
    if (card) markDone(card, id, "clicked", m);
  });
  if (chart) chart.destroy();
  if (window.Chart) {
    const keep = ["fuel", "electricity", "gas", "public_transport", "home_improvement", "automotive"].filter((c) => (d.monthly.series[c] || []).some((v) => v > 0));
    Chart.defaults.font.family = "Inter, sans-serif"; Chart.defaults.font.size = 10; Chart.defaults.color = "#64748b";
    chart = new Chart($("#monthly"), {
      type: "bar", data: {
        labels: d.monthly.months.map((m) => new Date(m + "-01").toLocaleString("en-GB", { month: "short" })),
        datasets: keep.map((c) => ({ label: NAMES[c], data: d.monthly.series[c], backgroundColor: COLOURS[c], stack: "s" }))
      },
      options: {
        maintainAspectRatio: false, plugins: { legend: { position: "bottom", labels: { boxWidth: 8, padding: 8 } }, tooltip: { callbacks: { label: (c) => `${c.dataset.label}: ${gbp(c.parsed.y)}` } } },
        scales: { x: { stacked: true, grid: { display: false } }, y: { stacked: true, ticks: { callback: (v) => "£" + num(v) }, grid: { color: "#f1f5f9" } } }
      }
    });
  }
  $("#staff").href = `../#/customer/${id}`;
}

async function start() {
  const list = await api("/api/customers?status=with&limit=40");
  const ids = list.rows.map((r) => r.customer_id);
  const wanted = location.hash.replace("#", "") || ids[0];
  if (!ids.includes(wanted)) ids.unshift(wanted);
  $("#who").innerHTML = ids.map((c) => `<option ${c === wanted ? "selected" : ""}>${c}</option>`).join("");
  $("#who").onchange = () => { location.hash = $("#who").value; };
  window.addEventListener("hashchange", () => load(location.hash.replace("#", "")));
  try { await load(wanted); } catch (e) { $("#screen").innerHTML = `<div class="none">Could not load: ${esc(e.message)}</div>`; }
}
start();
