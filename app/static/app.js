const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmtTime = ts => new Date(ts * 1000).toLocaleString();
const fmtBytes = b => b > 1e6 ? (b / 1e6).toFixed(1) + " MB" : Math.round(b / 1e3) + " KB";

async function api(path, opts = {}) {
  const r = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts });
  const body = r.headers.get("content-type")?.includes("json") ? await r.json() : await r.text();
  if (!r.ok) throw new Error(body.error || body.detail || r.statusText);
  return body;
}

// ---------- routing ----------
const views = { setup: loadSetup, documents: loadDocuments, chat: () => $("#question").focus(), admin: loadAdmin };
let docTimer;
async function route() {
  let name = location.hash.slice(1);
  if (!views[name]) {
    const { configured } = await api("/api/health");
    name = configured ? "documents" : "setup";
    history.replaceState(null, "", "#" + name);
  }
  $$("main > section").forEach(s => (s.hidden = s.id !== "view-" + name));
  $$("nav a").forEach(a => a.classList.toggle("active", a.hash === "#" + name));
  clearInterval(docTimer);
  views[name]();
}
window.addEventListener("hashchange", route);

// ---------- setup ----------
const form = $("#setup-form");
const get = (o, path) => path.split(".").reduce((x, k) => x?.[k], o);
function set(o, path, v) {
  const ks = path.split("."); const last = ks.pop();
  ks.reduce((x, k) => (x[k] ??= {}), o)[last] = v;
}
function formToConfig() {
  const cfg = {};
  $$("[name]", form).forEach(el => set(cfg, el.name, el.type === "number" ? Number(el.value) : el.value));
  return cfg;
}
function updateVisibility() {
  $$("[data-show]", form).forEach(div => {
    const [k, v] = div.dataset.show.split("=");
    div.hidden = form.elements[k].value !== v;
  });
}
async function updateTokenHint() {
  const t = await api("/api/tokenizer/limits", { method: "POST", body: JSON.stringify(formToConfig()) });
  const asked = Number(form.elements["rag.chunk_tokens"].value);
  $("#token-hint").textContent = `Tokenizer: ${t.tokenizer}. This embedding model takes about ${t.model_limit} tokens per chunk` +
    (asked > t.effective_chunk_tokens ? `, so chunks will be capped at ${t.effective_chunk_tokens} tokens.` : ".");
}
async function loadSetup() {
  const cfg = await api("/api/config");
  $$("[name]", form).forEach(el => (el.value = get(cfg, el.name) ?? ""));
  updateVisibility();
  updateTokenHint();
}
form.addEventListener("change", () => { updateVisibility(); updateTokenHint(); });
$$("[data-test]", form).forEach(btn => btn.addEventListener("click", async () => {
  const out = $("#test-" + btn.dataset.test);
  out.className = "result"; out.textContent = "Testing…";
  try {
    const r = await api("/api/test/" + btn.dataset.test, { method: "POST", body: JSON.stringify(formToConfig()) });
    out.className = "result " + (r.ok ? "ok" : "bad");
    out.textContent = r.ok ? `✓ ${r.detail} (${r.ms} ms)` : `✗ ${r.error}`;
  } catch (e) { out.className = "result bad"; out.textContent = "✗ " + e.message; }
}));
form.addEventListener("submit", async e => {
  e.preventDefault();
  const out = $("#save-result");
  try {
    await api("/api/config", { method: "PUT", body: JSON.stringify(formToConfig()) });
    out.className = "result ok"; out.textContent = "Saved. You can upload documents now.";
  } catch (err) { out.className = "result bad"; out.textContent = err.message; }
});

// ---------- documents ----------
const statusClass = { ready: "ok", failed: "bad", processing: "warn", queued: "muted" };
async function loadDocuments() {
  const docs = await api("/api/documents");
  $("#doc-rows").innerHTML = docs.map(d => `<tr>
    <td>${esc(d.filename)}</td>
    <td class="${statusClass[d.status]}" title="${esc(d.error)}">${d.status}${d.error ? ": " + esc(d.error).slice(0, 80) : ""}</td>
    <td>${d.pages ?? ""}</td><td>${d.chunks ?? ""}</td><td>${fmtBytes(d.size)}</td><td>${fmtTime(d.created)}</td>
    <td><button class="small-btn" data-reindex="${d.id}">Re-index</button>
        <button class="small-btn" data-del="${d.id}">Delete</button></td></tr>`).join("")
    || `<tr><td colspan="7" class="muted">No documents yet.</td></tr>`;
  clearInterval(docTimer);
  if (docs.some(d => d.status === "queued" || d.status === "processing")) docTimer = setInterval(loadDocuments, 2000);
}
$("#doc-rows").addEventListener("click", async e => {
  const { del, reindex } = e.target.dataset;
  if (del) await api("/api/documents/" + del, { method: "DELETE" });
  if (reindex) await api(`/api/documents/${reindex}/reindex`, { method: "POST" });
  if (del || reindex) loadDocuments();
});
async function uploadFiles(files) {
  const fd = new FormData();
  [...files].forEach(f => fd.append("files", f));
  const r = await fetch("/api/documents", { method: "POST", body: fd });
  if (!r.ok) alert((await r.json()).detail);
  loadDocuments();
}
const dz = $("#dropzone");
$("#file-input").addEventListener("change", e => uploadFiles(e.target.files));
dz.addEventListener("dragover", e => { e.preventDefault(); dz.classList.add("over"); });
dz.addEventListener("dragleave", () => dz.classList.remove("over"));
dz.addEventListener("drop", e => { e.preventDefault(); dz.classList.remove("over"); uploadFiles(e.dataTransfer.files); });

// ---------- chat ----------
function addMsg(cls, html) {
  const div = document.createElement("div");
  div.className = "msg " + cls; div.innerHTML = html;
  $("#messages").append(div); div.scrollIntoView({ behavior: "smooth" });
  return div;
}
const renderHits = hits => hits.map((h, i) => `<div class="hit"><strong>[${i + 1}] ${esc(h.filename)}, page ${h.page}</strong>
  <span class="muted">score ${Number(h.score).toFixed(3)}</span><div>${esc(h.text)}</div></div>`).join("");
$("#chat-form").addEventListener("submit", async e => {
  e.preventDefault();
  const q = $("#question").value.trim(); if (!q) return;
  $("#question").value = "";
  addMsg("user", esc(q));
  const pending = addMsg("bot", "<span class='muted'>Thinking…</span>");
  try {
    const r = await api("/api/chat", { method: "POST", body: JSON.stringify({ question: q }) });
    pending.innerHTML = esc(r.answer) + `<details class="sources"><summary>${r.sources.length} sources ·
      retrieve ${r.timing_ms.retrieve} ms · generate ${r.timing_ms.generate} ms</summary>${renderHits(r.sources)}</details>`;
  } catch (err) { pending.innerHTML = `<span class="bad">${esc(err.message)}</span>`; }
});

// ---------- admin ----------
async function loadAdmin() {
  loadEvents(); loadQueries();
  $("#cards").innerHTML = "<span class='muted'>Running checks…</span>";
  const s = await api("/api/admin/status");
  const docs = Object.values(s.documents);
  const total = docs.reduce((a, d) => a + d.n, 0);
  const q = s.queries_24h;
  const card = (v, l, cls = "") => `<div class="card"><div class="v ${cls}">${v ?? "–"}</div><div class="l">${l}</div></div>`;
  $("#cards").innerHTML = [
    card(total, "documents"),
    card(s.documents.ready?.n ?? 0, "ready", "ok"),
    card(s.documents.failed?.n ?? 0, "failed", s.documents.failed ? "bad" : ""),
    card(s.queue, "in ingestion queue"),
    card(s.vectors, "vectors stored"),
    card(q.n, "queries (24h)"),
    card(q.avg_ms ? Math.round(q.avg_ms) + " ms" : null, "avg query latency"),
    card(q.errors ?? 0, "query errors (24h)", q.errors ? "bad" : ""),
    card(`${s.system.disk_free_gb} / ${s.system.disk_total_gb} GB`, "disk free"),
    card(Math.round(s.system.uptime_s / 60) + " min", "uptime"),
  ].join("");
  $("#component-rows").innerHTML = Object.entries(s.components).map(([name, c]) => `<tr>
    <td>${name.replace("_", " ")} <span class="muted">(${esc(s.settings[name])})</span></td>
    <td class="${c.ok ? "ok" : "bad"}">${c.ok ? "healthy" : "down"}</td><td>${c.ms} ms</td>
    <td>${esc(c.ok ? c.detail : c.error)}</td></tr>`).join("") +
    `<tr><td>llm <span class="muted">(${esc(s.settings.llm)})</span></td><td colspan="3" class="muted">use “Check LLM” below</td></tr>`;
}
async function loadEvents() {
  const ev = await api("/api/admin/events?level=" + $("#log-level").value);
  $("#events").textContent = ev.map(e => `${fmtTime(e.ts)}  ${e.level.toUpperCase().padEnd(5)} [${e.source}] ${e.message}`).join("\n") || "No events.";
}
async function loadQueries() {
  const qs = await api("/api/admin/queries?limit=50");
  const ms = v => v == null ? "" : Math.round(v) + " ms";
  $("#query-rows").innerHTML = qs.map(q => `<tr><td>${fmtTime(q.ts)}</td><td>${esc(q.question)}</td>
    <td>${ms(q.retrieve_ms)}</td><td>${ms(q.generate_ms)}</td><td>${ms(q.total_ms)}</td>
    <td class="${q.ok ? "ok" : "bad"}">${q.ok ? "ok" : esc(q.error)}</td></tr>`).join("")
    || `<tr><td colspan="6" class="muted">No queries yet.</td></tr>`;
}
$("#refresh").addEventListener("click", loadAdmin);
$("#log-level").addEventListener("change", loadEvents);
$("#check-llm").addEventListener("click", async () => {
  const out = $("#llm-result"); out.className = "result"; out.textContent = "Checking…";
  const r = await api("/api/admin/check-llm", { method: "POST" });
  out.className = "result " + (r.ok ? "ok" : "bad");
  out.textContent = r.ok ? `✓ “${r.detail}” in ${r.ms} ms` : `✗ ${r.error}`;
});
$("#reindex-all").addEventListener("click", async () => {
  const r = await api("/api/documents/reindex-all", { method: "POST" });
  alert(`Queued ${r.queued} documents for re-indexing.`);
});
$("#debug-form").addEventListener("submit", async e => {
  e.preventDefault();
  const out = $("#debug-hits"); out.innerHTML = "<span class='muted'>Searching…</span>";
  try {
    const r = await api("/api/admin/retrieve", { method: "POST",
      body: JSON.stringify({ question: $("#debug-q").value, k: Number($("#debug-k").value) }) });
    out.innerHTML = `<p class="muted">${r.hits.length} hits in ${r.ms} ms</p>` + renderHits(r.hits);
  } catch (err) { out.innerHTML = `<span class="bad">${esc(err.message)}</span>`; }
});

route();
