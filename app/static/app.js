const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmtTime = ts => new Date(ts * 1000).toLocaleString();
const fmtBytes = b => b > 1e6 ? (b / 1e6).toFixed(1) + " MB" : Math.round(b / 1e3) + " KB";

async function api(path, opts = {}) {
  const r = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts });
  const body = r.headers.get("content-type")?.includes("json") ? await r.json() : await r.text();
  if (!r.ok) {
    if (r.status === 401 && !path.startsWith("/api/auth/")) showAuth("login");
    if (r.status === 403 && /temporary password/.test(body.error)) showAuth("password");
    throw new Error(body.error || body.detail || r.statusText);
  }
  return body;
}

// ---------- auth ----------
let session = { user: null, must_change_password: false };
function showAuth(view) {
  $("#nav").hidden = $("#account").hidden = view === "login" || session.must_change_password || !session.user;
  $$("main > section").forEach(s => (s.hidden = s.id !== "view-" + view));
  $("#password-reason").textContent = session.must_change_password
    ? "You logged in with the temporary password. Choose your own password to continue." : "";
  clearInterval(docTimer);
}
async function refreshSession() {
  session = await api("/api/auth/status");
  $("#account-name").textContent = session.user ? "Signed in as " + session.user : "";
  return session;
}
$("#login-form").addEventListener("submit", async e => {
  e.preventDefault();
  const f = e.target;
  try {
    await api("/api/auth/login", { method: "POST", body: JSON.stringify({ username: f.username.value, password: f.password.value }) });
    f.password.value = ""; $("#login-error").textContent = "";
    route();
  } catch (err) { $("#login-error").textContent = err.message; }
});
$("#password-form").addEventListener("submit", async e => {
  e.preventDefault();
  const f = e.target, out = $("#password-result");
  if (f.new_password.value !== f.repeat.value) { out.className = "result bad"; out.textContent = "The new passwords don't match"; return; }
  try {
    await api("/api/auth/password", { method: "POST",
      body: JSON.stringify({ current_password: f.current_password.value, new_password: f.new_password.value }) });
    f.reset(); out.className = "result ok"; out.textContent = "Password changed.";
    history.replaceState(null, "", "#"); route();
  } catch (err) { out.className = "result bad"; out.textContent = err.message; }
});
$("#logout").addEventListener("click", async () => {
  await api("/api/auth/logout", { method: "POST" });
  history.replaceState(null, "", "#"); route();
});

// ---------- routing ----------
const views = { setup: loadSetup, documents: loadDocuments, chat: () => $("#question").focus(), admin: loadAdmin };
let docTimer;
async function route() {
  await refreshSession();
  if (!session.user) return showAuth("login");
  if (session.must_change_password) return showAuth("password");
  $("#nav").hidden = $("#account").hidden = false;
  let name = location.hash.slice(1);
  if (name === "password") { $("#password-result").textContent = ""; return showAuth("password"); }
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
// Upload one file per request, a few at a time, so each file gets its own progress bar.
const UPLOAD_CONCURRENCY = 3;
const uploads = [];
function renderUploads() {
  $("#upload-panel").hidden = !uploads.length;
  const n = s => uploads.filter(u => u.state === s).length;
  const active = n("waiting") + n("uploading");
  $("#upload-summary").textContent = `${uploads.length} files: ${n("queued")} queued for ingestion, ` +
    `${n("skipped")} skipped, ${n("error")} failed` + (active ? `, ${active} uploading…` : "");
  $("#upload-list").innerHTML = uploads.map(u => `<li><span class="name">${esc(u.name)}</span>
    ${u.state === "uploading" || u.state === "waiting" ? `<progress max="100" value="${u.pct}"></progress>` : ""}
    <span class="${{ queued: "ok", skipped: "warn", error: "bad" }[u.state] || "muted"}">${esc(u.msg || u.state)}</span></li>`).join("");
}
function sendOne(u) {
  return new Promise(resolve => {
    const fd = new FormData();
    fd.append("files", u.file, u.name);
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/documents");
    xhr.upload.onprogress = e => { if (e.lengthComputable) { u.pct = Math.round(e.loaded / e.total * 100); renderUploads(); } };
    xhr.onload = () => {
      try {
        const body = JSON.parse(xhr.responseText);
        if (xhr.status === 401) showAuth("login");
        if (xhr.status >= 400) { u.state = "error"; u.msg = body.detail || body.error || xhr.statusText; }
        else { const r = body.results[0]; u.state = r.status; u.msg = r.reason || r.status; }
      } catch { u.state = "error"; u.msg = xhr.statusText || "upload failed"; }
      resolve();
    };
    xhr.onerror = () => { u.state = "error"; u.msg = "network error"; resolve(); };
    u.state = "uploading"; xhr.send(fd);
  });
}
let pumping = false;
async function pump() {
  if (pumping) return; pumping = true;
  const worker = async () => {
    let u;
    while ((u = uploads.find(x => x.state === "waiting"))) { await sendOne(u); renderUploads(); loadDocuments(); }
  };
  await Promise.all(Array.from({ length: UPLOAD_CONCURRENCY }, worker));
  pumping = false;
}
function uploadFiles(items) {
  for (const { file, name } of items) {
    const n = name || file.webkitRelativePath || file.name;
    if (!n.toLowerCase().endsWith(".pdf")) continue;  // folders often contain other files; ignore them quietly
    uploads.push({ file, name: n, state: "waiting", pct: 0 });
  }
  renderUploads(); pump();
}
// Walk dropped folders recursively (Chrome, Edge, Firefox, Safari all support webkitGetAsEntry).
async function entriesToFiles(entry, path = "") {
  if (entry.isFile) return [{ file: await new Promise((ok, err) => entry.file(ok, err)), name: path + entry.name }];
  const reader = entry.createReader(), out = [];
  let batch;
  do {
    batch = await new Promise((ok, err) => reader.readEntries(ok, err));
    for (const e of batch) out.push(...await entriesToFiles(e, path + entry.name + "/"));
  } while (batch.length);
  return out;
}
const dz = $("#dropzone");
$("#pick-files").addEventListener("click", () => $("#file-input").click());
$("#pick-folder").addEventListener("click", () => $("#folder-input").click());
for (const id of ["#file-input", "#folder-input"]) {
  $(id).addEventListener("change", e => { uploadFiles([...e.target.files].map(file => ({ file }))); e.target.value = ""; });
}
dz.addEventListener("dragover", e => { e.preventDefault(); dz.classList.add("over"); });
dz.addEventListener("dragleave", () => dz.classList.remove("over"));
dz.addEventListener("drop", async e => {
  e.preventDefault(); dz.classList.remove("over");
  const entries = [...e.dataTransfer.items].map(i => i.webkitGetAsEntry?.()).filter(Boolean);
  if (entries.length) uploadFiles((await Promise.all(entries.map(en => entriesToFiles(en)))).flat());
  else uploadFiles([...e.dataTransfer.files].map(file => ({ file })));
});
$("#clear-uploads").addEventListener("click", () => {
  for (let i = uploads.length - 1; i >= 0; i--) if (!["waiting", "uploading"].includes(uploads[i].state)) uploads.splice(i, 1);
  renderUploads();
});
function showImport(r) {
  const skipped = r.results.filter(x => x.status === "skipped");
  $("#import-result").className = "result " + (skipped.length ? "warn" : "ok");
  $("#import-result").textContent = `${r.queued} queued, ${skipped.length} skipped` +
    (skipped.length ? ": " + skipped.slice(0, 5).map(x => `${x.file} (${x.reason})`).join("; ") : "");
  loadDocuments();
}
$("#import-urls").addEventListener("click", async () => {
  const urls = $("#url-list").value.split("\n").map(s => s.trim()).filter(Boolean);
  if (!urls.length) return;
  $("#import-result").className = "result"; $("#import-result").textContent = `Downloading ${urls.length}…`;
  try { showImport(await api("/api/documents/urls", { method: "POST", body: JSON.stringify({ urls }) })); $("#url-list").value = ""; }
  catch (err) { $("#import-result").className = "result bad"; $("#import-result").textContent = err.message; }
});
$("#import-folder").addEventListener("click", async () => {
  $("#import-result").className = "result"; $("#import-result").textContent = "Scanning…";
  try { showImport(await api("/api/documents/import-folder", { method: "POST" })); }
  catch (err) { $("#import-result").className = "result bad"; $("#import-result").textContent = err.message; }
});

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
async function loadKeys() {
  $("#api-base").textContent = location.origin + "/v1";
  $("#mcp-url").textContent = location.origin + "/mcp";
  const keys = await api("/api/admin/keys");
  $("#key-rows").innerHTML = keys.map(k => `<tr><td>${esc(k.name)}</td><td><code>${esc(k.prefix)}…</code></td>
    <td>${fmtTime(k.created)}</td><td>${k.last_used ? fmtTime(k.last_used) : "never"}</td>
    <td><button class="small-btn" data-revoke="${k.id}">Revoke</button></td></tr>`).join("")
    || `<tr><td colspan="5" class="muted">No API keys yet.</td></tr>`;
}
$("#key-form").addEventListener("submit", async e => {
  e.preventDefault();
  const { key } = await api("/api/admin/keys", { method: "POST", body: JSON.stringify({ name: $("#key-name").value || "default" }) });
  $("#new-key").innerHTML = `New key (copy it now, it won't be shown again): <code>${esc(key)}</code>`;
  $("#key-name").value = ""; loadKeys();
});
$("#key-rows").addEventListener("click", async e => {
  if (e.target.dataset.revoke) { await api("/api/admin/keys/" + e.target.dataset.revoke, { method: "DELETE" }); loadKeys(); }
});
async function loadAdmin() {
  loadEvents(); loadQueries(); loadKeys();
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
