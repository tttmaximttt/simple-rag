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
    const detail = typeof body.detail === "object" && !Array.isArray(body.detail) ? body.detail : null;
    throw Object.assign(new Error(body.error || detail?.message || body.detail || r.statusText), { status: r.status, detail });
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
const views = { setup: loadSetup, documents: loadDocuments, chat: () => $("#question").focus(), connect: loadConnect,
                admin: loadAdmin };
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
// Each model input gets a dropdown of the provider's models; "Other…" (or a failed listing) falls back to typing a name.
const OTHER = "__other__";
const pickers = $$('input[name$=".model"]', form).map(input => {
  const select = document.createElement("select");
  const hint = document.createElement("span");
  hint.className = "muted small";
  input.before(select);
  input.after(hint);
  select.hidden = true;
  // Runs before the form's change listener, which re-renders from input.value.
  select.addEventListener("change", () => {
    input.value = select.value === OTHER ? "" : select.value;
    if (select.value === OTHER) setTimeout(() => input.focus());
  });
  return { input, select, hint, component: input.name.split(".")[0], key: null, models: null, error: "" };
});
function renderPicker(p) {
  const cur = p.input.value;
  p.select.hidden = !p.models?.length;
  if (p.select.hidden) {
    p.input.hidden = false;
    p.hint.textContent = p.error ? `Couldn't list models (${p.error}). Type a model name.`
      : p.models ? "The provider listed no models. Type a model name." : "";
    return;
  }
  p.select.innerHTML = p.models.map(m => `<option value="${esc(m)}">${esc(m)}</option>`).join("") +
    `<option value="${OTHER}">Other…</option>`;
  p.select.value = p.models.includes(cur) ? cur : OTHER;
  p.input.hidden = p.select.value !== OTHER;
  p.hint.textContent = "";
}
async function refreshModels() {
  const cfg = formToConfig();
  await Promise.all(pickers.map(async p => {
    if (p.input.closest("[data-show]").hidden) return;
    // Refetch only when the provider's connection settings change, not when the model does.
    const { model, ...conn } = get(cfg, p.input.name.replace(/\.model$/, ""));
    const key = JSON.stringify(conn);
    if (key !== p.key) {
      p.key = key;
      p.models = null; p.error = "";
      renderPicker(p);  // drop the previous endpoint's list while loading
      p.hint.textContent = "Loading models…";
      const r = await api("/api/models/" + p.component, { method: "POST", body: JSON.stringify(cfg) })
        .catch(e => ({ ok: false, error: e.message }));
      if (p.key !== key) return;  // settings changed while loading
      p.models = r.ok ? r.models : null;
      p.error = r.ok ? "" : r.error;
    }
    renderPicker(p);
  }));
}
async function loadSetup() {
  const cfg = await api("/api/config");
  $$("[name]", form).forEach(el => (el.value = get(cfg, el.name) ?? ""));
  updateVisibility();
  updateTokenHint();
  refreshModels();
}
form.addEventListener("change", () => { updateVisibility(); updateTokenHint(); refreshModels(); });
$$("[data-test]", form).forEach(btn => btn.addEventListener("click", async () => {
  const out = $("#test-" + btn.dataset.test);
  out.className = "result"; out.textContent = "Testing…";
  try {
    const r = await api("/api/test/" + btn.dataset.test, { method: "POST", body: JSON.stringify(formToConfig()) });
    out.className = "result " + (r.ok ? "ok" : "bad");
    out.textContent = r.ok ? `✓ ${r.detail} (${r.ms} ms)` : `✗ ${r.error}`;
  } catch (e) { out.className = "result bad"; out.textContent = "✗ " + e.message; }
}));
let saving = false;
async function saveConfig(reindex) {
  if (saving) return;  // a double click must not queue every document twice
  const out = $("#save-result");
  const body = formToConfig();
  if (reindex !== undefined) body.reindex = reindex;
  saving = true;
  $$("button", out).concat($$('button[type="submit"]', form)).forEach(b => (b.disabled = true));
  try {
    await api("/api/config", { method: "PUT", body: JSON.stringify(body) });
    out.className = "result ok";
    out.textContent = reindex ? "Saved. Re-indexing all documents; see Documents for progress." : "Saved. You can upload documents now.";
  } catch (err) {
    out.className = "result bad";
    out.textContent = err.message;
    const d = err.detail;
    if (d?.code !== "reindex_required") return;
    // The stored vectors no longer match: offer to re-index, into a new collection if the model changed.
    const btn = (label, onclick) => Object.assign(document.createElement("button"),
      { type: "button", className: "small-btn", textContent: label, onclick });
    out.append(" ", btn(d.collection ? `Use collection “${d.collection}” and re-index ${d.documents} documents`
                                     : `Save and re-index ${d.documents} documents`, () => {
      if (d.collection) form.elements["vector_store.collection"].value = d.collection;
      saveConfig(true);
    }));
    if (!d.collection) out.append(" ", btn("Save without re-indexing", () => saveConfig(false)));
  } finally {
    saving = false;
    $('button[type="submit"]', form).disabled = false;
  }
}
form.addEventListener("submit", e => { e.preventDefault(); saveConfig(); });

// ---------- documents ----------
const statusClass = { ready: "ok", failed: "bad", processing: "warn", queued: "muted", cancelled: "muted" };
const active = d => d.status === "queued" || d.status === "processing";
function statusCell(d) {
  const p = d.progress;
  if (p) {
    const what = p.stage === "reading" ? `reading pages ${p.done}/${p.total}` : `embedding ${p.done}/${p.total} chunks`;
    return `processing · ${what}<progress max="${p.total || 1}" value="${p.done}"></progress>`;
  }
  if (d.queue_position) return `queued (#${d.queue_position})`;
  return d.status + (d.error ? ": " + esc(d.error).slice(0, 80) : "");
}
async function loadDocuments() {
  const docs = await api("/api/documents");
  $("#doc-rows").innerHTML = docs.map(d => `<tr>
    <td>${esc(d.filename)}</td>
    <td class="${statusClass[d.status]}" title="${esc(d.error)}">${statusCell(d)}</td>
    <td>${d.pages ?? ""}</td><td>${d.chunks ?? ""}</td><td>${fmtBytes(d.size)}</td><td>${fmtTime(d.created)}</td>
    <td>${active(d) ? `<button class="small-btn" data-cancel="${d.id}">Cancel</button>`
                    : `<button class="small-btn" data-reindex="${d.id}">Re-index</button>`}
        <button class="small-btn" data-del="${d.id}">Delete</button></td></tr>`).join("")
    || `<tr><td colspan="7" class="muted">No documents yet.</td></tr>`;
  clearInterval(docTimer);
  if (docs.some(active)) docTimer = setInterval(loadDocuments, 2000);
}
$("#doc-rows").addEventListener("click", async e => {
  const { del, reindex, cancel } = e.target.dataset;
  if (del) await api("/api/documents/" + del, { method: "DELETE" });
  if (reindex) await api(`/api/documents/${reindex}/reindex`, { method: "POST" });
  if (cancel) await api(`/api/documents/${cancel}/cancel`, { method: "POST" }).catch(() => {});
  if (del || reindex || cancel) loadDocuments();
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

// ---------- connect ----------
// Ready-to-paste setups for each client, filled with this server's address and a key created on the page.
let connectKey = "";  // shown once and kept only while the page is open
const CLIENTS = [
  { name: "Claude Code", where: "works with localhost", text: "Run in a terminal:",
    code: (u, k) => `claude mcp add --transport http rag-studio ${u}/mcp \\\n  --header "Authorization: Bearer ${k}"` },
  { name: "Claude Desktop", where: "works with localhost",
    text: "Settings → Developer → Edit Config, add this to claude_desktop_config.json, then restart Claude Desktop (needs Node.js for npx). With a public HTTPS address you can instead add a custom connector under Settings → Connectors with the claude.ai URL below.",
    code: (u, k) => JSON.stringify({ mcpServers: { "rag-studio": { command: "npx",
      args: ["-y", "mcp-remote", `${u}/mcp`, "--header", "Authorization: Bearer ${RAG_KEY}"], env: { RAG_KEY: k } } } }, null, 2) },
  { name: "claude.ai (web and mobile)", where: "needs public HTTPS",
    text: "Settings → Connectors → Add custom connector, paste this URL, then turn the connector on from a chat's tools menu. The key is in the URL, so use a key just for this.",
    code: (u, k) => `${u}/mcp?key=${k}` },
  { name: "ChatGPT connector (MCP)", where: "needs public HTTPS",
    text: "Settings → Connectors → Advanced → turn on developer mode, then create a connector with this URL and no authentication.",
    code: (u, k) => `${u}/mcp?key=${k}` },
  { name: "ChatGPT custom GPT (Actions)", where: "needs public HTTPS",
    text: "Create a GPT → Configure → Create new action → Import from URL with the address below. Authentication: API Key, type Bearer, with your key. Instructions: “Use searchDocuments or askDocuments to answer from my documents and cite the filename and page.”",
    code: (u) => `${u}/v1/openapi.json` },
  { name: "HTTP API (curl)", where: "any app or script",
    text: "Ask a question and get an answer with sources, or fetch matching passages to use with your own LLM.",
    code: (u, k) => `# Answer with sources\ncurl -s ${u}/v1/query -H "Authorization: Bearer ${k}" \\\n  -H "Content-Type: application/json" -d '{"question": "How long does charging take?", "top_k": 5}'\n\n` +
      `# Matching passages only\ncurl -s ${u}/v1/retrieve -H "Authorization: Bearer ${k}" \\\n  -H "Content-Type: application/json" -d '{"query": "charging time", "top_k": 8}'\n\n` +
      `# Upload PDFs, list documents\ncurl -s ${u}/v1/documents -H "Authorization: Bearer ${k}" -F files=@manual.pdf\ncurl -s ${u}/v1/documents -H "Authorization: Bearer ${k}"` },
  { name: "Python", where: "any app or script", text: "With httpx or requests:",
    code: (u, k) => `import httpx\n\nRAG = "${u}/v1"\nHEADERS = {"Authorization": "Bearer ${k}"}\n\n` +
      `r = httpx.post(f"{RAG}/query", headers=HEADERS, json={"question": "How long does charging take?"}, timeout=120)\n` +
      `r.raise_for_status()\nprint(r.json()["answer"])\nfor s in r.json()["sources"]:\n    print(s["filename"], "page", s["page"])` },
  { name: "Claude API (MCP connector)", where: "needs public HTTPS", text: "Let Claude call the tools from your own app:",
    code: (u, k) => `curl https://api.anthropic.com/v1/messages \\\n  -H "x-api-key: $ANTHROPIC_API_KEY" -H "anthropic-version: 2023-06-01" \\\n  -H "anthropic-beta: mcp-client-2025-11-20" -H "content-type: application/json" \\\n  -d '${JSON.stringify({ model: "claude-opus-5-5", max_tokens: 4096,
      mcp_servers: [{ type: "url", url: `${u}/mcp`, name: "rag-studio", authorization_token: k }],
      tools: [{ type: "mcp_toolset", mcp_server_name: "rag-studio" }],
      messages: [{ role: "user", content: "Search my documents for the charging time and summarize it." }] })}'` },
  { name: "OpenAI API (Responses, remote MCP)", where: "needs public HTTPS", text: "Let an OpenAI model call the tools:",
    code: (u, k) => `curl https://api.openai.com/v1/responses \\\n  -H "Authorization: Bearer $OPENAI_API_KEY" -H "Content-Type: application/json" \\\n  -d '${JSON.stringify({ model: "gpt-5",
      tools: [{ type: "mcp", server_label: "rag-studio", server_url: `${u}/mcp`, headers: { Authorization: `Bearer ${k}` },
                require_approval: "never" }], input: "Search my documents for the charging time and summarize it." })}'` },
];
function copyButton(text) {
  const b = Object.assign(document.createElement("button"), { type: "button", className: "small-btn copy", textContent: "Copy" });
  b.onclick = async () => {
    try { await navigator.clipboard.writeText(text()); b.textContent = "Copied"; }
    catch { b.textContent = "Select and copy"; }
    setTimeout(() => (b.textContent = "Copy"), 1500);
  };
  return b;
}
let connectBase = location.origin;
function renderConnect() {
  const u = connectBase, k = connectKey || "YOUR_KEY";
  const urls = [["MCP server", `${u}/mcp`], ["HTTP API", `${u}/v1`], ["OpenAPI spec", `${u}/v1/openapi.json`]];
  $("#connect-urls").replaceChildren(...urls.map(([label, url]) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${esc(label)}</td><td><code>${esc(url)}</code></td><td></td>`;
    tr.lastChild.append(copyButton(() => url));
    return tr;
  }));
  $("#connect-local").hidden = !/^https?:\/\/(localhost|127\.|\[::1\]|0\.0\.0\.0)/.test(u);
  const open = new Set($$("#connect-clients details[open]").map(d => d.dataset.name));
  $("#connect-clients").replaceChildren(...CLIENTS.map(c => {
    const d = Object.assign(document.createElement("details"), { className: "client" });
    d.dataset.name = c.name;
    d.open = open.has(c.name);
    const code = c.code(u, k);
    d.innerHTML = `<summary>${esc(c.name)} <span class="muted small">· ${esc(c.where)}</span></summary>
      <p class="small">${esc(c.text)}</p><pre></pre>`;
    const pre = $("pre", d);
    pre.textContent = code;
    pre.prepend(copyButton(() => code));
    return d;
  }));
}
async function loadConnect() {
  renderConnect();
  try { connectBase = (await api("/api/connect")).public_url || location.origin; } catch { /* keep the browser's */ }
  renderConnect();
}
$("#connect-key-form").addEventListener("submit", async e => {
  e.preventDefault();
  const { key } = await api("/api/admin/keys", { method: "POST", body: JSON.stringify({ name: $("#connect-key-name").value || "connect" }) });
  connectKey = key;
  $("#connect-key").replaceChildren("New key, filled into the examples below (copy it now, it won't be shown again): ",
    Object.assign(document.createElement("code"), { textContent: key }), " ", copyButton(() => key));
  $("#connect-key-name").value = "";
  renderConnect();
});

// ---------- admin ----------
async function loadKeys() {
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
