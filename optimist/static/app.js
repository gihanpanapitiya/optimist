/* Science Assistant UI: chat on the left, workspace tree and viewer on the right. */
"use strict";

const $ = (sel) => document.querySelector(sel);
const store = {
  get(key, fallback) { try { const v = localStorage.getItem(key); return v === null ? fallback : JSON.parse(v); } catch { return fallback; } },
  set(key, value) { try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* storage unavailable */ } },
};

const state = {
  threadId: new URLSearchParams(location.search).get("thread") || store.get("threadId", null) || newId(),
  stream: null,              // the in-flight run: {threadId, abort, live}
  threads: [],
  workspace: "",
  tree: null,
  files: new Map(),          // rel path -> {size, mtime}
  knownPaths: null,          // paths seen at first load; anything newer is "fresh"
  freshPaths: new Set(),
  expanded: new Set(store.get("expanded", [""])),
  selected: null,
  viewerMode: "rendered",
  pollTimer: null,
  pendingApproval: null,     // the unanswered approval card, if any
};
store.set("threadId", state.threadId);

function newId() {
  return (crypto.randomUUID && crypto.randomUUID()) || String(Date.now()) + Math.random().toString(16).slice(2);
}

/* ------------------------------------------------------------------ */
/* markdown                                                            */
/* ------------------------------------------------------------------ */

const hasMarked = typeof window.marked !== "undefined";
if (hasMarked) marked.setOptions({ gfm: true, breaks: false });

function renderMarkdown(text) {
  if (!hasMarked) return `<pre>${escapeHtml(text)}</pre>`;
  const html = marked.parse(text);
  return window.DOMPurify ? DOMPurify.sanitize(html) : html;
}

function highlightWithin(el) {
  if (!window.hljs) return;
  el.querySelectorAll("pre code").forEach((block) => {
    if (!block.dataset.hl) { hljs.highlightElement(block); block.dataset.hl = "1"; }
  });
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

/* Turn workspace paths mentioned in text into links that open the viewer. */
function linkifyPaths(root) {
  if (!state.workspace) return;
  const prefix = state.workspace.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const absRe = new RegExp(`${prefix}/[^\\s)'"\`<>\\]]+`, "g");

  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
    acceptNode: (n) => (n.parentElement.closest("a, pre, .file-link") ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT),
  });
  const targets = [];
  while (walker.nextNode()) if (absRe.test(walker.currentNode.nodeValue)) targets.push(walker.currentNode);

  for (const node of targets) {
    const frag = document.createDocumentFragment();
    let last = 0;
    node.nodeValue.replace(absRe, (match, offset) => {
      const clean = match.replace(/[.,;:]+$/, "");
      frag.append(node.nodeValue.slice(last, offset), fileLink(clean.slice(state.workspace.length + 1), clean));
      last = offset + clean.length;
    });
    frag.append(node.nodeValue.slice(last));
    node.replaceWith(frag);
  }

  // Inline code naming a workspace file, e.g. `literature/x/final_report.md`.
  root.querySelectorAll("code").forEach((code) => {
    if (code.closest("pre, a")) return;
    // The agent writes workspace paths as "/literature/...": the workspace is its root.
    const rel = code.textContent.trim().replace(/^\.?\//, "");
    if (state.files.has(rel)) code.replaceWith(fileLink(rel, rel));
  });
}

function fileLink(rel, label) {
  const a = document.createElement("a");
  a.className = "file-link";
  a.textContent = label.startsWith(state.workspace) ? "…/" + rel : label;
  a.title = `Open ${rel}`;
  a.addEventListener("click", (e) => { e.preventDefault(); openFile(rel); revealInTree(rel); });
  return a;
}

/* ------------------------------------------------------------------ */
/* chat                                                                */
/* ------------------------------------------------------------------ */

const messagesEl = $("#messages");
const emptyEl = $("#empty");
const inputEl = $("#input");

const turn = { segment: null, segmentText: "", cards: new Map(), renderQueued: false };

function scrollToBottom(force = false) {
  const nearBottom = messagesEl.scrollHeight - messagesEl.scrollTop - messagesEl.clientHeight < 160;
  if (force || nearBottom) messagesEl.scrollTop = messagesEl.scrollHeight;
}

function addUser(text) {
  emptyEl.classList.add("hidden");
  const el = document.createElement("div");
  el.className = "msg user";
  el.innerHTML = `<div class="bubble"></div>`;
  el.firstChild.textContent = text;
  messagesEl.append(el);
  scrollToBottom(true);
}

function startSegment() {
  emptyEl.classList.add("hidden");
  const el = document.createElement("div");
  el.className = "msg assistant";
  el.innerHTML = `<div class="content md"></div>`;
  messagesEl.append(el);
  turn.segment = el.firstChild;
  turn.segmentText = "";
  return turn.segment;
}

function renderSegment(final = false) {
  if (!turn.segment) return;
  turn.segment.innerHTML = renderMarkdown(turn.segmentText);
  turn.segment.classList.toggle("cursor", !final);
  if (final) { highlightWithin(turn.segment); linkifyPaths(turn.segment); }
  scrollToBottom();
}

function onToken(text) {
  if (!turn.segment) startSegment();
  turn.segmentText += text;
  if (!turn.renderQueued) {
    turn.renderQueued = true;
    requestAnimationFrame(() => { turn.renderQueued = false; renderSegment(false); });
  }
}

function onMessage(text) {
  // The finished message is authoritative; it also covers providers that don't stream.
  if (text && text.trim()) {
    if (!turn.segment) startSegment();
    turn.segmentText = text;
    renderSegment(true);
  } else if (turn.segment) {
    turn.segment.parentElement.remove();
  }
  turn.segment = null;
  turn.segmentText = "";
}

const TOOL_LABELS = {
  search_literature: (a) => a.query,
  deep_literature_research: (a) => a.question,
  task: (a) => `${a.subagent_type || ""}: ${a.description}`,
  ls: (a) => a.path || "/",
  read_file: (a) => a.file_path,
  write_file: (a) => a.file_path,
  edit_file: (a) => a.file_path,
  delete: (a) => a.file_path,
  glob: (a) => a.pattern,
  grep: (a) => a.pattern,
  execute: (a) => a.command,
};

function onToolCall({ id, name, args }) {
  emptyEl.classList.add("hidden");
  const card = document.createElement("details");
  card.className = "tool running";
  const label = (TOOL_LABELS[name] || ((a) => JSON.stringify(a)))(args || {}) || "";
  card.innerHTML = `
    <summary>
      <span class="status"></span>
      <span class="tname"></span>
      <span class="targs"></span>
      <svg class="chev" viewBox="0 0 24 24"><path d="M9 6l6 6-6 6"/></svg>
    </summary>
    <div class="live hidden"></div>
    <div class="tbody">
      <div class="label">Arguments</div><pre class="args"></pre>
      <div class="plabel label hidden">Progress</div><div class="progress"></div>
      <div class="rlabel label hidden">Result</div><pre class="result hidden"></pre>
    </div>`;
  card.querySelector(".tname").textContent = name;
  card.querySelector(".targs").textContent = label;
  card.querySelector(".args").textContent = JSON.stringify(args, null, 2);
  messagesEl.append(card);
  turn.cards.set(id, card);
  scrollToBottom();
}

function onProgress({ tool_call_id, message }) {
  const card = turn.cards.get(tool_call_id);
  if (!card) return;
  const line = document.createElement("div");
  line.textContent = message;
  const progress = card.querySelector(".progress");
  progress.append(line);
  progress.scrollTop = progress.scrollHeight;
  card.querySelector(".plabel").classList.remove("hidden");
  const live = card.querySelector(".live");
  live.textContent = message;
  live.classList.toggle("hidden", card.open || !card.classList.contains("running"));
}

function onToolResult({ id, content, status }) {
  const card = turn.cards.get(id);
  if (!card) return;
  card.classList.remove("running");
  card.classList.add(status === "error" ? "err" : "ok");
  card.querySelector(".live").classList.add("hidden");
  const result = card.querySelector(".result");
  result.textContent = content;
  result.classList.remove("hidden");
  card.querySelector(".rlabel").classList.remove("hidden");
  linkifyPaths(card.querySelector(".tbody"));
  refreshTree();
}

function onError(message) {
  const el = document.createElement("div");
  el.className = "msg assistant error";
  el.innerHTML = `<div class="content"></div>`;
  el.firstChild.textContent = message;
  messagesEl.append(el);
  scrollToBottom(true);
}

function dispatch(event) {
  switch (event.type) {
    case "user": return addUser(event.text);
    case "token": return onToken(event.text);
    case "message": return onMessage(event.text);
    case "tool_call": return onToolCall(event);
    case "progress": return onProgress(event);
    case "tool_result": return onToolResult(event);
    case "approval": return onApproval(event);
    case "error": return onError(event.message);
  }
}

/* Tool calls waiting for the user (shell commands, deletions). The run has
   ended; answering posts the decisions and resumes it. */
function onApproval({ interrupts }) {
  finishTurn({ pending: true });
  const el = document.createElement("div");
  el.className = "approval";
  el.innerHTML = `<div class="atitle">Approval needed</div><div class="actions"></div>
    <input class="reason" type="text" placeholder="Reason, if rejecting (optional)">
    <div class="abuttons"><button class="approve">Approve</button><button class="reject">Reject</button></div>`;
  const list = el.querySelector(".actions");
  for (const i of interrupts) {
    for (const a of i.actions) {
      const row = document.createElement("div");
      row.className = "action";
      row.innerHTML = `<span class="tname"></span><pre></pre>`;
      row.querySelector(".tname").textContent = a.name;
      row.querySelector("pre").textContent = (TOOL_LABELS[a.name] || ((x) => JSON.stringify(x, null, 2)))(a.args || {});
      list.append(row);
    }
  }
  const answer = (type) => {
    const message = el.querySelector(".reason").value.trim();
    const decision = type === "reject" && message ? { type, message } : { type };
    const decisions = Object.fromEntries(interrupts.map((i) => [i.id, i.actions.map(() => decision)]));
    el.querySelectorAll("button, input").forEach((b) => { b.disabled = true; });
    el.classList.add(type === "approve" ? "approved" : "rejected");
    el.querySelector(".atitle").textContent = type === "approve" ? "Approved" : "Rejected";
    state.pendingApproval = null;
    resume(decisions);
  };
  el.querySelector(".approve").addEventListener("click", () => answer("approve"));
  el.querySelector(".reject").addEventListener("click", () => answer("reject"));
  messagesEl.append(el);
  state.pendingApproval = el;
  scrollToBottom(true);
}

/* A run keeps going on the server when you switch chats. `stream.live` says
   whether its events still belong on screen; once you leave, they don't. */
async function send(text) {
  if (state.stream || !text.trim()) return;
  // A new message instead of an answer drops the pending tool calls.
  state.pendingApproval?.remove();
  state.pendingApproval = null;
  addUser(text);
  await runStream({ thread_id: state.threadId, message: text });
}

function resume(decisions) {
  return runStream({ thread_id: state.threadId, decisions });
}

async function runStream(body) {
  if (state.stream) return;
  const stream = { threadId: state.threadId, abort: new AbortController(), live: true };
  state.stream = stream;
  updateComposer();
  renderThreads();
  startPolling();

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: stream.abort.signal,
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buffer.indexOf("\n\n")) >= 0) {
        const chunk = buffer.slice(0, idx);
        buffer = buffer.slice(idx + 2);
        const data = chunk.split("\n").filter((l) => l.startsWith("data: ")).map((l) => l.slice(6)).join("\n");
        if (!data) continue;
        const event = JSON.parse(data);
        if (event.type === "thread") loadThreads();   // a new chat appears in the sidebar right away
        else if (stream.live) dispatch(event);
      }
    }
  } catch (err) {
    if (stream.live) onError(err.name === "AbortError" ? "Stopped." : String(err));
  } finally {
    if (stream.live) finishTurn({ pending: Boolean(state.pendingApproval) });
    state.stream = null;
    stopPolling();
    updateComposer();
    refreshTree();
    loadThreads();
    // Came back to this chat mid-run: the screen shows history as of the switch.
    if (!stream.live && stream.threadId === state.threadId) showThread(state.threadId, { force: true });
  }
}

function finishTurn({ pending = false } = {}) {
  if (turn.segment) renderSegment(true);
  turn.segment = null;
  if (pending) return;   // cards awaiting approval stay running until answered
  for (const card of turn.cards.values()) {
    if (card.classList.contains("running")) { card.classList.remove("running"); card.classList.add("err"); }
  }
}

function startPolling() {
  clearInterval(state.pollTimer);
  // Deep research writes files throughout a run; watch them appear.
  state.pollTimer = setInterval(refreshTree, 3000);
}
function stopPolling() { clearInterval(state.pollTimer); }

const DEFAULT_PLACEHOLDER = inputEl.placeholder;

function updateComposer() {
  const running = Boolean(state.stream);
  const here = running && state.stream.threadId === state.threadId;
  $("#send").classList.toggle("hidden", here);
  $("#stop").classList.toggle("hidden", !here);
  $("#send").disabled = running || !inputEl.value.trim();
  inputEl.placeholder = running && !here
    ? "Another chat is running. You can send once it finishes."
    : DEFAULT_PLACEHOLDER;
  $("#running-note").classList.toggle("hidden", !(here && !state.stream.live));
}

async function loadHistory() {
  try {
    const events = await (await fetch(`/api/history?thread_id=${encodeURIComponent(state.threadId)}`)).json();
    for (const e of events) dispatch(e);
    if (!state.pendingApproval) turn.cards.forEach((card) => { if (card.classList.contains("running")) { card.classList.remove("running"); card.classList.add("ok"); } });
    scrollToBottom(true);
  } catch { /* unknown thread: nothing to show */ }
}

function clearChat() {
  messagesEl.querySelectorAll(".msg, .tool, .approval").forEach((el) => el.remove());
  state.pendingApproval = null;
  turn.cards.clear();
  turn.segment = null;
  turn.segmentText = "";
  emptyEl.classList.remove("hidden");
}

function setThread(id) {
  // Leaving the chat whose run is streaming: let it finish in the background.
  if (state.stream && state.stream.threadId !== id) state.stream.live = false;
  state.threadId = id;
  store.set("threadId", id);
  history.replaceState(null, "", `?thread=${encodeURIComponent(id)}${location.hash}`);
}

async function showThread(id, { force = false } = {}) {
  if (id === state.threadId && !force) return;
  setThread(id);
  if (state.stream && state.stream.threadId === id) state.stream.live = false;
  clearChat();
  updateComposer();
  renderThreads();
  await loadHistory();
  inputEl.focus();
}

function newChat() {
  setThread(newId());
  clearChat();
  updateComposer();
  renderThreads();
  inputEl.focus();
}

/* ------------------------------------------------------------------ */
/* conversation list                                                   */
/* ------------------------------------------------------------------ */

const threadListEl = $("#thread-list");
const threadFilterEl = $("#thread-filter");

async function loadThreads() {
  try { state.threads = await (await fetch("/api/threads")).json(); } catch { return; }
  renderThreads();
}

function dayGroup(ts) {
  const day = new Date(ts * 1000);
  day.setHours(0, 0, 0, 0);
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  const days = Math.round((today - day) / 864e5);
  if (days <= 0) return "Today";
  if (days === 1) return "Yesterday";
  if (days < 7) return "Previous 7 days";
  if (days < 30) return "Previous 30 days";
  return day.toLocaleDateString(undefined, { month: "long", year: "numeric" });
}

function renderThreads() {
  const query = threadFilterEl.value.trim().toLowerCase();
  const groups = new Map();
  for (const t of state.threads) {
    if (query && !t.title.toLowerCase().includes(query)) continue;
    const label = dayGroup(t.updated_at);
    if (!groups.has(label)) groups.set(label, []);
    groups.get(label).push(t);
  }
  threadListEl.innerHTML = "";
  if (!groups.size) {
    threadListEl.innerHTML = `<div class="thread-empty">${query ? "No matching chats." : "No conversations yet."}</div>`;
    return;
  }
  for (const [label, items] of groups) {
    const head = document.createElement("div");
    head.className = "thread-group";
    head.textContent = label;
    threadListEl.append(head);
    for (const t of items) threadListEl.append(threadItem(t));
  }
}

function threadItem(t) {
  const running = t.running || (state.stream && state.stream.threadId === t.id);
  const el = document.createElement("div");
  el.className = "thread-item" + (t.id === state.threadId ? " active" : "") + (running ? " running" : "");
  el.title = t.title;
  el.innerHTML = `
    <span class="t-dot" title="Running"></span>
    <span class="t-title"></span>
    <span class="t-actions">
      <button class="t-btn" data-act="rename" title="Rename"><svg viewBox="0 0 24 24"><path d="M12 20h9M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4z"/></svg></button>
      <button class="t-btn danger" data-act="delete" title="Delete"><svg viewBox="0 0 24 24"><path d="M3 6h18M8 6V4h8v2M19 6l-1 14H6L5 6"/></svg></button>
    </span>`;
  el.querySelector(".t-title").textContent = t.title;
  el.addEventListener("click", (e) => {
    const act = e.target.closest("[data-act]");
    if (act && act.dataset.act === "rename") return startRename(el, t);
    if (act && act.dataset.act === "delete") return deleteThread(t);
    if (!el.querySelector("input")) showThread(t.id);
  });
  el.querySelector(".t-title").addEventListener("dblclick", (e) => { e.stopPropagation(); startRename(el, t); });
  return el;
}

function startRename(el, t) {
  const titleEl = el.querySelector(".t-title");
  const input = document.createElement("input");
  input.className = "t-input";
  input.value = t.title;
  titleEl.replaceWith(input);
  input.focus();
  input.select();
  let done = false;
  const finish = async (save) => {
    if (done) return;
    done = true;
    const title = input.value.trim();
    if (save && title && title !== t.title) {
      const res = await fetch(`/api/threads/${encodeURIComponent(t.id)}`, {
        method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ title }),
      });
      if (res.ok) t.title = title;
    }
    renderThreads();
  };
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") finish(true);
    if (e.key === "Escape") finish(false);
  });
  input.addEventListener("blur", () => finish(true));
  input.addEventListener("click", (e) => e.stopPropagation());
}

async function deleteThread(t) {
  if (!confirm(`Delete “${t.title}”?\n\nThe conversation is removed permanently. Files it created stay in the workspace.`)) return;
  const res = await fetch(`/api/threads/${encodeURIComponent(t.id)}`, { method: "DELETE" });
  if (res.status === 409) { alert("This conversation is still running. Stop it first."); return; }
  if (!res.ok) { alert(`Could not delete: HTTP ${res.status}`); return; }
  state.threads = state.threads.filter((x) => x.id !== t.id);
  if (t.id === state.threadId) newChat();
  else renderThreads();
}

function setSidebar(collapsed) {
  document.body.classList.toggle("sidebar-collapsed", collapsed);
  store.set("sidebarCollapsed", collapsed);
}

threadFilterEl.addEventListener("input", renderThreads);
$("#sidebar-toggle").addEventListener("click", () => setSidebar(!document.body.classList.contains("sidebar-collapsed")));
$("#sidebar-new").addEventListener("click", newChat);
setSidebar(store.get("sidebarCollapsed", matchMedia("(max-width: 860px)").matches));

$("#composer").addEventListener("submit", (e) => {
  e.preventDefault();
  const text = inputEl.value;
  inputEl.value = "";
  autosize();
  send(text);
});
inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); $("#composer").requestSubmit(); }
});
inputEl.addEventListener("input", () => { autosize(); updateComposer(); });
function autosize() { inputEl.style.height = "auto"; inputEl.style.height = Math.min(inputEl.scrollHeight, 220) + "px"; }
$("#stop").addEventListener("click", () => state.stream && state.stream.abort.abort());
$("#new-chat").addEventListener("click", newChat);
document.querySelectorAll(".chip").forEach((chip) => chip.addEventListener("click", () => { if (!state.stream) send(chip.textContent); }));
messagesEl.addEventListener("toggle", (e) => {
  if (e.target.matches(".tool")) e.target.querySelector(".live").classList.toggle("hidden", e.target.open || !e.target.classList.contains("running"));
}, true);

/* ------------------------------------------------------------------ */
/* file tree                                                           */
/* ------------------------------------------------------------------ */

const treeEl = $("#tree");
const filterEl = $("#filter");

const ICONS = {
  dir: '<path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>',
  md: '<path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"/><path d="M14 3v6h6M8 13h8M8 17h5"/>',
  json: '<path d="M8 3H7a2 2 0 0 0-2 2v5a2 2 0 0 1-2 2 2 2 0 0 1 2 2v5a2 2 0 0 0 2 2h1M16 3h1a2 2 0 0 1 2 2v5a2 2 0 0 0 2 2 2 2 0 0 0-2 2v5a2 2 0 0 1-2 2h-1"/>',
  pdf: '<path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"/><path d="M14 3v6h6"/>',
  img: '<rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="9" cy="9" r="2"/><path d="M21 15l-5-5L5 21"/>',
  txt: '<path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"/><path d="M14 3v6h6"/>',
};

function iconFor(node) {
  if (node.type === "dir") return "dir";
  const ext = node.name.split(".").pop().toLowerCase();
  if (ext === "md") return "md";
  if (ext === "json") return "json";
  if (ext === "pdf") return "pdf";
  if (["png", "jpg", "jpeg", "gif", "svg", "webp"].includes(ext)) return "img";
  return "txt";
}

function formatSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

async function refreshTree() {
  try {
    const tree = await (await fetch("/api/tree")).json();
    const files = new Map();
    (function collect(n) {
      if (n.type === "file") files.set(n.path, { size: n.size, mtime: n.mtime });
      (n.children || []).forEach(collect);
    })(tree);

    if (state.knownPaths === null) state.knownPaths = new Set(files.keys());
    else for (const p of files.keys()) if (!state.knownPaths.has(p)) state.freshPaths.add(p);

    // Reload the open file if it changed, e.g. a report still being written.
    const prev = state.selected && state.files.get(state.selected);
    const now = state.selected && files.get(state.selected);
    state.tree = tree;
    state.files = files;
    renderTree();
    if (prev && now && prev.mtime !== now.mtime) openFile(state.selected, { keepScroll: true });
  } catch { /* server unavailable; keep the last tree */ }
}

function renderTree() {
  const query = filterEl.value.trim().toLowerCase();
  treeEl.innerHTML = "";
  const children = state.tree ? state.tree.children : [];
  if (!children.length) {
    treeEl.innerHTML = `<div class="tree-empty">The workspace is empty.<br>Deep research reports will appear here.</div>`;
    return;
  }
  const frag = document.createDocumentFragment();
  for (const child of children) renderNode(child, 0, query, frag);
  if (!frag.childNodes.length) treeEl.innerHTML = `<div class="tree-empty">No files match “${escapeHtml(query)}”.</div>`;
  else treeEl.append(frag);
}

function matches(node, query) {
  if (!query) return true;
  if (node.type === "file") return node.path.toLowerCase().includes(query);
  return (node.children || []).some((c) => matches(c, query));
}

function renderNode(node, depth, query, parent) {
  if (!matches(node, query)) return;
  const isDir = node.type === "dir";
  const open = isDir && (query ? true : state.expanded.has(node.path));
  const kind = iconFor(node);
  const row = document.createElement("div");
  row.className = "row" + (open ? " open" : "") + (node.path === state.selected ? " selected" : "") +
    (state.freshPaths.has(node.path) ? " fresh" : "");
  row.style.paddingLeft = `${8 + depth * 14}px`;
  row.dataset.path = node.path;
  row.title = node.path;
  row.innerHTML = `
    <svg class="twisty" viewBox="0 0 24 24">${isDir ? '<path d="M9 6l6 6-6 6"/>' : ""}</svg>
    <svg class="ficon ${kind}" viewBox="0 0 24 24">${ICONS[kind]}</svg>
    <span class="rname"></span>
    <span class="rmeta">${isDir ? (node.children || []).length || "" : formatSize(node.size)}</span>`;
  row.querySelector(".rname").textContent = node.name;
  row.addEventListener("click", () => {
    if (isDir) {
      state.expanded.has(node.path) ? state.expanded.delete(node.path) : state.expanded.add(node.path);
      store.set("expanded", [...state.expanded]);
      renderTree();
    } else {
      openFile(node.path);
    }
  });
  parent.append(row);
  if (open) for (const child of node.children || []) renderNode(child, depth + 1, query, parent);
}

function revealInTree(rel) {
  const parts = rel.split("/");
  for (let i = 1; i < parts.length; i++) state.expanded.add(parts.slice(0, i).join("/"));
  store.set("expanded", [...state.expanded]);
  renderTree();
  const row = treeEl.querySelector(`.row[data-path="${CSS.escape(rel)}"]`);
  if (row) row.scrollIntoView({ block: "nearest" });
}

filterEl.addEventListener("input", renderTree);
$("#refresh").addEventListener("click", refreshTree);

/* ------------------------------------------------------------------ */
/* viewer                                                              */
/* ------------------------------------------------------------------ */

const viewerEl = $("#viewer");
const crumbsEl = $("#crumbs");
const actionsEl = $("#viewer-actions");
let currentFile = null;

const LANGS = { py: "python", js: "javascript", ts: "typescript", yaml: "yaml", yml: "yaml", toml: "ini",
  sh: "bash", json: "json", csv: "plaintext", tex: "latex", bib: "latex", html: "xml", xml: "xml", css: "css", r: "r", jl: "julia" };

async function openFile(rel, { keepScroll = false } = {}) {
  const scroll = viewerEl.scrollTop;
  history.replaceState(null, "", "#file=" + encodeURIComponent(rel));
  state.selected = rel;
  state.freshPaths.delete(rel);
  renderTree();
  try {
    const res = await fetch(`/api/file?path=${encodeURIComponent(rel)}`);
    if (!res.ok) throw new Error((await res.json()).detail || `HTTP ${res.status}`);
    currentFile = await res.json();
    renderViewer();
    if (keepScroll) viewerEl.scrollTop = scroll;
  } catch (err) {
    viewerEl.innerHTML = `<div class="viewer-empty">Could not open ${escapeHtml(rel)}: ${escapeHtml(err.message)}</div>`;
  }
}

function renderViewer() {
  const f = currentFile;
  const parts = f.path.split("/");
  crumbsEl.innerHTML = parts.map((p, i) =>
    i === parts.length - 1 ? `<span class="leaf">${escapeHtml(p)}</span>` : `${escapeHtml(p)}<span class="sep">/</span>`).join("");
  crumbsEl.title = f.path;

  const rawUrl = `/api/raw?path=${encodeURIComponent(f.path)}`;
  actionsEl.innerHTML = "";
  if (f.kind === "markdown") {
    const seg = document.createElement("div");
    seg.className = "seg";
    for (const mode of ["rendered", "source"]) {
      const b = document.createElement("button");
      b.textContent = mode === "rendered" ? "Preview" : "Source";
      b.className = state.viewerMode === mode ? "on" : "";
      b.addEventListener("click", () => { state.viewerMode = mode; renderViewer(); });
      seg.append(b);
    }
    actionsEl.append(seg);
  }
  if (f.content !== undefined) {
    actionsEl.append(iconButton("Copy", '<rect x="9" y="9" width="12" height="12" rx="2"/><path d="M5 15V5a2 2 0 0 1 2-2h10"/>', async (btn) => {
      try { await navigator.clipboard.writeText(f.content); btn.title = "Copied"; } catch { btn.title = "Copy failed"; }
    }));
  }
  actionsEl.append(iconButton("Open in new tab", '<path d="M14 4h6v6M20 4l-9 9M18 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h5"/>', () => window.open(rawUrl, "_blank")));

  viewerEl.innerHTML = "";
  if (f.truncated) viewerEl.insertAdjacentHTML("beforeend", `<div class="notice">Large file: showing the first 2 MB.</div>`);

  if (f.kind === "markdown" && state.viewerMode === "rendered") {
    const div = document.createElement("div");
    div.className = "md";
    div.innerHTML = renderMarkdown(f.content);
    rewriteRelativeLinks(div, f.path);
    linkCitations(div, f.path);
    highlightWithin(div);
    linkifyPaths(div);
    viewerEl.append(div);
  } else if (f.kind === "markdown" || f.kind === "json" || f.kind === "text") {
    let text = f.content;
    if (f.kind === "json") { try { text = JSON.stringify(JSON.parse(text), null, 2); } catch { /* show as-is */ } }
    const ext = f.name.split(".").pop().toLowerCase();
    const lang = f.kind === "markdown" ? "markdown" : LANGS[ext] || "plaintext";
    const pre = document.createElement("pre");
    pre.className = "code";
    const code = document.createElement("code");
    code.className = `language-${lang}`;
    code.textContent = text;
    pre.append(code);
    viewerEl.append(pre);
    if (window.hljs && text.length < 400_000) hljs.highlightElement(code);
  } else if (f.kind === "image") {
    viewerEl.innerHTML = `<img src="${rawUrl}" alt="${escapeHtml(f.name)}">`;
  } else if (f.kind === "pdf") {
    viewerEl.innerHTML = `<iframe src="${rawUrl}" title="${escapeHtml(f.name)}"></iframe>`;
  } else {
    viewerEl.innerHTML = `<div class="viewer-empty">No preview for this file type (${formatSize(f.size)}).<br><a class="file-link" href="${rawUrl}" target="_blank">Download</a></div>`;
  }
}

function iconButton(title, path, onClick) {
  const b = document.createElement("button");
  b.className = "ghost icon";
  b.title = title;
  b.innerHTML = `<svg viewBox="0 0 24 24">${path}</svg>`;
  b.addEventListener("click", () => onClick(b));
  return b;
}

/* optimistresearch cites documents as [doc_0014]; link those to the paper's notes. */
function linkCitations(root, filePath) {
  const parts = filePath.split("/");
  let corpus = null;
  for (let i = parts.length - 1; i > 0 && corpus === null; i--) {
    const dir = parts.slice(0, i).join("/");
    if ([...state.files.keys()].some((p) => p.startsWith(dir + "/sources/"))) corpus = dir;
  }
  if (corpus === null) return;
  const re = /doc_\d{3,}/g;
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
    acceptNode: (n) => (n.parentElement.closest("a, pre") ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT),
  });
  const targets = [];
  while (walker.nextNode()) if (/doc_\d{3,}/.test(walker.currentNode.nodeValue)) targets.push(walker.currentNode);
  for (const node of targets) {
    const frag = document.createDocumentFragment();
    let last = 0;
    node.nodeValue.replace(re, (id, offset) => {
      const notes = `${corpus}/sources/${id}/notes.md`;
      frag.append(node.nodeValue.slice(last, offset));
      if (state.files.has(notes)) {
        const a = document.createElement("a");
        a.className = "cite";
        a.textContent = id;
        a.title = `Open notes for ${id}`;
        a.addEventListener("click", (e) => { e.preventDefault(); openFile(notes); revealInTree(notes); });
        frag.append(a);
      } else {
        frag.append(id);
      }
      last = offset + id.length;
    });
    frag.append(node.nodeValue.slice(last));
    node.replaceWith(frag);
  }
}

/* Relative links inside a markdown file (e.g. subtopics/x.md) open in the viewer. */
function rewriteRelativeLinks(root, filePath) {
  const base = filePath.split("/").slice(0, -1);
  root.querySelectorAll("a[href]").forEach((a) => {
    const href = a.getAttribute("href");
    if (/^[a-z]+:|^#|^\//i.test(href)) { a.target = "_blank"; a.rel = "noopener"; return; }
    const parts = [...base];
    for (const seg of decodeURIComponent(href.split("#")[0]).split("/")) {
      if (seg === "..") parts.pop(); else if (seg && seg !== ".") parts.push(seg);
    }
    const rel = parts.join("/");
    a.addEventListener("click", (e) => { e.preventDefault(); openFile(rel); revealInTree(rel); });
  });
}

/* ------------------------------------------------------------------ */
/* resizable panes                                                     */
/* ------------------------------------------------------------------ */

function makeResizable(gutter, { axis, container, cssVar, key, min, max }) {
  const saved = store.get(key, null);
  if (saved) container.style.setProperty(cssVar, saved + "%");
  gutter.addEventListener("pointerdown", (e) => {
    e.preventDefault();
    gutter.setPointerCapture(e.pointerId);
    gutter.classList.add("dragging");
    document.body.classList.add("resizing");
    const rect = container.getBoundingClientRect();
    const move = (ev) => {
      const pct = axis === "x" ? ((ev.clientX - rect.left) / rect.width) * 100 : ((ev.clientY - rect.top) / rect.height) * 100;
      const clamped = Math.min(max, Math.max(min, pct));
      container.style.setProperty(cssVar, clamped + "%");
      store.set(key, clamped);
    };
    const up = () => {
      gutter.classList.remove("dragging");
      document.body.classList.remove("resizing");
      gutter.removeEventListener("pointermove", move);
      gutter.removeEventListener("pointerup", up);
    };
    gutter.addEventListener("pointermove", move);
    gutter.addEventListener("pointerup", up);
  });
}

makeResizable($("#gutter-v"), { axis: "x", container: $("#layout"), cssVar: "--left", key: "split-x", min: 28, max: 78 });
makeResizable($("#gutter-h"), { axis: "y", container: $("#side-pane"), cssVar: "--top", key: "split-y", min: 12, max: 85 });

/* ------------------------------------------------------------------ */
/* theme                                                               */
/* ------------------------------------------------------------------ */

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  $("#hljs-dark").disabled = theme !== "dark";
  $("#hljs-light").disabled = theme === "dark";
}
applyTheme(store.get("theme", matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark"));
$("#theme-toggle").addEventListener("click", () => {
  const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  store.set("theme", next);
  applyTheme(next);
});

/* ------------------------------------------------------------------ */
/* boot                                                                */
/* ------------------------------------------------------------------ */

(async function boot() {
  try {
    const info = await (await fetch("/api/info")).json();
    state.workspace = info.workspace;
    $("#model-pill").textContent = info.model;
  } catch { $("#model-pill").textContent = "offline"; }
  await refreshTree();
  const linked = new URLSearchParams(location.hash.slice(1)).get("file");
  if (linked) { openFile(linked); revealInTree(linked); }
  setThread(state.threadId);
  loadThreads();
  await loadHistory();
  updateComposer();
  inputEl.focus();
})();
