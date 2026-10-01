export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

export function escapeHtml(str) {
  return String(str ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

export const icon = (name) => `<svg><use href="#i-${name}"/></svg>`;

export function formatNumber(n) {
  if (n == null) return "-";
  if (n >= 1e9) return (n / 1e9).toFixed(1) + "B";
  if (n >= 1e6) return (n / 1e6).toFixed(1) + "M";
  if (n >= 1e3) return (n / 1e3).toFixed(1) + "K";
  return String(n);
}

export function formatBytes(b) {
  if (!b) return "0 B";
  const u = ["B", "KB", "MB", "GB", "TB"];
  const i = Math.min(Math.floor(Math.log(b) / Math.log(1024)), u.length - 1);
  return (b / 1024 ** i).toFixed(i ? 1 : 0) + " " + u[i];
}

export function formatDate(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  return d.toLocaleString("vi-VN", { dateStyle: "short", timeStyle: "short" });
}

export function relativeGroup(iso) {
  const d = new Date(iso);
  const start = new Date();
  start.setHours(0, 0, 0, 0);
  const diff = (start - d) / 86400000;
  if (diff <= 0) return "Hôm nay";
  if (diff <= 1) return "Hôm qua";
  if (diff <= 7) return "7 ngày trước";
  if (diff <= 30) return "30 ngày trước";
  return "Cũ hơn";
}

// ---------- toast ----------
export function toast(message, type = "info", timeout = 3500) {
  const el = document.createElement("div");
  el.className = `toast ${type}`;
  el.textContent = message;
  $("#toasts").appendChild(el);
  setTimeout(() => el.remove(), timeout);
}

// ---------- modal ----------
const modal = () => $("#modal");

export function openModal({ title, body, actions = [] }) {
  $("#modalTitle").textContent = title;
  const bodyEl = $("#modalBody");
  bodyEl.innerHTML = "";
  if (typeof body === "string") bodyEl.innerHTML = body;
  else if (body) bodyEl.appendChild(body);
  const foot = $("#modalFoot");
  foot.innerHTML = "";
  for (const a of actions) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = `btn ${a.className || ""}`;
    btn.innerHTML = (a.icon ? icon(a.icon) : "") + escapeHtml(a.label);
    btn.addEventListener("click", async () => {
      if (a.onClick) {
        const keep = await a.onClick(btn);
        if (keep === true) return;
      }
      closeModal();
    });
    foot.appendChild(btn);
  }
  modal().hidden = false;
}

export function closeModal() {
  modal().hidden = true;
}

export function confirmDialog(title, message, { danger = true, okLabel = "Xác nhận" } = {}) {
  return new Promise((resolve) => {
    openModal({
      title,
      body: `<p>${escapeHtml(message)}</p>`,
      actions: [
        { label: "Huỷ", onClick: () => resolve(false) },
        { label: okLabel, className: danger ? "btn-danger" : "btn-primary", onClick: () => resolve(true) },
      ],
    });
    const once = () => resolve(false);
    $("#modal").addEventListener("modal-dismiss", once, { once: true });
  });
}

export function initModal() {
  const m = modal();
  m.addEventListener("click", (e) => {
    if (e.target === m || e.target.closest("[data-close]")) {
      m.dispatchEvent(new Event("modal-dismiss"));
      closeModal();
    }
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !m.hidden) {
      m.dispatchEvent(new Event("modal-dismiss"));
      closeModal();
    }
  });
}

// ---------- theme ----------
export function getThemePref() {
  return localStorage.getItem("theme") || "system";
}

export function applyTheme(pref) {
  if (pref === "system") localStorage.removeItem("theme");
  else localStorage.setItem("theme", pref);
  const resolved = pref === "system" ? (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light") : pref;
  document.documentElement.dataset.theme = resolved;
  document.querySelector('meta[name="theme-color"]').content = resolved === "dark" ? "#0f1117" : "#ffffff";
  $$(".theme-label").forEach((el) => (el.textContent = resolved === "dark" ? "Chế độ sáng" : "Chế độ tối"));
  $$("[data-theme-opt]").forEach((b) => b.classList.toggle("active", b.dataset.themeOpt === pref));
}

export function toggleTheme() {
  applyTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark");
}

// ---------- markdown (safe subset) ----------
function inline(text) {
  return text
    .replace(/`([^`\n]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*([^*\n]+)\*(?!\*)/g, "$1<em>$2</em>");
}

function renderBlock(text) {
  const lines = escapeHtml(text).split("\n");
  let html = "";
  let list = null;
  let para = [];
  const flushPara = () => {
    if (para.length) html += `<p>${inline(para.join("<br>"))}</p>`;
    para = [];
  };
  const flushList = () => {
    if (list) html += `<${list.type}>${list.items.map((i) => `<li>${inline(i)}</li>`).join("")}</${list.type}>`;
    list = null;
  };
  for (const line of lines) {
    const ul = line.match(/^\s*[-*•]\s+(.*)$/);
    const ol = line.match(/^\s*\d+[.)]\s+(.*)$/);
    const h = line.match(/^(#{1,4})\s+(.*)$/);
    if (ul || ol) {
      flushPara();
      const type = ul ? "ul" : "ol";
      if (list && list.type !== type) flushList();
      list = list || { type, items: [] };
      list.items.push((ul || ol)[1]);
    } else if (h) {
      flushPara();
      flushList();
      html += `<p><strong>${inline(h[2])}</strong></p>`;
    } else if (!line.trim()) {
      flushPara();
      flushList();
    } else {
      flushList();
      para.push(line);
    }
  }
  flushPara();
  flushList();
  return html;
}

export function renderMarkdown(src) {
  const parts = String(src ?? "").split(/```/);
  return parts
    .map((part, i) => {
      if (i % 2 === 0) return renderBlock(part);
      const nl = part.indexOf("\n");
      const code = nl >= 0 ? part.slice(nl + 1) : part;
      return `<pre><button class="icon-btn sm copy-code" type="button" aria-label="Sao chép">${icon("copy")}</button><code>${escapeHtml(code.replace(/\n$/, ""))}</code></pre>`;
    })
    .join("");
}

export async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    toast("Đã sao chép", "success", 1500);
  } catch {
    toast("Không thể sao chép", "error");
  }
}
