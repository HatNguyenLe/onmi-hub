import { api } from "./api.js";
import { store } from "./state.js";
import { $, confirmDialog, escapeHtml, formatBytes, formatDate, icon, toast } from "./ui.js";

let filter = "all";
let pollTimer = null;
let handlers = {};

export function initModels({ onChat, onFinetune }) {
  handlers = { onChat, onFinetune };
  $("#modelFilter").addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b) return;
    filter = b.dataset.filter;
    document.querySelectorAll("#modelFilter button").forEach((x) => x.classList.toggle("active", x === b));
    render();
  });
  $("#localModels").addEventListener("click", onClick);
  store.onModels(() => {
    render();
    schedulePoll();
  });
}

function schedulePoll() {
  clearTimeout(pollTimer);
  if (store.models.some((m) => m.status === "downloading")) {
    pollTimer = setTimeout(() => store.refreshModels(), 2000);
  }
}

const STATUS = {
  ready: ["green", "Sẵn sàng"],
  downloading: ["orange", "Đang tải"],
  error: ["red", "Lỗi"],
};

function render() {
  const box = $("#localModels");
  const models = store.models.filter((m) => filter === "all" || m.source === filter);
  if (!models.length) {
    box.innerHTML = `<div class="empty">${icon("box")}<div>Chưa có model nào. Hãy vào <a href="#/hub">Model Hub</a> để tải về.</div></div>`;
    return;
  }
  box.innerHTML = models
    .map((m) => {
      const [color, label] = STATUS[m.status] || ["gray", m.status];
      const pct = Math.round((m.progress || 0) * 100);
      return `
      <div class="card model-card" data-id="${escapeHtml(m.id)}">
        <div class="title">${escapeHtml(m.repo_id)}</div>
        <div class="badges">
          <span class="badge ${color}">${label}</span>
          <span class="badge">${escapeHtml(store.taskLabel(m.task))}</span>
          <span class="badge gray">${m.source === "finetuned" ? "Fine-tuned" : "Hugging Face"}</span>
        </div>
        ${m.status === "downloading" ? `<div class="progress"><div style="width:${pct}%"></div></div><div class="muted small">${pct}% · ${formatBytes(m.size_bytes)} / ${formatBytes(m.total_bytes)}</div>` : ""}
        ${m.status === "error" ? `<div class="small" style="color:var(--danger)">${escapeHtml(m.error || "")}</div>` : ""}
        <div class="meta">
          <span>${formatBytes(m.size_bytes)}</span>
          ${m.architecture ? `<span>${escapeHtml(m.architecture)}</span>` : ""}
          ${m.base_model ? `<span>Gốc: ${escapeHtml(m.base_model)}</span>` : ""}
          <span>${escapeHtml(formatDate(m.created_at))}</span>
        </div>
        <div class="actions">
          ${m.status === "ready" ? `<button class="btn btn-sm btn-primary" data-act="chat" type="button">${icon("chat")}Chat</button>
          <button class="btn btn-sm" data-act="tune" type="button">${icon("tune")}Fine-tune</button>` : ""}
          ${m.status === "error" && m.source === "hub" ? `<button class="btn btn-sm" data-act="retry" type="button">${icon("refresh")}Thử lại</button>` : ""}
          ${m.status !== "downloading" ? `<button class="icon-btn" data-act="delete" type="button" aria-label="Xoá">${icon("trash")}</button>` : ""}
        </div>
      </div>`;
    })
    .join("");
}

async function onClick(e) {
  const btn = e.target.closest("[data-act]");
  if (!btn) return;
  const id = btn.closest(".model-card").dataset.id;
  const model = store.models.find((m) => m.id === id);
  switch (btn.dataset.act) {
    case "chat":
      handlers.onChat(id);
      break;
    case "tune":
      handlers.onFinetune(id);
      break;
    case "retry":
      try {
        await api.post("/api/models/download", { repo_id: model.repo_id });
      } catch (err) {
        toast(err.message, "error");
      }
      store.refreshModels();
      break;
    case "delete":
      if (!(await confirmDialog("Xoá model", `Xoá "${model.repo_id}" khỏi ổ đĩa?`))) return;
      try {
        await api.del(`/api/models/${encodeURIComponent(id)}`);
        toast("Đã xoá model", "success");
      } catch (err) {
        toast(err.message, "error");
      }
      store.refreshModels();
      break;
  }
}
