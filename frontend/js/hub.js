import { api } from "./api.js";
import { store } from "./state.js";
import { $, escapeHtml, formatBytes, formatDate, formatNumber, icon, openModal, toast } from "./ui.js";

let loaded = false;

export function initHub() {
  const taskSel = $("#hubTask");
  taskSel.innerHTML += Object.entries(store.tasks)
    .map(([k, v]) => `<option value="${escapeHtml(k)}">${escapeHtml(v)}</option>`)
    .join("");
  taskSel.value = "text-generation";

  $("#hubForm").addEventListener("submit", (e) => {
    e.preventDefault();
    search();
  });
  taskSel.addEventListener("change", search);
  $("#hubSort").addEventListener("change", search);
  $("#hubResults").addEventListener("click", onClick);
  store.onModels(() => loaded && markDownloaded());
}

export function showHub() {
  if (!loaded) search();
}

async function search() {
  loaded = true;
  const box = $("#hubResults");
  box.innerHTML = Array.from({ length: 6 }, () => `<div class="skeleton"></div>`).join("");
  const params = new URLSearchParams({ sort: $("#hubSort").value, limit: "30" });
  const q = $("#hubQuery").value.trim();
  if (q) params.set("q", q);
  if ($("#hubTask").value) params.set("task", $("#hubTask").value);
  try {
    const results = await api.get(`/api/hub/models?${params}`);
    if (!results.length) {
      box.innerHTML = `<div class="empty">${icon("hub")}<div>Không tìm thấy model phù hợp</div></div>`;
      return;
    }
    box.innerHTML = results.map(card).join("");
    markDownloaded();
  } catch (e) {
    box.innerHTML = `<div class="empty">${escapeHtml(e.message)}</div>`;
  }
}

function card(m) {
  return `
    <div class="card model-card" data-repo="${escapeHtml(m.id)}">
      <div class="title">${escapeHtml(m.id)}</div>
      <div class="badges">
        <span class="badge">${escapeHtml(store.taskLabel(m.task))}</span>
        ${m.gated ? `<span class="badge orange">Gated</span>` : ""}
        <span class="badge green" data-local hidden>Đã tải</span>
      </div>
      <div class="meta">
        <span>${icon("download")}${formatNumber(m.downloads)}</span>
        <span>${icon("heart")}${formatNumber(m.likes)}</span>
        ${m.last_modified ? `<span>${escapeHtml(formatDate(m.last_modified))}</span>` : ""}
      </div>
      <div class="actions">
        <button class="btn btn-sm" data-act="info" type="button">Chi tiết</button>
        <button class="btn btn-sm btn-primary" data-act="download" type="button">${icon("download")}Tải về</button>
      </div>
    </div>`;
}

function markDownloaded() {
  const local = new Map(store.models.map((m) => [m.repo_id, m.status]));
  document.querySelectorAll("#hubResults .model-card").forEach((el) => {
    const status = local.get(el.dataset.repo);
    el.querySelector("[data-local]").hidden = status !== "ready";
    const btn = el.querySelector('[data-act="download"]');
    btn.disabled = status === "ready" || status === "downloading";
    btn.innerHTML = status === "downloading" ? `<span class="spinner"></span>Đang tải` : status === "ready" ? "Đã có" : `${icon("download")}Tải về`;
  });
}

async function onClick(e) {
  const btn = e.target.closest("[data-act]");
  if (!btn) return;
  const repo = btn.closest(".model-card").dataset.repo;
  if (btn.dataset.act === "download") download(repo, btn);
  else showDetails(repo);
}

export async function download(repo, btn) {
  if (btn) {
    btn.disabled = true;
    btn.innerHTML = `<span class="spinner"></span>Đang chuẩn bị`;
  }
  try {
    await api.post("/api/models/download", { repo_id: repo });
    toast(`Bắt đầu tải ${repo}`, "success");
  } catch (e) {
    toast(e.message, "error", 6000);
  }
  await store.refreshModels();
}

async function showDetails(repo) {
  openModal({ title: repo, body: `<div style="text-align:center;padding:30px"><span class="spinner"></span></div>` });
  try {
    const d = await api.get(`/api/hub/models/${repo.split("/").map(encodeURIComponent).join("/")}`);
    const langs = Array.isArray(d.languages) ? d.languages.join(", ") : d.languages;
    const files = d.files
      .slice(0, 30)
      .map((f) => `<tr><td>${escapeHtml(f.name)}</td><td>${formatBytes(f.size)}</td></tr>`)
      .join("");
    openModal({
      title: repo,
      body: `
        <dl class="kv">
          <dt>Tác vụ</dt><dd>${escapeHtml(store.taskLabel(d.task))}</dd>
          <dt>Thư viện</dt><dd>${escapeHtml(d.library || "-")}</dd>
          <dt>Tham số</dt><dd>${d.parameters ? formatNumber(d.parameters) : "-"}</dd>
          <dt>Dung lượng tải</dt><dd>${formatBytes(d.download_size)}</dd>
          <dt>License</dt><dd>${escapeHtml(d.license || "-")}</dd>
          <dt>Ngôn ngữ</dt><dd>${escapeHtml(langs || "-")}</dd>
          <dt>Lượt tải</dt><dd>${formatNumber(d.downloads)}</dd>
        </dl>
        <p class="badges" style="margin:14px 0">${d.tags.map((t) => `<span class="badge gray">${escapeHtml(t)}</span>`).join("")}</p>
        <details><summary>Danh sách file (${d.files.length})</summary>
          <div class="table-wrap" style="margin-top:8px"><table><tbody>${files}</tbody></table></div>
        </details>
        <p><a href="https://huggingface.co/${repo.split("/").map(encodeURIComponent).join("/")}" target="_blank" rel="noopener noreferrer">Xem trên Hugging Face ↗</a></p>`,
      actions: [
        { label: "Đóng" },
        { label: "Tải về", className: "btn-primary", icon: "download", onClick: () => download(repo) },
      ],
    });
  } catch (e) {
    openModal({ title: repo, body: `<p>${escapeHtml(e.message)}</p>`, actions: [{ label: "Đóng" }] });
  }
}
