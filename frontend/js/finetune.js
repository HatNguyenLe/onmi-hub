import { api } from "./api.js";
import { store } from "./state.js";
import { $, confirmDialog, escapeHtml, formatBytes, formatDate, icon, toast } from "./ui.js";

const TASK_FOR_MODEL = {
  "text-generation": "causal-lm",
  "text2text-generation": "seq2seq",
  summarization: "seq2seq",
  translation: "seq2seq",
};
const COLUMN_FIELDS = {
  "causal-lm": [
    ["text", "Cột văn bản (text)", "Dùng 1 cột văn bản thuần, hoặc để trống và chọn cặp prompt/response"],
    ["prompt", "Cột prompt / instruction", ""],
    ["response", "Cột response / output", ""],
  ],
  seq2seq: [
    ["input", "Cột đầu vào", ""],
    ["target", "Cột đầu ra (target)", ""],
  ],
  classification: [
    ["text", "Cột văn bản", ""],
    ["label", "Cột nhãn (label)", ""],
  ],
};
const GUESS = {
  text: ["text", "content", "sentence", "document"],
  prompt: ["prompt", "instruction", "question", "input", "query"],
  response: ["response", "output", "answer", "completion"],
  input: ["input", "source", "document", "article", "text", "src"],
  target: ["target", "summary", "output", "highlights", "tgt", "label"],
  label: ["label", "labels", "category", "class", "sentiment"],
};
const STATUS = {
  queued: ["gray", "Đang chờ"],
  running: ["orange", "Đang huấn luyện"],
  cancelling: ["orange", "Đang huỷ"],
  completed: ["green", "Hoàn tất"],
  failed: ["red", "Thất bại"],
  cancelled: ["gray", "Đã huỷ"],
};

let dsSource = "upload";
let dataset = null; // {source, file_id | name/config/split, columns}
let pollTimer = null;
const openLogs = new Set();

export function initFinetune() {
  store.onModels(renderBaseSelect);

  $("#ftBase").addEventListener("change", () => {
    const m = store.models.find((x) => x.id === $("#ftBase").value);
    if (m) {
      $("#ftTask").value = TASK_FOR_MODEL[m.task] || "classification";
      if (!$("#ftOutput").value) $("#ftOutput").value = `${m.id.split("__").pop()}-ft`;
    }
    renderColumns();
  });
  $("#ftTask").addEventListener("change", renderColumns);

  $("#dsSource").addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b) return;
    dsSource = b.dataset.src;
    document.querySelectorAll("#dsSource button").forEach((x) => x.classList.toggle("active", x === b));
    $("#dsUpload").hidden = dsSource !== "upload";
    $("#dsHub").hidden = dsSource !== "hub";
    setDataset(null);
  });

  const dz = $("#dropzone");
  $("#dsFile").addEventListener("change", (e) => e.target.files[0] && upload(e.target.files[0]));
  dz.addEventListener("dragover", (e) => {
    e.preventDefault();
    dz.classList.add("drag");
  });
  dz.addEventListener("dragleave", () => dz.classList.remove("drag"));
  dz.addEventListener("drop", (e) => {
    e.preventDefault();
    dz.classList.remove("drag");
    if (e.dataTransfer.files[0]) upload(e.dataTransfer.files[0]);
  });
  $("#dsExisting").addEventListener("change", onExistingDataset);
  $("#dsHubPreviewBtn").addEventListener("click", previewHub);

  $("#hpLora").addEventListener("change", (e) => ($("#loraOpts").hidden = !e.target.checked));
  $("#ftForm").addEventListener("submit", submit);
  $("#jobList").addEventListener("click", onJobClick);
  $("#jobList").addEventListener("toggle", onLogToggle, true);

  renderColumns();
  loadDatasets();
  loadJobs();
}

export function showFinetune(modelId) {
  loadJobs();
  loadDatasets();
  if (modelId) {
    $("#ftBase").value = modelId;
    $("#ftOutput").value = "";
    $("#ftBase").dispatchEvent(new Event("change"));
  }
}

function renderBaseSelect() {
  const sel = $("#ftBase");
  const cur = sel.value;
  const ready = store.readyModels();
  sel.innerHTML = ready.length
    ? ready.map((m) => `<option value="${escapeHtml(m.id)}">${escapeHtml(m.repo_id)} — ${escapeHtml(store.taskLabel(m.task))}</option>`).join("")
    : `<option value="">Chưa có model — hãy tải từ Model Hub</option>`;
  if (ready.some((m) => m.id === cur)) sel.value = cur;
  else if (ready.length) sel.dispatchEvent(new Event("change"));
}

// ---------- dataset ----------
async function loadDatasets() {
  try {
    const list = await api.get("/api/datasets");
    const sel = $("#dsExisting");
    const cur = sel.value;
    sel.innerHTML = `<option value="">-- Chọn --</option>` + list
      .map((d) => `<option value="${escapeHtml(d.id)}">${escapeHtml(d.original_name)} (${d.num_rows} dòng, ${formatBytes(d.size)})</option>`)
      .join("");
    sel.value = cur;
    sel._data = list;
  } catch { /* non-critical */ }
}

function onExistingDataset() {
  const sel = $("#dsExisting");
  const d = (sel._data || []).find((x) => x.id === sel.value);
  setDataset(d ? { source: "upload", file_id: d.id, columns: d.columns, rows: d.rows, num_rows: d.num_rows, name: d.original_name } : null);
}

async function upload(file) {
  const fd = new FormData();
  fd.append("file", file);
  $("#dsPreview").innerHTML = `<div class="muted small"><span class="spinner"></span> Đang tải lên ${escapeHtml(file.name)}...</div>`;
  try {
    const d = await api.post("/api/datasets/upload", fd);
    setDataset({ source: "upload", file_id: d.id, columns: d.columns, rows: d.rows, num_rows: d.num_rows, name: d.original_name });
    await loadDatasets();
    $("#dsExisting").value = d.id;
    toast("Đã tải dataset lên", "success");
  } catch (e) {
    $("#dsPreview").innerHTML = "";
    toast(e.message, "error");
  }
  $("#dsFile").value = "";
}

async function previewHub() {
  const name = $("#dsHubName").value.trim();
  if (!name) return toast("Nhập tên dataset", "error");
  const body = { name, config: $("#dsHubConfig").value.trim() || null, split: $("#dsHubSplit").value.trim() || "train" };
  $("#dsPreview").innerHTML = `<div class="muted small"><span class="spinner"></span> Đang tải dữ liệu xem trước...</div>`;
  try {
    const d = await api.post("/api/datasets/hub-preview", body);
    setDataset({ source: "hub", ...body, columns: d.columns, rows: d.rows });
  } catch (e) {
    $("#dsPreview").innerHTML = "";
    toast(e.message, "error", 6000);
  }
}

function setDataset(d) {
  dataset = d;
  const box = $("#dsPreview");
  if (!d) {
    box.innerHTML = "";
  } else {
    const cols = d.columns;
    box.innerHTML = `
      <div class="muted small" style="margin-bottom:6px">${escapeHtml(d.name)}${d.num_rows != null ? ` · ${d.num_rows} dòng` : ""} · ${cols.length} cột</div>
      <div class="table-wrap"><table>
        <thead><tr>${cols.map((c) => `<th>${escapeHtml(c)}</th>`).join("")}</tr></thead>
        <tbody>${d.rows.map((r) => `<tr>${cols.map((c) => `<td title="${escapeHtml(r[c])}">${escapeHtml(r[c])}</td>`).join("")}</tr>`).join("")}</tbody>
      </table></div>`;
  }
  renderColumns();
}

function renderColumns() {
  const task = $("#ftTask").value;
  const cols = dataset?.columns || [];
  const used = new Set();
  const find = (key) => GUESS[key].find((g) => cols.includes(g) && !used.has(g)) || "";
  // Causal LM: prefer prompt/response pairs when present, otherwise a single text column.
  const pairMode = task === "causal-lm" && find("prompt") && find("response");
  $("#colMapping").innerHTML = COLUMN_FIELDS[task]
    .map(([key, label, help]) => {
      const skip = task === "causal-lm" && (pairMode ? key === "text" : key !== "text");
      const guess = skip ? "" : find(key);
      if (guess) used.add(guess);
      const control = cols.length
        ? `<select class="select" data-col="${key}"><option value="">-- Không dùng --</option>${cols
            .map((c) => `<option value="${escapeHtml(c)}" ${c === guess ? "selected" : ""}>${escapeHtml(c)}</option>`)
            .join("")}</select>`
        : `<input class="input" data-col="${key}" placeholder="Tên cột">`;
      return `<label class="field"><span>${label}</span>${control}${help ? `<small class="muted">${help}</small>` : ""}</label>`;
    })
    .join("");
}

// ---------- submit ----------
async function submit(e) {
  e.preventDefault();
  if (!dataset) return toast("Hãy chọn hoặc tải lên dataset", "error");
  const columns = {};
  document.querySelectorAll("#colMapping [data-col]").forEach((el) => {
    if (el.value.trim()) columns[el.dataset.col] = el.value.trim();
  });
  const num = (id) => Number($(id).value);
  const payload = {
    base_model: $("#ftBase").value,
    task: $("#ftTask").value,
    output_name: $("#ftOutput").value.trim(),
    dataset: dataset.source === "upload"
      ? { source: "upload", file_id: dataset.file_id }
      : { source: "hub", name: dataset.name, config: dataset.config, split: dataset.split },
    columns,
    params: {
      epochs: num("#hpEpochs"),
      learning_rate: num("#hpLr"),
      batch_size: num("#hpBatch"),
      grad_accum: num("#hpAccum"),
      max_length: num("#hpMaxLen"),
      warmup_ratio: num("#hpWarmup"),
      weight_decay: num("#hpWd"),
      eval_ratio: num("#hpEval"),
      max_samples: num("#hpMaxSamples"),
      use_lora: $("#hpLora").checked,
      lora_r: num("#hpLoraR"),
      lora_alpha: num("#hpLoraAlpha"),
      lora_dropout: num("#hpLoraDrop"),
    },
  };
  const btn = e.submitter;
  btn.disabled = true;
  try {
    await api.post("/api/finetune/jobs", payload);
    toast("Đã tạo job fine-tune", "success");
    $("#ftOutput").value = "";
    loadJobs();
  } catch (err) {
    toast(err.message, "error", 6000);
  } finally {
    btn.disabled = false;
  }
}

// ---------- jobs ----------
async function loadJobs() {
  clearTimeout(pollTimer);
  let jobs = [];
  try {
    jobs = await api.get("/api/finetune/jobs");
  } catch {
    return;
  }
  renderJobs(jobs);
  const active = jobs.some((j) => ["queued", "running", "cancelling"].includes(j.status));
  if (active) pollTimer = setTimeout(loadJobs, 2500);
  if (jobs.some((j) => j.status === "completed" && !store.models.some((m) => m.id === j.output_model))) {
    store.refreshModels();
  }
}

function renderJobs(jobs) {
  const box = $("#jobList");
  if (!jobs.length) {
    box.innerHTML = `<div class="empty">${icon("tune")}<div>Chưa có job nào</div></div>`;
    return;
  }
  box.innerHTML = jobs
    .map((j) => {
      const [color, label] = STATUS[j.status] || ["gray", j.status];
      const pct = Math.round((j.progress || 0) * 100);
      const metrics = j.logs.filter((l) => l.type === "metric");
      const evalLoss = [...metrics].reverse().find((m) => m.eval_loss != null)?.eval_loss;
      const logText = j.logs
        .map((l) => (l.type === "metric" ? `[step ${l.step}] ` + Object.entries(l).filter(([k]) => !["type", "step"].includes(k)).map(([k, v]) => `${k}=${typeof v === "number" ? +v.toPrecision(4) : v}`).join(" ") : l.message))
        .join("\n");
      const active = ["queued", "running"].includes(j.status);
      return `
      <div class="card job" data-id="${j.id}">
        <div class="job-head">
          <div class="title">${escapeHtml(j.output_model)}</div>
          <span class="badge ${color}">${label}</span>
          ${active ? `<button class="btn btn-sm" data-act="cancel" type="button">Huỷ</button>` : ""}
          ${j.status === "completed" ? `<button class="btn btn-sm btn-primary" data-act="chat" data-model="${escapeHtml(j.output_model)}" type="button">${icon("chat")}Chat</button>` : ""}
          ${!active && j.status !== "cancelling" ? `<button class="icon-btn sm" data-act="delete" type="button" aria-label="Xoá">${icon("trash")}</button>` : ""}
        </div>
        <div class="muted small">${escapeHtml(j.base_model)} · ${escapeHtml(j.task)} · ${escapeHtml(formatDate(j.created_at))}${j.params.use_lora ? " · LoRA" : ""}</div>
        <div class="progress"><div style="width:${pct}%"></div></div>
        <div class="job-stats">
          <div>Tiến độ<b>${pct}%</b></div>
          <div>Epoch<b>${j.epoch != null ? j.epoch.toFixed(2) : "-"}</b></div>
          <div>Loss<b>${j.loss != null ? j.loss.toFixed(4) : "-"}${evalLoss != null ? ` / ${evalLoss.toFixed(3)}` : ""}</b></div>
        </div>
        ${metrics.length > 1 ? lossChart(metrics) : ""}
        ${j.error ? `<div class="small" style="color:var(--danger)">${escapeHtml(j.error)}</div>` : ""}
        <details ${openLogs.has(j.id) ? "open" : ""} data-log="${j.id}"><summary>Nhật ký</summary><pre class="job-logs">${escapeHtml(logText || "...")}</pre></details>
      </div>`;
    })
    .join("");
}

function lossChart(metrics) {
  const train = metrics.filter((m) => m.loss != null).map((m) => [m.step, m.loss]);
  const ev = metrics.filter((m) => m.eval_loss != null).map((m) => [m.step, m.eval_loss]);
  const all = [...train, ...ev];
  if (all.length < 2) return "";
  const W = 400, H = 120, P = 22;
  const xs = all.map((p) => p[0]), ys = all.map((p) => p[1]);
  const [x0, x1] = [Math.min(...xs), Math.max(...xs) || 1];
  const [y0, y1] = [Math.min(...ys), Math.max(...ys)];
  const sx = (x) => P + ((x - x0) / (x1 - x0 || 1)) * (W - P - 6);
  const sy = (y) => 6 + (1 - (y - y0) / (y1 - y0 || 1)) * (H - P - 6);
  const path = (pts) => pts.map((p, i) => `${i ? "L" : "M"}${sx(p[0]).toFixed(1)},${sy(p[1]).toFixed(1)}`).join("");
  const area = train.length > 1 ? `${path(train)}L${sx(train.at(-1)[0])},${H - P}L${sx(train[0][0])},${H - P}Z` : "";
  return `
    <svg class="loss-chart" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">
      <line class="grid-line" x1="${P}" y1="${H - P}" x2="${W}" y2="${H - P}"/>
      <line class="grid-line" x1="${P}" y1="6" x2="${W}" y2="6"/>
      <text x="0" y="12">${y1.toFixed(2)}</text>
      <text x="0" y="${H - P}">${y0.toFixed(2)}</text>
      <text x="${W - 60}" y="${H - 6}">step ${x1}</text>
      ${area ? `<path class="area" d="${area}"/>` : ""}
      ${train.length > 1 ? `<path class="line" d="${path(train)}"/>` : ""}
      ${ev.length > 1 ? `<path class="eval" d="${path(ev)}"/>` : ""}
    </svg>`;
}

function onLogToggle(e) {
  const d = e.target.closest?.("details[data-log]");
  if (!d) return;
  const id = Number(d.dataset.log);
  if (d.open) openLogs.add(id);
  else openLogs.delete(id);
}

let chatHandler = null;
export function setFinetuneChatHandler(fn) {
  chatHandler = fn;
}

async function onJobClick(e) {
  const btn = e.target.closest("[data-act]");
  if (!btn) return;
  const id = btn.closest(".job").dataset.id;
  try {
    if (btn.dataset.act === "cancel") {
      await api.post(`/api/finetune/jobs/${id}/cancel`);
      loadJobs();
    } else if (btn.dataset.act === "delete") {
      if (!(await confirmDialog("Xoá job", "Xoá job khỏi danh sách? (Model đã tạo vẫn được giữ lại)"))) return;
      await api.del(`/api/finetune/jobs/${id}`);
      loadJobs();
    } else if (btn.dataset.act === "chat") {
      await store.refreshModels();
      chatHandler?.(btn.dataset.model);
    }
  } catch (err) {
    toast(err.message, "error");
  }
}
