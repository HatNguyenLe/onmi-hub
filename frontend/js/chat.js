import { api, streamSSE } from "./api.js";
import { store } from "./state.js";
import {
  $, confirmDialog, copyText, escapeHtml, formatDate, icon, openModal, relativeGroup, renderMarkdown, toast,
} from "./ui.js";

const PARAMS_KEY = "chatParams";
const TASK_HINTS = {
  "question-answering": "Định dạng: câu hỏi, xuống dòng ---, rồi ngữ cảnh.",
  "zero-shot-classification": "Định dạng: văn bản, xuống dòng ---, rồi các nhãn cách nhau bởi dấu phẩy.",
  "fill-mask": "Chèn [MASK] vào vị trí cần điền từ.",
  "sentence-similarity": "Nhập 2 câu cách nhau bởi dòng --- để tính độ tương đồng.",
  "text-classification": "Nhập văn bản cần phân loại.",
  "token-classification": "Nhập văn bản để nhận dạng thực thể.",
  summarization: "Nhập văn bản cần tóm tắt.",
  translation: "Nhập văn bản cần dịch.",
};
const SUGGESTIONS = [
  "Giải thích fine-tuning là gì một cách đơn giản",
  "Viết một đoạn văn ngắn về Hà Nội mùa thu",
  "So sánh LoRA và full fine-tuning",
  "Viết hàm Python đảo ngược chuỗi",
];

const state = {
  conversations: [],
  currentId: null,
  messages: [],
  controller: null,
  searchTimer: null,
};

const els = {};

export function initChat() {
  Object.assign(els, {
    view: $("#view-chat"),
    list: $("#historyList"),
    search: $("#historySearch"),
    messages: $("#messages"),
    input: $("#chatInput"),
    send: $("#sendBtn"),
    form: $("#composer"),
    modelSelect: $("#chatModelSelect"),
    taskBadge: $("#chatTaskBadge"),
    hint: $("#composerHint"),
    system: $("#systemPrompt"),
    params: $("#paramsPanel"),
  });

  loadParams();
  store.onModels(renderModelSelect);

  $("#newChatBtn").addEventListener("click", () => {
    newChat();
    closeHistory();
  });
  $("#historyToggle").addEventListener("click", () => els.view.classList.toggle("history-open"));
  $("#historyBackdrop").addEventListener("click", closeHistory);
  $("#paramsToggle").addEventListener("click", () => (els.params.hidden = !els.params.hidden));
  $("#clearHistoryBtn").addEventListener("click", clearAll);

  els.search.addEventListener("input", () => {
    clearTimeout(state.searchTimer);
    state.searchTimer = setTimeout(loadConversations, 250);
  });

  els.modelSelect.addEventListener("change", async () => {
    localStorage.setItem("chatModel", els.modelSelect.value);
    updateModelInfo();
    if (state.currentId) {
      await api.patch(`/api/conversations/${state.currentId}`, { model_id: els.modelSelect.value }).catch(() => {});
    }
  });

  els.system.addEventListener("change", async () => {
    if (state.currentId) {
      await api.patch(`/api/conversations/${state.currentId}`, { system_prompt: els.system.value }).catch(() => {});
    }
  });

  for (const [id, out] of [["pTemp", "tempVal"], ["pTopP", "topPVal"], ["pMaxTok", "maxTokVal"], ["pRep", "repVal"]]) {
    $("#" + id).addEventListener("input", (e) => {
      $("#" + out).textContent = e.target.value;
      saveParams();
    });
  }

  els.input.addEventListener("input", autosize);
  els.input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      els.form.requestSubmit();
    }
  });
  els.form.addEventListener("submit", (e) => {
    e.preventDefault();
    if (state.controller) stopStreaming();
    else sendMessage(els.input.value);
  });

  els.list.addEventListener("click", onHistoryClick);
  els.messages.addEventListener("click", onMessagesClick);

  renderWelcome();
  loadConversations();
}

export function openChatWithModel(modelId) {
  newChat();
  if ([...els.modelSelect.options].some((o) => o.value === modelId)) {
    els.modelSelect.value = modelId;
    localStorage.setItem("chatModel", modelId);
    updateModelInfo();
  }
}

function closeHistory() {
  els.view.classList.remove("history-open");
}

// ---------- params ----------
function loadParams() {
  try {
    const p = JSON.parse(localStorage.getItem(PARAMS_KEY) || "{}");
    const map = { temperature: ["pTemp", "tempVal"], top_p: ["pTopP", "topPVal"], max_new_tokens: ["pMaxTok", "maxTokVal"], repetition_penalty: ["pRep", "repVal"] };
    for (const [k, [inp, out]] of Object.entries(map)) {
      if (p[k] != null) {
        $("#" + inp).value = p[k];
        $("#" + out).textContent = p[k];
      }
    }
  } catch { /* ignore corrupt local settings */ }
}

function getParams() {
  return {
    temperature: parseFloat($("#pTemp").value),
    top_p: parseFloat($("#pTopP").value),
    max_new_tokens: parseInt($("#pMaxTok").value, 10),
    repetition_penalty: parseFloat($("#pRep").value),
  };
}

function saveParams() {
  localStorage.setItem(PARAMS_KEY, JSON.stringify(getParams()));
}

// ---------- model select ----------
function renderModelSelect() {
  const ready = store.readyModels();
  const current = els.modelSelect.value || localStorage.getItem("chatModel");
  if (!ready.length) {
    els.modelSelect.innerHTML = `<option value="">Chưa có model — hãy tải từ Model Hub</option>`;
  } else {
    const groups = {};
    for (const m of ready) (groups[m.task] ||= []).push(m);
    els.modelSelect.innerHTML = Object.entries(groups)
      .map(([task, ms]) => `<optgroup label="${escapeHtml(store.taskLabel(task))}">${ms
        .map((m) => `<option value="${escapeHtml(m.id)}">${escapeHtml(m.repo_id)}${m.source === "finetuned" ? " ★" : ""}</option>`)
        .join("")}</optgroup>`)
      .join("");
    if (current && ready.some((m) => m.id === current)) els.modelSelect.value = current;
  }
  updateModelInfo();
}

function currentModel() {
  return store.models.find((m) => m.id === els.modelSelect.value);
}

function updateModelInfo() {
  const m = currentModel();
  els.taskBadge.textContent = m ? store.taskLabel(m.task) : "";
  els.hint.textContent = m ? TASK_HINTS[m.task] || "Lịch sử trò chuyện được lưu tự động." : "Chọn một model để bắt đầu.";
}

// ---------- history ----------
async function loadConversations() {
  const q = els.search.value.trim();
  try {
    state.conversations = await api.get("/api/conversations" + (q ? `?q=${encodeURIComponent(q)}` : ""));
  } catch (e) {
    toast(e.message, "error");
    return;
  }
  renderHistory();
}

function renderHistory() {
  if (!state.conversations.length) {
    els.list.innerHTML = `<div class="muted small" style="padding:20px 10px;text-align:center">${els.search.value ? "Không tìm thấy kết quả" : "Chưa có cuộc trò chuyện nào"}</div>`;
    return;
  }
  let html = "";
  let group = null;
  for (const c of state.conversations) {
    const g = relativeGroup(c.updated_at);
    if (g !== group) {
      html += `<div class="history-group">${g}</div>`;
      group = g;
    }
    const model = store.models.find((m) => m.id === c.model_id);
    html += `
      <div class="conv-item ${c.id === state.currentId ? "active" : ""}" data-id="${c.id}">
        <div class="conv-text">
          <div class="conv-title">${escapeHtml(c.title)}</div>
          <div class="conv-sub">${escapeHtml(model?.repo_id || c.model_id || "—")} · ${c.message_count} tin</div>
        </div>
        <div class="conv-actions">
          <button class="icon-btn sm" data-act="rename" type="button" aria-label="Đổi tên">${icon("edit")}</button>
          <button class="icon-btn sm" data-act="delete" type="button" aria-label="Xoá">${icon("trash")}</button>
        </div>
      </div>`;
  }
  els.list.innerHTML = html;
}

async function onHistoryClick(e) {
  const item = e.target.closest(".conv-item");
  if (!item) return;
  const id = Number(item.dataset.id);
  const act = e.target.closest("[data-act]")?.dataset.act;
  if (act === "delete") {
    if (!(await confirmDialog("Xoá cuộc trò chuyện", "Bạn có chắc muốn xoá cuộc trò chuyện này?"))) return;
    await api.del(`/api/conversations/${id}`);
    if (state.currentId === id) newChat();
    loadConversations();
  } else if (act === "rename") {
    renameConversation(id);
  } else {
    selectConversation(id);
    closeHistory();
  }
}

function renameConversation(id) {
  const conv = state.conversations.find((c) => c.id === id);
  const input = document.createElement("input");
  input.className = "input";
  input.maxLength = 200;
  input.value = conv?.title || "";
  openModal({
    title: "Đổi tên cuộc trò chuyện",
    body: input,
    actions: [
      { label: "Huỷ" },
      {
        label: "Lưu",
        className: "btn-primary",
        onClick: async () => {
          const title = input.value.trim();
          if (!title) return true;
          await api.patch(`/api/conversations/${id}`, { title });
          loadConversations();
        },
      },
    ],
  });
  setTimeout(() => input.focus(), 50);
}

async function clearAll() {
  if (!(await confirmDialog("Xoá toàn bộ lịch sử", "Tất cả cuộc trò chuyện sẽ bị xoá vĩnh viễn."))) return;
  await api.del("/api/conversations");
  newChat();
  loadConversations();
  toast("Đã xoá lịch sử", "success");
}

async function selectConversation(id) {
  if (state.controller) stopStreaming();
  try {
    const conv = await api.get(`/api/conversations/${id}`);
    state.currentId = conv.id;
    state.messages = conv.messages;
    els.system.value = conv.system_prompt || "";
    if (conv.model_id && [...els.modelSelect.options].some((o) => o.value === conv.model_id)) {
      els.modelSelect.value = conv.model_id;
    } else if (conv.model_id) {
      toast("Model của cuộc trò chuyện này không còn tồn tại, hãy chọn model khác.", "error");
    }
    updateModelInfo();
    renderMessages();
    renderHistory();
  } catch (e) {
    toast(e.message, "error");
  }
}

function newChat() {
  if (state.controller) stopStreaming();
  state.currentId = null;
  state.messages = [];
  renderWelcome();
  renderHistory();
  els.input.focus();
}

// ---------- messages ----------
function renderWelcome() {
  els.messages.innerHTML = `
    <div class="welcome">
      <div class="brand-logo">${icon("spark")}</div>
      <h2>Xin chào! Hôm nay bạn muốn làm gì?</h2>
      <p class="muted">Chọn model ở trên và bắt đầu trò chuyện. Mọi cuộc hội thoại đều được lưu lại.</p>
      <div class="suggestions">
        ${SUGGESTIONS.map((s) => `<div class="card suggestion" data-suggest>${escapeHtml(s)}</div>`).join("")}
      </div>
    </div>`;
}

function messageHtml(m, index) {
  const isUser = m.role === "user";
  const body = isUser ? escapeHtml(m.content) : renderMarkdown(m.content);
  const isLast = index === state.messages.length - 1;
  return `
    <div class="msg ${m.role} ${m.error ? "error" : ""}" data-index="${index}">
      <div class="avatar">${isUser ? "B" : icon("spark")}</div>
      <div class="bubble-col">
        <div class="bubble">${body}</div>
        <div class="msg-meta">
          <span>${escapeHtml(formatDate(m.created_at))}</span>
          <button class="icon-btn" data-act="copy" type="button" aria-label="Sao chép">${icon("copy")}</button>
          ${!isUser && isLast && !m.error ? `<button class="icon-btn" data-act="regen" type="button" aria-label="Tạo lại">${icon("refresh")}</button>` : ""}
        </div>
      </div>
    </div>`;
}

function renderMessages() {
  if (!state.messages.length) return renderWelcome();
  els.messages.innerHTML = `<div class="msg-wrap">${state.messages.map(messageHtml).join("")}</div>`;
  scrollBottom(true);
}

function scrollBottom(force = false) {
  const el = els.messages;
  if (force || el.scrollHeight - el.scrollTop - el.clientHeight < 160) el.scrollTop = el.scrollHeight;
}

function onMessagesClick(e) {
  const sug = e.target.closest("[data-suggest]");
  if (sug) {
    els.input.value = sug.textContent;
    autosize();
    els.input.focus();
    return;
  }
  const codeBtn = e.target.closest(".copy-code");
  if (codeBtn) return copyText(codeBtn.parentElement.querySelector("code").textContent);
  const act = e.target.closest("[data-act]")?.dataset.act;
  const msgEl = e.target.closest(".msg");
  if (!act || !msgEl) return;
  const m = state.messages[Number(msgEl.dataset.index)];
  if (act === "copy") copyText(m.content);
  if (act === "regen" && !state.controller) {
    state.messages.pop();
    streamReply({ content: "regenerate", regenerate: true });
  }
}

function autosize() {
  els.input.style.height = "auto";
  els.input.style.height = Math.min(els.input.scrollHeight, 200) + "px";
}

function setStreaming(on) {
  els.send.classList.toggle("stop", on);
  els.send.innerHTML = icon(on ? "stop" : "send");
  els.send.setAttribute("aria-label", on ? "Dừng" : "Gửi");
}

function stopStreaming() {
  state.controller?.abort();
}

async function sendMessage(text) {
  text = text.trim();
  if (!text) return;
  const model = currentModel();
  if (!model) {
    toast("Hãy chọn (hoặc tải) một model trước.", "error");
    return;
  }
  if (!state.currentId) {
    try {
      const conv = await api.post("/api/conversations", { model_id: model.id, system_prompt: els.system.value || null });
      state.currentId = conv.id;
    } catch (e) {
      toast(e.message, "error");
      return;
    }
  }
  els.input.value = "";
  autosize();
  state.messages.push({ role: "user", content: text, created_at: new Date().toISOString() });
  await streamReply({ content: text, regenerate: false });
}

async function streamReply({ content, regenerate }) {
  const reply = { role: "assistant", content: "", created_at: new Date().toISOString() };
  state.messages.push(reply);
  renderMessages();
  const bubble = els.messages.querySelector(".msg:last-child .bubble");
  bubble.innerHTML = `<span class="typing"><i></i><i></i><i></i></span>`;

  state.controller = new AbortController();
  setStreaming(true);
  let pending = false;
  const paint = () => {
    pending = false;
    bubble.innerHTML = renderMarkdown(reply.content);
    bubble.classList.add("cursor");
    scrollBottom();
  };

  try {
    await streamSSE(
      `/api/conversations/${state.currentId}/messages`,
      { content, regenerate, params: getParams() },
      (ev) => {
        if (ev.type === "token") {
          reply.content += ev.text;
          if (!pending) {
            pending = true;
            requestAnimationFrame(paint);
          }
        } else if (ev.type === "error") {
          reply.error = true;
          reply.content = ev.message;
        }
      },
      state.controller.signal,
    );
  } catch (e) {
    if (e.name !== "AbortError") {
      reply.error = true;
      reply.content = e.message;
    }
  } finally {
    state.controller = null;
    setStreaming(false);
    if (!reply.content) {
      reply.error = true;
      reply.content = "(Không có phản hồi)";
    }
    renderMessages();
    loadConversations();
  }
}

export function refreshChatHistory() {
  loadConversations();
}
