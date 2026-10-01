import { api } from "./api.js";
import { store } from "./state.js";
import { $, applyTheme, escapeHtml, getThemePref, toast } from "./ui.js";

export function initSettings() {
  $("#tokenForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const token = $("#hfToken").value.trim();
    if (!token) return;
    await saveToken(token);
  });
  $("#clearToken").addEventListener("click", () => saveToken(""));
  $("#themeSeg").addEventListener("click", (e) => {
    const b = e.target.closest("[data-theme-opt]");
    if (b) applyTheme(b.dataset.themeOpt);
  });
  $("#unloadBtn").addEventListener("click", async () => {
    await api.post("/api/models/unload");
    toast("Đã giải phóng bộ nhớ", "success");
    showSettings();
  });
}

async function saveToken(token) {
  try {
    const s = await api.put("/api/settings", { hf_token: token });
    $("#hfToken").value = "";
    renderToken(s);
    toast(token ? "Đã lưu token" : "Đã xoá token", "success");
  } catch (e) {
    toast(e.message, "error");
  }
}

function renderToken(s) {
  $("#tokenStatus").textContent = s.hf_token_set ? `Token hiện tại: ${s.hf_token_hint}` : "Chưa cấu hình token.";
}

export async function showSettings() {
  applyTheme(getThemePref());
  try {
    const [s, sys] = await Promise.all([api.get("/api/settings"), api.get("/api/system")]);
    renderToken(s);
    const d = sys.device;
    const ready = store.readyModels();
    const totalSize = store.models.reduce((a, m) => a + (m.size_bytes || 0), 0);
    $("#sysInfo").innerHTML = `
      <dt>Thiết bị</dt><dd>${escapeHtml(d.device.toUpperCase())}${d.gpu ? ` · ${escapeHtml(d.gpu)}` : ""}</dd>
      <dt>PyTorch</dt><dd>${escapeHtml(d.torch || "Chưa cài")}</dd>
      <dt>Model đang nạp</dt><dd>${escapeHtml(sys.loaded_model || "Không có")}</dd>
      <dt>Model sẵn sàng</dt><dd>${ready.length}</dd>
      <dt>Dung lượng model</dt><dd>${(totalSize / 1024 ** 3).toFixed(2)} GB</dd>`;
  } catch (e) {
    toast(e.message, "error");
  }
}
