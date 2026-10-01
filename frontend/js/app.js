import { api } from "./api.js";
import { initChat, openChatWithModel, refreshChatHistory } from "./chat.js";
import { initFinetune, setFinetuneChatHandler, showFinetune } from "./finetune.js";
import { initHub, showHub } from "./hub.js";
import { initModels } from "./models.js";
import { initSettings, showSettings } from "./settings.js";
import { store } from "./state.js";
import { $, $$, applyTheme, getThemePref, initModal, toggleTheme } from "./ui.js";

const TITLES = { chat: "Chat", hub: "Model Hub", models: "Model của tôi", finetune: "Fine-tuning", settings: "Cài đặt" };
let pendingFinetuneModel = null;

function route() {
  const view = location.hash.replace(/^#\//, "") || "chat";
  const name = TITLES[view] ? view : "chat";
  $$(".view").forEach((v) => v.classList.toggle("active", v.id === `view-${name}`));
  $$(".nav a").forEach((a) => a.classList.toggle("active", a.dataset.view === name));
  $("#topbarTitle").textContent = TITLES[name];
  document.title = `${TITLES[name]} · OmniHub NLP`;
  $(".app").classList.remove("nav-open");

  if (name === "hub") showHub();
  if (name === "models") store.refreshModels();
  if (name === "chat") refreshChatHistory();
  if (name === "settings") showSettings();
  if (name === "finetune") {
    showFinetune(pendingFinetuneModel);
    pendingFinetuneModel = null;
  }
}

function goChat(modelId) {
  location.hash = "#/chat";
  openChatWithModel(modelId);
}

async function init() {
  initModal();
  applyTheme(getThemePref());
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    if (getThemePref() === "system") applyTheme("system");
  });
  $("#themeToggle").addEventListener("click", toggleTheme);
  $("#topThemeBtn").addEventListener("click", toggleTheme);
  $("#menuBtn").addEventListener("click", () => $(".app").classList.add("nav-open"));
  $("#backdrop").addEventListener("click", () => $(".app").classList.remove("nav-open"));

  try {
    const sys = await api.get("/api/system");
    store.tasks = sys.tasks;
    store.device = sys.device;
    const d = sys.device;
    $("#devicePill").textContent = d.gpu ? `GPU · ${d.gpu}` : `Thiết bị: ${d.device.toUpperCase()}`;
    $("#devicePill").title = $("#devicePill").textContent;
  } catch {
    $("#devicePill").textContent = "Không kết nối được server";
  }

  initChat();
  initHub();
  initModels({
    onChat: goChat,
    onFinetune: (id) => {
      pendingFinetuneModel = id;
      if (location.hash === "#/finetune") route();
      else location.hash = "#/finetune";
    },
  });
  initFinetune();
  setFinetuneChatHandler(goChat);
  initSettings();

  await store.refreshModels();
  window.addEventListener("hashchange", route);
  route();
}

init();
