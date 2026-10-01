import { api } from "./api.js";

const listeners = new Set();

export const store = {
  models: [],
  tasks: {},
  device: null,

  onModels(fn) {
    listeners.add(fn);
    return () => listeners.delete(fn);
  },

  async refreshModels() {
    try {
      this.models = await api.get("/api/models");
    } catch {
      this.models = [];
    }
    listeners.forEach((fn) => fn(this.models));
    return this.models;
  },

  readyModels() {
    return this.models.filter((m) => m.status === "ready");
  },

  taskLabel(task) {
    return this.tasks[task] || task || "";
  },
};
