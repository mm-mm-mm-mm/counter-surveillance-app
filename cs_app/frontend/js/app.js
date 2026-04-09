import { connect, disconnect } from "./websocket.js";
import { reset as resetPlayer } from "./player.js";
import { clearAll } from "./observations.js";

const selectorScreen = document.getElementById("screen-selector");
const playerScreen   = document.getElementById("screen-player");
const videoList      = document.getElementById("video-list");
const noVideos       = document.getElementById("no-videos");
const statusBanner   = document.getElementById("status-banner");

let _statusInterval = null;

// ── Startup ───────────────────────────────────────────────────
window.addEventListener("DOMContentLoaded", () => {
  _pollStatus();
  _loadVideoList();
});

function _pollStatus() {
  _statusInterval = setInterval(async () => {
    try {
      const res  = await fetch("/api/status");
      const data = await res.json();
      if (data.ready) {
        clearInterval(_statusInterval);
        _setReady();
      }
    } catch {
      // Server still starting
    }
  }, 1000);
}

function _setReady() {
  statusBanner.className = "status-banner ready";
  statusBanner.innerHTML = "&#10003; Models loaded — select a video to begin";
  // Enable file items
  document.querySelectorAll("#video-list li").forEach(li => li.classList.remove("disabled"));
}

async function _loadVideoList() {
  try {
    const res  = await fetch("/api/videos");
    const data = await res.json();
    const files = data.videos || [];

    if (files.length === 0) {
      noVideos.classList.remove("hidden");
      return;
    }

    for (const filename of files) {
      const li = document.createElement("li");
      li.textContent = filename;
      li.className = "disabled"; // enabled once models are ready
      li.addEventListener("click", () => _selectFile(filename));
      videoList.appendChild(li);
    }
  } catch {
    showToast("Could not fetch video list.");
  }
}

function _selectFile(filename) {
  // Switch screens
  selectorScreen.classList.remove("active");
  playerScreen.classList.add("active");

  resetPlayer();
  clearAll();

  const completeEl = document.getElementById("session-complete");
  if (completeEl) completeEl.classList.add("hidden");

  connect(filename);
}

// ── Toast ─────────────────────────────────────────────────────
let _toastTimer = null;

export function showToast(message, type = "error") {
  const toast = document.getElementById("toast");
  if (!toast) return;

  toast.textContent = message;
  toast.className = "toast";
  toast.style.removeProperty("display");

  if (_toastTimer) clearTimeout(_toastTimer);
  _toastTimer = setTimeout(() => {
    toast.classList.add("hidden");
  }, 4000);
}
