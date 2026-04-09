import { onFrame, freezeLastFrame } from "./player.js";
import { addCard, removeCard, updateCard, clearAll } from "./observations.js";
import { renderPlate } from "./plate.js";
import { showToast } from "./app.js";

let _ws = null;

export function connect(filename) {
  const url = `ws://${location.host}/ws`;
  _ws = new WebSocket(url);

  _ws.onopen = () => {
    _ws.send(JSON.stringify({ type: "start", filename }));
  };

  _ws.onmessage = (event) => {
    let msg;
    try {
      msg = JSON.parse(event.data);
    } catch {
      return;
    }
    _dispatch(msg);
  };

  _ws.onerror = () => {
    showToast("WebSocket error — check the server.");
  };

  _ws.onclose = () => {
    _ws = null;
  };
}

export function disconnect() {
  if (_ws) {
    _ws.close();
    _ws = null;
  }
}

function _dispatch(msg) {
  switch (msg.type) {
    case "frame":
      _handleFrame(msg);
      break;
    case "session_end":
      _handleSessionEnd(msg);
      break;
    case "metadata_warning":
      showToast(msg.message, "warning");
      break;
    case "error":
      showToast(msg.message || "Unknown error");
      break;
  }
}

function _handleFrame(msg) {
  // Render video frame
  onFrame(msg);

  // Track new observations
  if (msg.new_observations) {
    for (const obsId of msg.new_observations) {
      const obs = (msg.active_observations || []).find(o => o.observation_id === obsId);
      if (obs) {
        addCard(obs);
        // Render the plate widget inside the newly-created card
        const plateContainer = document.querySelector(`#card-${CSS.escape(obsId)} .plate-container`);
        if (plateContainer) {
          renderPlate(plateContainer, obs.plate_text, obs.plate_color, obs.nationality);
        }
      }
    }
  }

  // Update existing cards
  if (msg.active_observations) {
    for (const obs of msg.active_observations) {
      updateCard(obs);
      // Update plate widget if plate text was just resolved
      const plateContainer = document.querySelector(`#card-${CSS.escape(obs.observation_id)} .plate-container`);
      if (plateContainer && obs.plate_text) {
        const existing = plateContainer.querySelector(".plate-widget");
        if (!existing || existing.textContent === "—") {
          renderPlate(plateContainer, obs.plate_text, obs.plate_color, obs.nationality);
        }
      }
    }
  }

  // Remove departed observations
  if (msg.departed_observations) {
    for (const obsId of msg.departed_observations) {
      removeCard(obsId);
    }
  }
}

function _handleSessionEnd(msg) {
  freezeLastFrame();

  // Keep final cards visible but stop updating
  const completeEl = document.getElementById("session-complete");
  if (completeEl) {
    completeEl.textContent = `Session complete — ${msg.csv_path}`;
    completeEl.classList.remove("hidden");
  }

  disconnect();
}
