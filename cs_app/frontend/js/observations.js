import { renderPlate } from "./plate.js";

const cardMap = new Map(); // observation_id → <li> element

/**
 * Add a new observation card at the top of the list.
 */
export function addCard(obs) {
  if (cardMap.has(obs.observation_id)) {
    updateCard(obs);
    return;
  }

  const list = document.getElementById("observation-cards");
  const li = document.createElement("li");
  li.className = "obs-card slide-in";
  li.id = `card-${obs.observation_id}`;
  li.innerHTML = _buildCardHTML(obs);

  // Prepend so newest is at top
  list.insertBefore(li, list.firstChild);
  cardMap.set(obs.observation_id, li);

  // Remove animation class after it completes to allow re-animation on re-add
  li.addEventListener("animationend", () => li.classList.remove("slide-in"), { once: true });
}

/**
 * Remove a card from the list with a slide-out animation.
 */
export function removeCard(observationId) {
  const li = cardMap.get(observationId);
  if (!li) return;

  li.classList.add("slide-out");
  li.addEventListener("animationend", () => {
    li.remove();
    cardMap.delete(observationId);
  }, { once: true });
}

/**
 * Update elapsed time and any newly available plate data.
 */
export function updateCard(obs) {
  const li = cardMap.get(obs.observation_id);
  if (!li) return;

  // Update elapsed time
  const elapsedEl = li.querySelector(".elapsed-time");
  if (elapsedEl) {
    elapsedEl.textContent = _formatElapsed(obs.elapsed_since_first);
  }

  // Update plate if it arrived after initial render
  const plateEl = li.querySelector(".plate-container");
  if (plateEl && obs.plate_text) {
    renderPlate(plateEl, obs.plate_text, obs.plate_color, obs.nationality);
  }

}

export function clearAll() {
  cardMap.clear();
  const list = document.getElementById("observation-cards");
  if (list) list.innerHTML = "";
}

function _buildCardHTML(obs) {
  const elapsed = _formatElapsed(obs.elapsed_since_first);

  return `
    <div class="plate-container"></div>
    <div class="card-row">
      <span class="label">Make:</span>
      <span class="val">${obs.make || "—"}</span>
      <span class="label">Model:</span>
      <span class="val">${obs.model || "—"}</span>
    </div>
    <div class="card-row">
      <span class="label">Color:</span>
      <span class="val">${obs.color || "—"}</span>
    </div>
    <div class="card-row">
      <span class="label">First seen:</span>
      <span class="val">${obs.first_seen || "—"}</span>
    </div>
    <div class="elapsed-time">${elapsed}</div>
  `;
}

function _formatElapsed(seconds) {
  if (typeof seconds !== "number") return "";
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `Observed for ${m > 0 ? m + "m " : ""}${s}s`;
}

// Initialise plate widget after HTML is set
export function initCardPlates() {
  for (const [obsId, li] of cardMap.entries()) {
    // Plates are rendered via updateCard, not needed here unless re-initialising
  }
}
