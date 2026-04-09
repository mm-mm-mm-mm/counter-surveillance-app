/**
 * Renders a licence plate widget into a container element.
 * @param {HTMLElement} container
 * @param {string} text       - plate text (may be empty)
 * @param {string} plateColor - "white" | "yellow" | "blue"
 * @param {string} nationality
 */
export function renderPlate(container, text, plateColor, nationality) {
  container.innerHTML = "";

  const plate = document.createElement("div");
  plate.className = `plate-widget plate-${plateColor || "white"}`;
  plate.textContent = text || "—";

  if (nationality && nationality !== "Unknown") {
    const nat = document.createElement("span");
    nat.style.cssText = "font-size:0.65rem;font-weight:400;margin-left:4px;opacity:0.7;";
    nat.textContent = nationality;
    plate.appendChild(nat);
  }

  container.appendChild(plate);
}
