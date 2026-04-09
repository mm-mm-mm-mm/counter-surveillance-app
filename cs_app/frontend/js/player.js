const canvas = document.getElementById("video-canvas");
const ctx    = canvas.getContext("2d");
const tsEl   = document.getElementById("timestamp-display");

let _frozen = false;

/**
 * Render a frame message onto the canvas.
 * Draws the video frame, then overlays bounding boxes and plate labels.
 */
export async function onFrame(msg) {
  if (_frozen) return;

  // Decode base64 JPEG → ImageBitmap
  const blob   = _b64ToBlob(msg.frame_data, "image/jpeg");
  const bitmap = await createImageBitmap(blob);

  // Resize canvas to match source on first frame (or if size changed)
  if (canvas.width !== bitmap.width || canvas.height !== bitmap.height) {
    canvas.width  = bitmap.width;
    canvas.height = bitmap.height;
  }

  ctx.drawImage(bitmap, 0, 0);
  bitmap.close();

  // Draw bounding boxes and labels
  if (msg.active_observations) {
    for (const obs of msg.active_observations) {
      _drawBox(obs);
    }
  }

  // Update timestamp
  if (tsEl) tsEl.textContent = msg.timestamp_display || "—";
}

export function freezeLastFrame() {
  _frozen = true;
}

export function reset() {
  _frozen = false;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (tsEl) tsEl.textContent = "—";
}

function _drawBox(obs) {
  if (!obs.bbox || obs.bbox.length < 4) return;
  const [x1, y1, x2, y2] = obs.bbox;
  const w = x2 - x1;
  const h = y2 - y1;

  // Box color based on category
  const colors = { normal: "#00c896", taxi: "#f5c800", diplomatic: "#6699ff" };
  const color  = colors[obs.category] || "#00c896";

  ctx.strokeStyle = color;
  ctx.lineWidth   = 2;
  ctx.strokeRect(x1, y1, w, h);

  // Label background + text
  const label = obs.plate_text || obs.observation_id;
  ctx.font         = "bold 12px monospace";
  const textWidth  = ctx.measureText(label).width;
  const labelH     = 18;
  const labelY     = y1 > labelH ? y1 - labelH : y1 + labelH;

  ctx.fillStyle = color;
  ctx.fillRect(x1, labelY - labelH + 2, textWidth + 6, labelH);

  ctx.fillStyle = "#000";
  ctx.fillText(label, x1 + 3, labelY - 3);
}

function _b64ToBlob(b64, type) {
  const bin = atob(b64);
  const arr = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
  return new Blob([arr], { type });
}
