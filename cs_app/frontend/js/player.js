const canvas = document.getElementById("video-canvas");
const ctx    = canvas.getContext("2d");
const tsEl   = document.getElementById("timestamp-display");

let _frozen = false;

// Current letterbox transform: maps video coords → canvas coords
let _transform = { dx: 0, dy: 0, scale: 1 };

/**
 * Render a frame message onto the canvas.
 * Scales the image to fill the canvas while maintaining aspect ratio (letterbox).
 * Bounding box coordinates are remapped from video space to canvas space.
 */
export async function onFrame(msg) {
  if (_frozen) return;

  const blob   = _b64ToBlob(msg.frame_data, "image/jpeg");
  const bitmap = await createImageBitmap(blob);

  // Match canvas buffer to its CSS display size (the wrapper div)
  const wrapper = canvas.parentElement;
  const cw = wrapper.clientWidth;
  const ch = wrapper.clientHeight;

  if (canvas.width !== cw || canvas.height !== ch) {
    canvas.width  = cw;
    canvas.height = ch;
  }

  // Compute letterbox transform (fit bitmap inside canvas, preserve aspect ratio)
  const scale = Math.min(cw / bitmap.width, ch / bitmap.height);
  const dw    = Math.round(bitmap.width  * scale);
  const dh    = Math.round(bitmap.height * scale);
  const dx    = Math.round((cw - dw) / 2);
  const dy    = Math.round((ch - dh) / 2);
  _transform  = { dx, dy, scale };

  // Draw frame
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, cw, ch);
  ctx.drawImage(bitmap, dx, dy, dw, dh);
  bitmap.close();

  // Draw bounding boxes (coords are in video pixel space → transform to canvas space)
  if (msg.active_observations) {
    for (const obs of msg.active_observations) {
      _drawBox(obs);
    }
  }

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

  const { dx, dy, scale } = _transform;
  const [vx1, vy1, vx2, vy2] = obs.bbox;

  // Remap from video pixel space to canvas space
  const x1 = dx + vx1 * scale;
  const y1 = dy + vy1 * scale;
  const w  = (vx2 - vx1) * scale;
  const h  = (vy2 - vy1) * scale;

  const color = "#00c896";

  ctx.strokeStyle = color;
  ctx.lineWidth   = 2;
  ctx.strokeRect(x1, y1, w, h);

  // Label
  const label     = obs.plate_text || obs.observation_id;
  ctx.font        = "bold 12px monospace";
  const textWidth = ctx.measureText(label).width;
  const labelH    = 18;
  const labelY    = y1 > labelH ? y1 - labelH : y1 + labelH;

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
