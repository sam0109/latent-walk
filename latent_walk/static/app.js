const $ = (selector) => document.querySelector(selector);

const elements = {
  dropzone: $("#dropzone"),
  empty: $("#emptyState"),
  image: $("#outputImage"),
  input: $("#fileInput"),
  choose: $("#chooseButton"),
  walk: $("#walkButton"),
  walkLabel: $("#walkLabel"),
  reset: $("#resetButton"),
  download: $("#downloadButton"),
  loading: $("#loading"),
  loadingText: $("#loadingText"),
  hud: $("#stageHud"),
  stepCount: $("#stepCount"),
  drift: $("#driftValue"),
  status: $("#statusText"),
  statusDot: $("#statusDot"),
  timeline: $("#timeline"),
  frames: $("#frames"),
  defaults: $("#defaultsButton"),
  resolution: $("#resolution"),
  noiseStrength: $("#noiseStrength"),
  denoiseSteps: $("#denoiseSteps"),
  fps: $("#fps"),
};

let sourceFile = null;
let socket = null;
let walking = false;
let waitingForFrame = false;
let nextFrameMeta = null;
let currentUrl = null;
let reconnectToken = 0;
let requestStartedAt = 0;
const historyUrls = [];

const controls = [
  [elements.noiseStrength, $("#noiseStrengthOutput"), (value) => Number(value).toFixed(2)],
  [elements.denoiseSteps, $("#denoiseStepsOutput"), (value) => value],
  [elements.fps, $("#fpsOutput"), (value) => `${value} fps`],
];

for (const [input, output, format] of controls) {
  input.addEventListener("input", () => { output.value = format(input.value); });
}

function setStatus(text, state = "") {
  elements.status.textContent = text;
  elements.statusDot.className = `status-dot ${state}`;
}

function setLoading(visible, text = "Encoding starting point…") {
  elements.loading.hidden = !visible;
  elements.loadingText.textContent = text;
}

function replaceImage(blob, meta, addToHistory = true) {
  const url = URL.createObjectURL(blob);
  if (currentUrl) URL.revokeObjectURL(currentUrl);
  currentUrl = url;
  elements.image.src = url;
  elements.image.hidden = false;
  elements.empty.hidden = true;
  elements.hud.hidden = false;
  elements.stepCount.textContent = String(meta.step ?? 0).padStart(4, "0");
  elements.drift.textContent = Number(meta.change ?? 0).toFixed(3);
  elements.download.disabled = false;

  if (addToHistory && meta.step > 0 && meta.step % 2 === 0) {
    const historyUrl = URL.createObjectURL(blob);
    historyUrls.unshift(historyUrl);
    const item = document.createElement("div");
    item.className = "frame";
    item.innerHTML = `<img alt="Latent walk step ${meta.step}"><span>${String(meta.step).padStart(4, "0")}</span>`;
    item.querySelector("img").src = historyUrl;
    elements.frames.prepend(item);
    elements.timeline.hidden = false;
    while (elements.frames.children.length > 8) {
      elements.frames.lastElementChild.remove();
      URL.revokeObjectURL(historyUrls.pop());
    }
  }
}

function clearHistory() {
  historyUrls.splice(0).forEach(URL.revokeObjectURL);
  elements.frames.replaceChildren();
  elements.timeline.hidden = true;
}

function stopWalking() {
  walking = false;
  elements.walk.classList.remove("active");
  elements.dropzone.classList.remove("walking");
  elements.walkLabel.textContent = "Continue walk";
  setStatus(socket?.readyState === WebSocket.OPEN ? "Paused" : "Disconnected", "ready");
}

function requestStep() {
  if (!walking || waitingForFrame || socket?.readyState !== WebSocket.OPEN) return;
  waitingForFrame = true;
  requestStartedAt = performance.now();
  socket.send(JSON.stringify({
    type: "step",
    noiseStrength: Number(elements.noiseStrength.value),
    denoiseSteps: Number(elements.denoiseSteps.value),
  }));
}

function toggleWalk() {
  if (walking) {
    stopWalking();
    return;
  }
  walking = true;
  elements.walk.classList.add("active");
  elements.dropzone.classList.add("walking");
  elements.walkLabel.textContent = "Pause walk";
  setStatus("Walking through latent space", "busy");
  requestStep();
}

function disconnect() {
  reconnectToken += 1;
  if (socket) {
    socket.onclose = null;
    socket.close();
    socket = null;
  }
  waitingForFrame = false;
  stopWalking();
}

async function connectAndEncode() {
  if (!sourceFile) return;
  disconnect();
  const token = reconnectToken;
  clearHistory();
  setLoading(true);
  setStatus("Loading model", "busy");
  elements.walk.disabled = true;
  elements.reset.disabled = true;
  elements.resolution.disabled = true;

  const scheme = location.protocol === "https:" ? "wss" : "ws";
  socket = new WebSocket(`${scheme}://${location.host}/ws?size=${elements.resolution.value}`);
  socket.binaryType = "blob";

  socket.onopen = async () => {
    if (token !== reconnectToken) return;
    socket.send(await sourceFile.arrayBuffer());
  };

  socket.onmessage = (event) => {
    if (token !== reconnectToken) return;
    if (typeof event.data === "string") {
      const message = JSON.parse(event.data);
      if (message.type === "error") {
        setLoading(false);
        setStatus(message.message, "error");
        elements.resolution.disabled = false;
        return;
      }
      if (message.type === "status") {
        setLoading(true, message.message);
        return;
      }
      nextFrameMeta = message;
      return;
    }

    const meta = nextFrameMeta || { step: 0, distance: 0 };
    replaceImage(event.data, meta);
    nextFrameMeta = null;
    setLoading(false);
    elements.walk.disabled = false;
    elements.reset.disabled = false;
    elements.resolution.disabled = false;
    waitingForFrame = false;

    if (meta.type === "ready") {
      setStatus("Ready to begin", "ready");
      elements.walkLabel.textContent = "Begin walk";
    } else if (walking) {
      const elapsed = performance.now() - requestStartedAt;
      const delay = Math.max(0, 1000 / Number(elements.fps.value) - elapsed);
      window.setTimeout(requestStep, delay);
    }
  };

  socket.onerror = () => {
    setLoading(false);
    setStatus("Could not reach the local model server", "error");
    elements.resolution.disabled = false;
  };

  socket.onclose = () => {
    if (token !== reconnectToken) return;
    setLoading(false);
    stopWalking();
    elements.walk.disabled = true;
  };
}

function selectFile(file) {
  if (!file || !file.type.startsWith("image/")) {
    setStatus("Choose a JPEG, PNG, or WebP image", "error");
    return;
  }
  if (file.size > 12 * 1024 * 1024) {
    setStatus("Image must be smaller than 12 MB", "error");
    return;
  }
  sourceFile = file;
  connectAndEncode();
}

elements.choose.addEventListener("click", () => elements.input.click());
elements.input.addEventListener("change", () => selectFile(elements.input.files[0]));
elements.walk.addEventListener("click", toggleWalk);
elements.reset.addEventListener("click", connectAndEncode);
elements.resolution.addEventListener("change", () => sourceFile && connectAndEncode());
elements.download.addEventListener("click", () => {
  if (!currentUrl) return;
  const link = document.createElement("a");
  link.href = currentUrl;
  link.download = `latent-walk-${elements.stepCount.textContent}.jpg`;
  link.click();
});

elements.defaults.addEventListener("click", () => {
  const defaults = ["0.45", "2", "4"];
  controls.forEach(([input], index) => {
    input.value = defaults[index];
    input.dispatchEvent(new Event("input"));
  });
});

for (const eventName of ["dragenter", "dragover"]) {
  elements.dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    elements.dropzone.classList.add("dragging");
  });
}
for (const eventName of ["dragleave", "drop"]) {
  elements.dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    elements.dropzone.classList.remove("dragging");
  });
}
elements.dropzone.addEventListener("drop", (event) => selectFile(event.dataTransfer.files[0]));
window.addEventListener("beforeunload", disconnect);
