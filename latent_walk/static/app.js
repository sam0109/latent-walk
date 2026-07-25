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
  video: $("#videoButton"),
  loading: $("#loading"),
  loadingText: $("#loadingText"),
  loadingHint: $("#loadingHint"),
  hud: $("#stageHud"),
  stepCount: $("#stepCount"),
  drift: $("#driftValue"),
  semanticHud: $("#semanticHud"),
  semantic: $("#semanticValue"),
  status: $("#statusText"),
  statusDot: $("#statusDot"),
  timeline: $("#timeline"),
  frames: $("#frames"),
  defaults: $("#defaultsButton"),
  resolution: $("#resolution"),
  noiseStrength: $("#noiseStrength"),
  denoiseSteps: $("#denoiseSteps"),
  fps: $("#fps"),
  frequencyEnabled: $("#frequencyEnabled"),
  frequencyLow: $("#frequencyLow"),
  frequencyMid: $("#frequencyMid"),
  frequencyHigh: $("#frequencyHigh"),
  frequencyPersistence: $("#frequencyPersistence"),
  clipEnabled: $("#clipEnabled"),
  clipStep: $("#clipStep"),
  clipMomentum: $("#clipMomentum"),
  clipGuidance: $("#clipGuidance"),
  ipEnabled: $("#ipEnabled"),
  ipWeight: $("#ipWeight"),
  ipMemory: $("#ipMemory"),
  ipLag: $("#ipLag"),
  ipDecay: $("#ipDecay"),
};

let sourceFile = null;
let socket = null;
let walking = false;
let waitingForFrame = false;
let nextFrameMeta = null;
let currentUrl = null;
let reconnectToken = 0;
let requestStartedAt = 0;
let busyBlocked = false;
let busyRetryTimer = null;
let exportingVideo = false;
const historyUrls = [];

const controls = [
  [elements.noiseStrength, $("#noiseStrengthOutput"), (value) => Number(value).toFixed(2)],
  [elements.denoiseSteps, $("#denoiseStepsOutput"), (value) => value],
  [elements.fps, $("#fpsOutput"), (value) => `${value} fps`],
  [elements.frequencyLow, $("#frequencyLowOutput"), (value) => Number(value).toFixed(2)],
  [elements.frequencyMid, $("#frequencyMidOutput"), (value) => Number(value).toFixed(2)],
  [elements.frequencyHigh, $("#frequencyHighOutput"), (value) => Number(value).toFixed(2)],
  [elements.frequencyPersistence, $("#frequencyPersistenceOutput"), (value) => Number(value).toFixed(2)],
  [elements.clipStep, $("#clipStepOutput"), (value) => Number(value).toFixed(3)],
  [elements.clipMomentum, $("#clipMomentumOutput"), (value) => Number(value).toFixed(2)],
  [elements.clipGuidance, $("#clipGuidanceOutput"), (value) => Number(value).toFixed(3)],
  [elements.ipWeight, $("#ipWeightOutput"), (value) => Number(value).toFixed(2)],
  [elements.ipLag, $("#ipLagOutput"), (value) => value],
  [elements.ipDecay, $("#ipDecayOutput"), (value) => Number(value).toFixed(2)],
];

for (const [input, output, format] of controls) {
  input.addEventListener("input", () => { output.value = format(input.value); });
}
for (const checkbox of [
  elements.frequencyEnabled,
  elements.clipEnabled,
  elements.ipEnabled,
]) {
  checkbox.addEventListener("click", (event) => event.stopPropagation());
}

function setStatus(text, state = "") {
  elements.status.textContent = text;
  elements.statusDot.className = `status-dot ${state}`;
}

function setLoading(visible, text = "Encoding starting point…", hint = "The first denoised step loads SDXL-Turbo") {
  elements.loading.hidden = !visible;
  elements.loadingText.textContent = text;
  elements.loadingHint.textContent = hint;
}

function discardRecording() {
  exportingVideo = false;
  elements.video.disabled = true;
}

function downloadRecording() {
  if (
    exportingVideo
    || Number(elements.stepCount.textContent) < 1
    || socket?.readyState !== WebSocket.OPEN
  ) return;
  if (walking) stopWalking();

  exportingVideo = true;
  elements.video.disabled = true;
  setStatus("Encoding fixed-rate MP4", "busy");
  socket.send(JSON.stringify({
    type: "export",
    fps: Number(elements.fps.value),
  }));
}

async function replaceImage(blob, meta, addToHistory = true) {
  elements.video.disabled = Number(meta.step ?? 0) < 1;
  const url = URL.createObjectURL(blob);
  if (currentUrl) URL.revokeObjectURL(currentUrl);
  currentUrl = url;
  elements.image.src = url;
  elements.image.hidden = false;
  elements.empty.hidden = true;
  elements.hud.hidden = false;
  elements.stepCount.textContent = String(meta.step ?? 0).padStart(4, "0");
  elements.drift.textContent = Number(meta.change ?? 0).toFixed(3);
  elements.semanticHud.hidden = meta.semanticChange == null;
  if (meta.semanticChange != null) {
    elements.semantic.textContent = Number(meta.semanticChange).toFixed(3);
  }
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
    experiments: {
      frequency: {
        enabled: elements.frequencyEnabled.checked,
        low: Number(elements.frequencyLow.value),
        mid: Number(elements.frequencyMid.value),
        high: Number(elements.frequencyHigh.value),
        persistence: Number(elements.frequencyPersistence.value),
      },
      clip: {
        enabled: elements.clipEnabled.checked,
        semanticStep: Number(elements.clipStep.value),
        momentum: Number(elements.clipMomentum.value),
        guidance: Number(elements.clipGuidance.value),
      },
      ipAdapter: {
        enabled: elements.ipEnabled.checked,
        weight: Number(elements.ipWeight.value),
        memory: elements.ipMemory.value,
        lag: Number(elements.ipLag.value),
        decay: Number(elements.ipDecay.value),
      },
    },
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
  if (busyRetryTimer) {
    window.clearTimeout(busyRetryTimer);
    busyRetryTimer = null;
  }
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
  busyBlocked = false;
  clearHistory();
  discardRecording();
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

  socket.onmessage = async (event) => {
    if (token !== reconnectToken) return;
    if (typeof event.data === "string") {
      const message = JSON.parse(event.data);
      if (message.type === "error") {
        setLoading(false);
        setStatus(message.message, "error");
        elements.resolution.disabled = false;
        return;
      }
      if (message.type === "busy") {
        busyBlocked = true;
        walking = false;
        setLoading(
          true,
          "The studio is occupied",
          "Waiting for the current visitor to finish. Retrying automatically…",
        );
        setStatus("Another walk is in progress", "error");
        elements.walk.disabled = true;
        return;
      }
      if (message.type === "expired") {
        walking = false;
        setLoading(
          true,
          "Your idle session was released",
          "Reset the image when you are ready to continue.",
        );
        setStatus("Session released", "error");
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
    if (meta.type === "video") {
      const url = URL.createObjectURL(event.data);
      const link = document.createElement("a");
      link.href = url;
      link.download = meta.filename;
      link.click();
      window.setTimeout(() => URL.revokeObjectURL(url), 1000);
      nextFrameMeta = null;
      exportingVideo = false;
      elements.video.disabled = false;
      setLoading(false);
      setStatus(`Exported ${meta.frames} frames at ${meta.fps} fps`, "ready");
      return;
    }
    await replaceImage(event.data, meta);
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
    if (busyBlocked) {
      busyRetryTimer = window.setTimeout(connectAndEncode, 5000);
      return;
    }
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
elements.video.addEventListener("click", downloadRecording);

elements.defaults.addEventListener("click", () => {
  const defaults = ["0.45", "2", "4", "1", "1", "1", "0.5", "0.08", "0.85", "0.005", "0.2", "4", "0.85"];
  controls.forEach(([input], index) => {
    input.value = defaults[index];
    input.dispatchEvent(new Event("input"));
  });
  elements.frequencyEnabled.checked = false;
  elements.clipEnabled.checked = false;
  elements.ipEnabled.checked = false;
  elements.ipMemory.value = "previous";
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
window.addEventListener("beforeunload", () => {
  discardRecording();
  disconnect();
});
