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
  recordingCanvas: $("#recordingCanvas"),
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
let busyBlocked = false;
let busyRetryTimer = null;
let recorder = null;
let videoChunks = [];
let recordingMime = "";
let recordedFrames = 0;
const historyUrls = [];
const recordingContext = elements.recordingCanvas.getContext("2d", { alpha: false });

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

function setLoading(visible, text = "Encoding starting point…", hint = "The first denoised step loads SDXL-Turbo") {
  elements.loading.hidden = !visible;
  elements.loadingText.textContent = text;
  elements.loadingHint.textContent = hint;
}

function supportedVideoType() {
  if (!window.MediaRecorder || !elements.recordingCanvas.captureStream) return "";
  const candidates = [
    "video/mp4;codecs=avc1.42E01E",
    "video/webm;codecs=vp9",
    "video/webm;codecs=vp8",
    "video/webm",
  ];
  return candidates.find((type) => MediaRecorder.isTypeSupported(type)) || "";
}

async function drawRecordingFrame(blob) {
  const bitmap = await createImageBitmap(blob);
  recordingContext.drawImage(bitmap, 0, 0, 512, 512);
  bitmap.close();
  recordedFrames += 1;
  elements.video.disabled = !recordingMime || recordedFrames < 2;
}

function startRecording() {
  if (!recordingMime) return;
  if (recorder?.state === "paused") {
    recorder.resume();
    return;
  }
  if (recorder?.state === "recording") return;

  videoChunks = [];
  const instance = new MediaRecorder(
    elements.recordingCanvas.captureStream(Number(elements.fps.value)),
    { mimeType: recordingMime, videoBitsPerSecond: 5_000_000 },
  );
  recorder = instance;
  instance.addEventListener("dataavailable", (event) => {
    if (recorder === instance && event.data.size) videoChunks.push(event.data);
  });
  instance.start(1000);
}

function pauseRecording() {
  if (recorder?.state === "recording") recorder.pause();
}

function discardRecording() {
  const instance = recorder;
  recorder = null;
  if (instance && instance.state !== "inactive") instance.stop();
  videoChunks = [];
  recordedFrames = 0;
  elements.video.disabled = true;
}

function downloadRecording() {
  if (!recorder || recordedFrames < 2) return;
  const instance = recorder;
  const wasWalking = walking;
  instance.addEventListener("stop", () => {
    if (!videoChunks.length) return;
    const blob = new Blob(videoChunks, { type: recordingMime });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `latent-walk-${Date.now()}.${recordingMime.startsWith("video/mp4") ? "mp4" : "webm"}`;
    link.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    recorder = null;
    videoChunks = [];
    recordedFrames = 0;
    elements.video.disabled = true;
    if (wasWalking) startRecording();
  }, { once: true });
  if (instance.state !== "inactive") instance.stop();
}

async function replaceImage(blob, meta, addToHistory = true) {
  await drawRecordingFrame(blob);
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
  pauseRecording();
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
  startRecording();
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
recordingMime = supportedVideoType();
if (!recordingMime) elements.video.title = "Video recording is not supported by this browser";
window.addEventListener("beforeunload", () => {
  discardRecording();
  disconnect();
});
