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
  manifest: $("#manifestButton"),
  bundle: $("#bundleButton"),
  loading: $("#loading"),
  loadingText: $("#loadingText"),
  loadingHint: $("#loadingHint"),
  hud: $("#stageHud"),
  stepCount: $("#stepCount"),
  drift: $("#driftValue"),
  semanticHud: $("#semanticHud"),
  semantic: $("#semanticValue"),
  escapeHud: $("#escapeHud"),
  escapePressure: $("#escapePressureValue"),
  status: $("#statusText"),
  statusDot: $("#statusDot"),
  timeline: $("#timeline"),
  frames: $("#frames"),
  defaults: $("#defaultsButton"),
  resolution: $("#resolution"),
  noiseStrength: $("#noiseStrength"),
  denoiseSteps: $("#denoiseSteps"),
  fps: $("#fps"),
  preset: $("#preset"),
  seed: $("#seed"),
  randomSeed: $("#randomSeedButton"),
  importManifest: $("#importManifestButton"),
  manifestInput: $("#manifestInput"),
  frequencyEnabled: $("#frequencyEnabled"),
  frequencyLow: $("#frequencyLow"),
  frequencyMid: $("#frequencyMid"),
  frequencyHigh: $("#frequencyHigh"),
  frequencyPersistence: $("#frequencyPersistence"),
  clipEnabled: $("#clipEnabled"),
  clipStep: $("#clipStep"),
  clipMomentum: $("#clipMomentum"),
  clipGuidance: $("#clipGuidance"),
  escapeEnabled: $("#escapeEnabled"),
  escapeStrength: $("#escapeStrength"),
  escapeSensitivity: $("#escapeSensitivity"),
  ipEnabled: $("#ipEnabled"),
  ipWeight: $("#ipWeight"),
  ipMemory: $("#ipMemory"),
  ipLag: $("#ipLag"),
  ipDecay: $("#ipDecay"),
  ipModulation: $("#ipModulation"),
  ipModulationRate: $("#ipModulationRate"),
  ipPulsePeriod: $("#ipPulsePeriod"),
  ipPulseDuty: $("#ipPulseDuty"),
  ipFeedbackTarget: $("#ipFeedbackTarget"),
  metricsEnabled: $("#metricsEnabled"),
  metricReadout: $("#metricReadout"),
  lpips: $("#lpipsValue"),
  multiscale: $("#multiscaleValue"),
  edge: $("#edgeValue"),
  effectiveIp: $("#effectiveIpValue"),
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
let exportingArtifact = false;
let replaySteps = [];
let replayActive = false;
let applyingSettings = false;
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
  [elements.escapeStrength, $("#escapeStrengthOutput"), (value) => Number(value).toFixed(4)],
  [elements.escapeSensitivity, $("#escapeSensitivityOutput"), (value) => Number(value).toFixed(2)],
  [elements.ipWeight, $("#ipWeightOutput"), (value) => Number(value).toFixed(2)],
  [elements.ipLag, $("#ipLagOutput"), (value) => value],
  [elements.ipDecay, $("#ipDecayOutput"), (value) => Number(value).toFixed(2)],
  [elements.ipModulationRate, $("#ipModulationRateOutput"), (value) => Number(value).toFixed(2)],
  [elements.ipPulsePeriod, $("#ipPulsePeriodOutput"), (value) => value],
  [elements.ipPulseDuty, $("#ipPulseDutyOutput"), (value) => Number(value).toFixed(2)],
  [elements.ipFeedbackTarget, $("#ipFeedbackTargetOutput"), (value) => Number(value).toFixed(3)],
];

for (const [input, output, format] of controls) {
  input.addEventListener("input", () => { output.value = format(input.value); });
}
for (const checkbox of [
  elements.frequencyEnabled,
  elements.clipEnabled,
  elements.escapeEnabled,
  elements.escapeStrength,
  elements.ipEnabled,
]) {
  checkbox.addEventListener("click", (event) => event.stopPropagation());
}

function newSeed() {
  const values = new Uint32Array(2);
  crypto.getRandomValues(values);
  return (values[0] & 0x1fffff) * 4294967296 + values[1];
}

elements.seed.value = String(newSeed());

function currentSettings() {
  return {
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
      escape: {
        enabled: elements.escapeEnabled.checked,
        strength: Number(elements.escapeStrength.value),
        sensitivity: Number(elements.escapeSensitivity.value),
      },
      ipAdapter: {
        enabled: elements.ipEnabled.checked,
        weight: Number(elements.ipWeight.value),
        memory: elements.ipMemory.value,
        lag: Number(elements.ipLag.value),
        decay: Number(elements.ipDecay.value),
        modulation: elements.ipModulation.value,
        modulationRate: Number(elements.ipModulationRate.value),
        pulsePeriod: Number(elements.ipPulsePeriod.value),
        pulseDuty: Number(elements.ipPulseDuty.value),
        feedbackTarget: Number(elements.ipFeedbackTarget.value),
      },
      metrics: {
        enabled: elements.metricsEnabled.checked,
      },
    },
  };
}

function setControl(input, value) {
  if (value == null) return;
  if (input.type === "checkbox") {
    input.checked = Boolean(value);
  } else {
    input.value = String(value);
    input.dispatchEvent(new Event("input"));
  }
}

function applySettings(settings) {
  applyingSettings = true;
  const experiments = settings?.experiments ?? {};
  const frequency = experiments.frequency ?? {};
  const clip = experiments.clip ?? {};
  const escape = experiments.escape ?? {};
  const ip = experiments.ipAdapter ?? {};
  setControl(elements.noiseStrength, settings?.noiseStrength);
  setControl(elements.denoiseSteps, settings?.denoiseSteps);
  setControl(elements.frequencyEnabled, frequency.enabled);
  setControl(elements.frequencyLow, frequency.low);
  setControl(elements.frequencyMid, frequency.mid);
  setControl(elements.frequencyHigh, frequency.high);
  setControl(elements.frequencyPersistence, frequency.persistence);
  setControl(elements.clipEnabled, clip.enabled);
  setControl(elements.clipStep, clip.semanticStep);
  setControl(elements.clipMomentum, clip.momentum);
  setControl(elements.clipGuidance, clip.guidance);
  setControl(elements.escapeEnabled, escape.enabled);
  setControl(elements.escapeStrength, escape.strength);
  setControl(elements.escapeSensitivity, escape.sensitivity);
  setControl(elements.ipEnabled, ip.enabled);
  setControl(elements.ipWeight, ip.weight);
  setControl(elements.ipMemory, ip.memory);
  setControl(elements.ipLag, ip.lag);
  setControl(elements.ipDecay, ip.decay);
  setControl(elements.ipModulation, ip.modulation);
  setControl(elements.ipModulationRate, ip.modulationRate);
  setControl(elements.ipPulsePeriod, ip.pulsePeriod);
  setControl(elements.ipPulseDuty, ip.pulseDuty);
  setControl(elements.ipFeedbackTarget, ip.feedbackTarget);
  setControl(elements.metricsEnabled, experiments.metrics?.enabled);
  applyingSettings = false;
}

const presets = {
  baseline: {},
  composition: {
    frequency: { enabled: true, low: 1.4, mid: 1, high: 0.8, persistence: 0.65 },
  },
  semantic: {
    clip: { enabled: true, semanticStep: 0.08, momentum: 0.85, guidance: 0.005 },
  },
  memory: {
    ipAdapter: { enabled: true, weight: 0.2, memory: "ema", modulation: "decay" },
  },
  turbulent: {
    frequency: { enabled: true, low: 1.2, mid: 1, high: 0.8, persistence: 0.55 },
    clip: { enabled: true, semanticStep: 0.1, momentum: 0.75, guidance: 0.005 },
    ipAdapter: { enabled: true, weight: 0.15, memory: "random", modulation: "pulse" },
  },
};

function applyPreset(name) {
  const base = currentSettings();
  base.experiments.frequency.enabled = false;
  base.experiments.clip.enabled = false;
  base.experiments.escape.enabled = false;
  base.experiments.ipAdapter.enabled = false;
  const preset = presets[name] ?? {};
  for (const [section, values] of Object.entries(preset)) {
    Object.assign(base.experiments[section], values);
  }
  applySettings(base);
}

for (const control of [
  ...controls.map(([input]) => input),
  elements.frequencyEnabled,
  elements.clipEnabled,
  elements.escapeEnabled,
  elements.ipEnabled,
  elements.ipMemory,
  elements.ipModulation,
  elements.metricsEnabled,
]) {
  control.addEventListener("change", () => {
    if (applyingSettings) return;
    elements.preset.value = "custom";
    replayActive = false;
    replaySteps = [];
  });
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
  exportingArtifact = false;
  elements.video.disabled = true;
  elements.manifest.disabled = true;
  elements.bundle.disabled = true;
}

function downloadRecording() {
  if (
    exportingArtifact
    || Number(elements.stepCount.textContent) < 1
    || socket?.readyState !== WebSocket.OPEN
  ) return;
  if (walking) stopWalking();

  exportingArtifact = true;
  elements.video.disabled = true;
  setStatus("Encoding fixed-rate MP4", "busy");
  socket.send(JSON.stringify({
    type: "export",
    fps: Number(elements.fps.value),
  }));
}

function requestExport(type, status) {
  if (
    exportingArtifact
    || Number(elements.stepCount.textContent) < 1
    || socket?.readyState !== WebSocket.OPEN
  ) return;
  if (walking) stopWalking();
  exportingArtifact = true;
  elements.video.disabled = true;
  elements.manifest.disabled = true;
  elements.bundle.disabled = true;
  setStatus(status, "busy");
  socket.send(JSON.stringify({ type }));
}

async function replaceImage(blob, meta, addToHistory = true) {
  const noGeneratedFrames = Number(meta.step ?? 0) < 1;
  elements.video.disabled = noGeneratedFrames;
  elements.manifest.disabled = noGeneratedFrames;
  elements.bundle.disabled = noGeneratedFrames;
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
  elements.escapeHud.hidden = meta.effective?.escapeActive !== true;
  elements.escapePressure.textContent = Number(
    meta.effective?.escapePressure ?? 0,
  ).toFixed(2);
  const metrics = meta.metrics ?? {};
  elements.metricReadout.hidden = Object.keys(metrics).length <= 1;
  elements.lpips.textContent = metrics.lpips == null ? "—" : Number(metrics.lpips).toFixed(3);
  elements.multiscale.textContent = metrics.pixelMultiscale == null ? "—" : Number(metrics.pixelMultiscale).toFixed(3);
  elements.edge.textContent = metrics.edgeChange == null ? "—" : Number(metrics.edgeChange).toFixed(3);
  elements.effectiveIp.textContent = meta.effective?.ipAdapterWeight == null
    ? "—"
    : Number(meta.effective.ipAdapterWeight).toFixed(3);
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
  const completedSteps = Number(elements.stepCount.textContent);
  let settings = currentSettings();
  if (replayActive) {
    const replayStep = replaySteps[completedSteps];
    if (!replayStep) {
      stopWalking();
      replayActive = false;
      setStatus("Replay complete", "ready");
      return;
    }
    settings = replayStep.settings;
    applySettings(settings);
  }
  waitingForFrame = true;
  requestStartedAt = performance.now();
  socket.send(JSON.stringify({ type: "step", ...settings }));
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
  const params = new URLSearchParams({
    size: elements.resolution.value,
    seed: elements.seed.value,
  });
  socket = new WebSocket(`${scheme}://${location.host}/ws?${params}`);
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
        exportingArtifact = false;
        setLoading(false);
        setStatus(message.message, "error");
        elements.resolution.disabled = false;
        const hasFrames = Number(elements.stepCount.textContent) > 0;
        elements.video.disabled = !hasFrames;
        elements.manifest.disabled = !hasFrames;
        elements.bundle.disabled = !hasFrames;
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
    if (["video", "manifest", "bundle"].includes(meta.type)) {
      const url = URL.createObjectURL(event.data);
      const link = document.createElement("a");
      link.href = url;
      link.download = meta.filename;
      link.click();
      window.setTimeout(() => URL.revokeObjectURL(url), 1000);
      nextFrameMeta = null;
      exportingArtifact = false;
      elements.video.disabled = false;
      elements.manifest.disabled = false;
      elements.bundle.disabled = false;
      setLoading(false);
      if (meta.type === "video") {
        setStatus(`Exported ${meta.frames} frames at ${meta.fps} fps`, "ready");
      } else if (meta.type === "bundle") {
        setStatus(`Packed ${meta.frames} frames and manifest`, "ready");
      } else {
        setStatus("Exported replay manifest", "ready");
      }
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
      elements.seed.value = String(meta.seed);
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

async function importManifest(file) {
  if (!file) return;
  let manifest;
  try {
    manifest = JSON.parse(await file.text());
  } catch {
    setStatus("Manifest is not valid JSON", "error");
    return;
  }
  const legacyCompatible = manifest.version === 1
    && Array.isArray(manifest.steps)
    && manifest.steps.every(
      (step) => step?.settings?.experiments?.escape?.enabled !== true,
    );
  if (
    (manifest.version !== 2 && !legacyCompatible)
    || !Number.isSafeInteger(manifest.seed)
    || !Array.isArray(manifest.steps)
    || manifest.truncated === true
    || manifest.steps.some(
      (step, index) => !step?.settings || step.step !== index + 1,
    )
  ) {
    setStatus("Unsupported or incomplete replay manifest", "error");
    return;
  }
  if (legacyCompatible) elements.escapeEnabled.checked = false;
  elements.seed.value = String(manifest.seed);
  replaySteps = manifest.steps;
  replayActive = true;
  elements.preset.value = "custom";
  if (replaySteps[0]) applySettings(replaySteps[0].settings);
  setStatus(`Loaded ${replaySteps.length}-step replay`, "ready");
  if (sourceFile) connectAndEncode();
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
elements.manifest.addEventListener("click", () => requestExport(
  "exportManifest",
  "Exporting replay manifest",
));
elements.bundle.addEventListener("click", () => requestExport(
  "exportBundle",
  "Packing frames and manifest",
));
elements.importManifest.addEventListener("click", () => elements.manifestInput.click());
elements.manifestInput.addEventListener("change", () => {
  importManifest(elements.manifestInput.files[0]);
  elements.manifestInput.value = "";
});
elements.randomSeed.addEventListener("click", () => {
  elements.seed.value = String(newSeed());
  replayActive = false;
  replaySteps = [];
  if (sourceFile) connectAndEncode();
});
elements.preset.addEventListener("change", () => {
  if (elements.preset.value !== "custom") applyPreset(elements.preset.value);
  replayActive = false;
  replaySteps = [];
});

elements.defaults.addEventListener("click", () => {
  const defaults = [
    "0.45", "2", "4", "1", "1", "1", "0.5", "0.08", "0.85",
    "0.005", "0.02", "1.2", "0.2", "4", "0.85", "0.08", "8", "0.5", "0.08",
  ];
  controls.forEach(([input], index) => {
    input.value = defaults[index];
    input.dispatchEvent(new Event("input"));
  });
  elements.frequencyEnabled.checked = false;
  elements.clipEnabled.checked = false;
  elements.escapeEnabled.checked = false;
  elements.ipEnabled.checked = false;
  elements.ipMemory.value = "previous";
  elements.ipModulation.value = "constant";
  elements.metricsEnabled.checked = false;
  elements.preset.value = "baseline";
  replayActive = false;
  replaySteps = [];
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
