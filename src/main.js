import "./style.css";
import * as ort from "onnxruntime-web/webgpu";
import {
  DetectionService,
  RecognitionService,
  TextImageUnwarpingService,
  getTextImageUnwarpingPresetOptions,
  getTextRecognitionPresetOptions,
  normalizeInputToRgb,
} from "paddleocr";

const UVDOC_URL =
  "https://huggingface.co/PaddlePaddle/UVDoc_onnx/resolve/main/inference.onnx";
const DET_URL =
  "https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_det_onnx/resolve/main/inference.onnx";

const REC_MODELS = {
  server: {
    label: "PP-OCRv5 server recognizer",
    preset: "PP-OCRv5_server_rec",
    url: "https://huggingface.co/PaddlePaddle/PP-OCRv5_server_rec_onnx/resolve/main/inference.onnx",
  },
  mobile: {
    label: "PP-OCRv5 mobile recognizer",
    preset: "PP-OCRv5_mobile_rec",
    url: "https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_rec_onnx/resolve/main/inference.onnx",
  },
};

const DICT_URL =
  "https://cdn.jsdelivr.net/gh/PaddlePaddle/PaddleOCR@main/ppocr/utils/dict/ppocrv5_dict.txt";

const MODEL_CACHE = "bookocr-models-v5-exp";
const MAX_PAGE_SIDE = 2400;

ort.env.wasm.numThreads = 1;
ort.env.wasm.simd = true;
ort.env.wasm.wasmPaths =
  "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/";

const $ = (id) => document.getElementById(id);

const state = {
  file: null,
  bitmap: null,
  rightInput: null,
  leftInput: null,
  rightFlat: null,
  leftFlat: null,
  uvdoc: null,
  uvSession: null,
  detector: null,
  detSession: null,
  rightDetection: null,
  leftDetection: null,
  rightStrips: [],
  leftStrips: [],
  recognizer: null,
  recSession: null,
  recPreset: null,
  recDictionary: null,
  rightRecognition: [],
  leftRecognition: [],
  webGpuAvailable: null,
  backends: {},
  recModelBuffer: null,
  recSpec: null,
  recForcedWasm: false,
  fullInput: null,
  routedMode: null,
  routerAnalysis: null,
  generalPages: [],
  generalRecognition: [],
};

let webGpuProbePromise = null;

async function probeWebGpu() {
  if (state.webGpuAvailable !== null) {
    return state.webGpuAvailable;
  }

  if (!("gpu" in navigator)) {
    state.webGpuAvailable = false;
    return false;
  }

  try {
    const adapter = await navigator.gpu.requestAdapter();
    state.webGpuAvailable = !!adapter;
    return state.webGpuAvailable;
  } catch (error) {
    console.warn("WebGPU probe failed:", error);
    state.webGpuAvailable = false;
    return false;
  }
}

function updateRuntimeBadge() {
  const badge = $("runtimeBadge");
  const values = Object.values(state.backends);

  if (values.includes("webgpu")) {
    badge.textContent = "GPU / WebGPU";
    badge.title =
      "至少一個 ONNX 模型已用 WebGPU session 啟動；不支援的節點可由 WASM fallback。";
    return;
  }

  if (values.includes("wasm")) {
    badge.textContent = "CPU / WASM";
    badge.title = "目前模型使用 WASM/CPU。";
    return;
  }

  if (state.webGpuAvailable === true) {
    badge.textContent = "WebGPU 可用";
    badge.title = "下一個模型 session 會優先嘗試 GPU。";
  } else if (state.webGpuAvailable === false) {
    badge.textContent = "CPU / WASM";
    badge.title = "目前瀏覽器沒有可用的 WebGPU adapter。";
  } else {
    badge.textContent = "偵測 GPU…";
  }
}

function nextPaint() {
  return new Promise((resolve) => {
    requestAnimationFrame(() => requestAnimationFrame(resolve));
  });
}

async function createSessionAuto(modelBuffer, modelName) {
  const canGpu = await probeWebGpu();
  const mb = (modelBuffer.byteLength / 1024 / 1024).toFixed(1);

  if (canGpu) {
    try {
      setStatus(
        `建立 ${modelName} GPU session…`,
        `模型 ${mb} MB 已下載完成；正在 WebGPU 編譯與配置權重。`,
        86,
      );
      await nextPaint();

      const session = await ort.InferenceSession.create(modelBuffer, {
        executionProviders: ["webgpu", "wasm"],
        graphOptimizationLevel: "all",
      });

      state.backends[modelName] = "webgpu";
      updateRuntimeBadge();
      console.info(`${modelName}: WebGPU session ready`);
      return session;
    } catch (error) {
      console.warn(`${modelName}: WebGPU failed; falling back to WASM`, error);
      setStatus(
        `${modelName} GPU 啟動失敗，改用 CPU / WASM…`,
        error instanceof Error ? error.message : String(error),
        86,
      );
      await nextPaint();
    }
  }

  setStatus(
    `建立 ${modelName} CPU session…`,
    `模型 ${mb} MB 已下載完成；正在建立 WASM session。`,
    86,
  );
  await nextPaint();

  const session = await ort.InferenceSession.create(modelBuffer, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });

  state.backends[modelName] = "wasm";
  updateRuntimeBadge();
  console.info(`${modelName}: WASM session ready`);
  return session;
}

async function createWasmSession(modelBuffer, modelName) {
  const mb = (modelBuffer.byteLength / 1024 / 1024).toFixed(1);
  setStatus(
    `建立 ${modelName} CPU / WASM session…`,
    `模型 ${mb} MB；正在切換到 CPU fallback。`,
    88,
  );
  await nextPaint();

  const session = await ort.InferenceSession.create(modelBuffer, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });

  state.backends[modelName] = "wasm";
  updateRuntimeBadge();
  return session;
}

function setStatus(text, detail = "", progress = null) {
  $("status").textContent = text;
  $("detail").textContent = detail;
  if (progress != null) {
    $("progressBar").style.width = `${Math.max(0, Math.min(100, progress))}%`;
  }
}

function showCanvas(canvasId, emptyId) {
  $(canvasId).style.display = "block";
  $(emptyId).style.display = "none";
}

function clearCanvas(canvasId, emptyId, text) {
  const c = $(canvasId);
  c.width = 1;
  c.height = 1;
  c.style.display = "none";
  $(emptyId).style.display = "block";
  $(emptyId).textContent = text;
}

function drawBitmapCropToCanvas(bitmap, sx, sy, sw, sh, canvas) {
  const scale = Math.min(1, MAX_PAGE_SIDE / Math.max(sw, sh));
  canvas.width = Math.max(1, Math.round(sw * scale));
  canvas.height = Math.max(1, Math.round(sh * scale));
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(bitmap, sx, sy, sw, sh, 0, 0, canvas.width, canvas.height);
}

function canvasToPixels(canvas) {
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  return {
    width: canvas.width,
    height: canvas.height,
    data: new Uint8Array(
      imageData.data.buffer.slice(
        imageData.data.byteOffset,
        imageData.data.byteOffset + imageData.data.byteLength,
      ),
    ),
  };
}

function pixelsToCanvas(image, canvas) {
  const { width, height, data } = image;
  canvas.width = width;
  canvas.height = height;

  let rgba;
  if (data.length === width * height * 4) {
    rgba = new Uint8ClampedArray(
      data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength),
    );
  } else if (data.length === width * height * 3) {
    rgba = new Uint8ClampedArray(width * height * 4);
    for (let si = 0, di = 0; si < data.length; ) {
      rgba[di++] = data[si++];
      rgba[di++] = data[si++];
      rgba[di++] = data[si++];
      rgba[di++] = 255;
    }
  } else if (data.length === width * height) {
    rgba = new Uint8ClampedArray(width * height * 4);
    for (let i = 0, j = 0; i < data.length; i++) {
      const v = data[i];
      rgba[j++] = v;
      rgba[j++] = v;
      rgba[j++] = v;
      rgba[j++] = 255;
    }
  } else {
    throw new Error(
      `未知像素格式：${data.length} bytes for ${width}x${height}`,
    );
  }

  canvas.getContext("2d").putImageData(new ImageData(rgba, width, height), 0, 0);
}

async function fetchArrayBufferCached(url, modelName, progressBase, progressSpan) {
  const cache = "caches" in window ? await caches.open(MODEL_CACHE) : null;

  if (cache) {
    const cached = await cache.match(url);
    if (cached) {
      setStatus(`載入 ${modelName}…`, `${modelName} 已從瀏覽器快取取得。`, progressBase + progressSpan);
      return cached.arrayBuffer();
    }
  }

  const response = await fetch(url, { mode: "cors" });
  if (!response.ok) {
    throw new Error(`${resourceName} 下載失敗：HTTP ${response.status}`);
  }

  const total = Number(response.headers.get("content-length")) || 0;
  if (!response.body) {
    const buffer = await response.arrayBuffer();
    if (cache) {
      await cache.put(url, new Response(buffer.slice(0)));
    }
    return buffer;
  }

  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.byteLength;

    const ratio = total ? received / total : 0.2;
    setStatus(
      `下載 ${modelName}…`,
      total
        ? `${(received/1024/1024).toFixed(1)} / ${(total/1024/1024).toFixed(1)} MB`
        : `${(received/1024/1024).toFixed(1)} MB`,
      progressBase + progressSpan * ratio,
    );
  }

  const merged = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) {
    merged.set(chunk, offset);
    offset += chunk.byteLength;
  }

  if (cache) {
    await cache.put(url, new Response(merged.slice().buffer));
  }
  return merged.buffer;
}

async function ensureUvDoc() {
  if (state.uvdoc) return state.uvdoc;

  const model = await fetchArrayBufferCached(UVDOC_URL, "UVDoc", 2, 45);
  setStatus("建立 UVDoc session…", "ONNX Runtime Web / WASM", 50);

  state.uvSession = await createSessionAuto(model, "UVDoc");

  state.uvdoc = new TextImageUnwarpingService(
    ort,
    state.uvSession,
    getTextImageUnwarpingPresetOptions("UVDoc"),
  );
  return state.uvdoc;
}

function detectorRuntimeOptions() {
  return {
    // These map to the Colab/PaddleOCR values we settled on.
    textPixelThreshold: Number($("pixelThreshold").value),
    boxScoreThreshold: Number($("boxThreshold").value),
    unclipRatio: Number($("unclipRatio").value),
    maxSideLength: Number($("maxSideLength").value),
    limitType: "max",
    maxSideLimit: 4000,

    // Paddle's official PP-OCRv5 inference config decodes BGR.
    channelOrder: "bgr",

    // Keep raw DB boxes as geometry hints. Do not add service-level padding.
    paddingBoxVertical: 0,
    paddingBoxHorizontal: 0,
    minimumAreaThreshold: 20,
    maxCandidates: 1000,
    boxType: "quad",
  };
}

async function ensureDetector() {
  if (state.detector) return state.detector;

  const model = await fetchArrayBufferCached(DET_URL, "PP-OCRv5 mobile detector", 50, 30);
  setStatus("建立 Detector session…", "PP-OCRv5_mobile_det ONNX", 82);

  state.detSession = await createSessionAuto(
    model,
    "PP-OCRv5 Detector",
  );

  state.detector = new DetectionService(
    ort,
    state.detSession,
    detectorRuntimeOptions(),
  );

  return state.detector;
}


function bitmapToPixels(bitmap, maxSide = 2400) {
  const scale = Math.min(1, maxSide / Math.max(bitmap.width, bitmap.height));
  const canvas = document.createElement("canvas");
  canvas.width = Math.max(1, Math.round(bitmap.width * scale));
  canvas.height = Math.max(1, Math.round(bitmap.height * scale));

  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
  return canvasToPixels(canvas);
}

function boxBounds(box) {
  const pts = boxPoints(box);
  const xs = pts.map((p) => p.x);
  const ys = pts.map((p) => p.y);

  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);

  return {
    minX,
    maxX,
    minY,
    maxY,
    width: Math.max(1, maxX - minX),
    height: Math.max(1, maxY - minY),
    cx: (minX + maxX) / 2,
    cy: (minY + maxY) / 2,
  };
}

async function detectRawImage(image, statusName = "圖片") {
  const detector = await ensureDetector();
  const detectorImage = normalizeInputToRgb(image);

  return detector.run(detectorImage, {
    ...detectorRuntimeOptions(),
    onProgress(event) {
      const stageName = {
        preprocess: "前處理",
        infer: "模型推理",
        postprocess: "後處理",
      }[event.stage] || event.stage;

      setStatus(
        `Auto 分析：${statusName} ${stageName}…`,
        event.detectedCount != null ? `偵測到 ${event.detectedCount} 個文字區塊` : "",
        18,
      );
    },
  });
}

function analyzeLayoutFromBoxes(boxes, image) {
  let verticalWeight = 0;
  let horizontalWeight = 0;
  let mixedWeight = 0;

  for (const box of boxes) {
    const b = boxBounds(box);
    const areaWeight = Math.max(1, Math.sqrt(b.width * b.height));

    if (b.height >= b.width * 1.55) {
      verticalWeight += areaWeight;
    } else if (b.width >= b.height * 1.55) {
      horizontalWeight += areaWeight;
    } else {
      mixedWeight += areaWeight;
    }
  }

  const total = verticalWeight + horizontalWeight + mixedWeight || 1;
  const verticalRatio = verticalWeight / total;
  const horizontalRatio = horizontalWeight / total;
  const mixedRatio = mixedWeight / total;
  const aspect = image.width / Math.max(1, image.height);
  const looksLikeSpread = aspect >= 1.16;

  let mode;
  let reason;

  if (
    verticalRatio >= 0.52 &&
    verticalRatio >= horizontalRatio * 1.22
  ) {
    if (looksLikeSpread) {
      mode = "traditional";
      reason = "文字框大多高瘦，而且圖片接近跨頁比例。";
    } else {
      mode = "vertical-single";
      reason = "文字框大多高瘦，但圖片比較像單頁／單張直排內容。";
    }
  } else if (
    horizontalRatio >= 0.50 &&
    horizontalRatio >= verticalRatio * 1.18
  ) {
    mode = "horizontal";
    reason = looksLikeSpread
      ? "文字框大多橫向，而且圖片像攤開書籍。"
      : "文字框大多橫向，適合一般橫排文件流程。";
  } else {
    mode = "general";
    reason = "文字框方向較混合，先用一般圖片／招牌 OCR。";
  }

  const winning = Math.max(verticalRatio, horizontalRatio, mixedRatio);
  const confidence = Math.round(Math.min(99, 55 + winning * 44));

  return {
    mode,
    confidence,
    verticalRatio,
    horizontalRatio,
    mixedRatio,
    looksLikeSpread,
    boxCount: boxes.length,
    reason,
  };
}

function displayModeName(mode) {
  return {
    auto: "Auto",
    traditional: "傳統直書",
    "vertical-single": "直排單頁",
    horizontal: "橫排書籍／文件",
    general: "一般圖片／招牌",
  }[mode] || mode;
}

function updateModeUi(mode, analysis = null) {
  const isTraditional = mode === "traditional";
  const isBookish = isTraditional || mode === "horizontal";

  $("verticalPipeline").classList.toggle("hidden", !isTraditional);
  $("generalPipeline").classList.toggle("hidden", isTraditional);
  $("verticalSteps").classList.toggle("hidden", !isTraditional);

  document.querySelectorAll(".book-only-control").forEach((el) => {
    el.classList.toggle("dimmed", !isBookish && $("ocrMode").value !== "auto");
  });

  if (mode === "traditional") {
    $("readingOrderLabel").textContent = "右頁 → 左頁；右欄 → 左欄";
    $("readingOrderHint").textContent = "Column 01 現在也會真的顯示在畫面最右邊。";
    $("fullTextOrder").textContent =
      "傳統直書：右頁 → 左頁；每頁最右欄 → 最左欄；每欄上 → 下。";
  } else if (mode === "vertical-single") {
    $("readingOrderLabel").textContent = "最右直欄 → 最左直欄";
    $("readingOrderHint").textContent = "單頁直排不切成左右跨頁。";
    $("fullTextOrder").textContent =
      "直排單頁：文字區塊依右 → 左排序；高瘦區塊會自動旋正後辨識。";
  } else if (mode === "horizontal") {
    $("readingOrderLabel").textContent = "上 → 下；每行左 → 右";
    $("readingOrderHint").textContent = analysis?.looksLikeSpread
      ? "若為跨頁橫排書籍，左頁先於右頁。"
      : "單頁橫排依一般閱讀順序。";
    $("fullTextOrder").textContent =
      analysis?.looksLikeSpread
        ? "橫排跨頁：左頁 → 右頁；每頁上 → 下、左 → 右。"
        : "橫排單頁：上 → 下；同一行左 → 右。";
  } else {
    $("readingOrderLabel").textContent = "一般版面：上 → 下、左 → 右";
    $("readingOrderHint").textContent =
      "招牌／海報等混合版面先依文字區塊位置排序。";
    $("fullTextOrder").textContent =
      "一般圖片：以文字區塊的視覺位置排序；必要時可手動切換模式重跑。";
  }
}

function showRouterAnalysis(analysis, forced = false) {
  $("routerCard").classList.remove("hidden");
  $("routerTitle").textContent =
    `${forced ? "手動模式" : "Auto 判定"}：${displayModeName(analysis.mode)}`;
  $("routerConfidence").textContent =
    forced ? "手動指定" : `信心 ${analysis.confidence}%`;

  const v = Math.round((analysis.verticalRatio || 0) * 100);
  const h = Math.round((analysis.horizontalRatio || 0) * 100);
  const m = Math.round((analysis.mixedRatio || 0) * 100);

  $("routerDetail").textContent =
    forced
      ? analysis.reason
      : `${analysis.reason} Detector：${analysis.boxCount} 區塊；直向 ${v}% / 橫向 ${h}% / 混合 ${m}%。`;
}

async function resolveRunMode() {
  const selected = $("ocrMode").value;

  if (selected !== "auto") {
    const aspect = state.fullInput.width / Math.max(1, state.fullInput.height);
    const forcedAnalysis = {
      mode: selected,
      confidence: 100,
      verticalRatio: selected === "traditional" ? 1 : 0,
      horizontalRatio: selected === "horizontal" ? 1 : 0,
      mixedRatio: selected === "general" ? 1 : 0,
      looksLikeSpread: aspect >= 1.16,
      boxCount: 0,
      reason: `使用者手動指定「${displayModeName(selected)}」。`,
    };
    state.routerAnalysis = forcedAnalysis;
    state.routedMode = selected;
    showRouterAnalysis(forcedAnalysis, true);
    updateModeUi(selected, forcedAnalysis);
    return selected;
  }

  setStatus("Auto：先分析文字方向…", "這一步只做輕量 Detector，不先假設是書。", 5);
  const boxes = await detectRawImage(state.fullInput, "原圖");
  const analysis = analyzeLayoutFromBoxes(boxes, state.fullInput);

  state.routerAnalysis = analysis;
  state.routedMode = analysis.mode;

  showRouterAnalysis(analysis, false);
  updateModeUi(analysis.mode, analysis);
  return analysis.mode;
}

function splitPixelsForBook() {
  const image = state.fullInput;
  const w = image.width;
  const h = image.height;

  const splitPct = Number($("splitRange").value) / 100;
  const gutterPct = Number($("gutterRange").value) / 100;
  const splitX = Math.round(w * splitPct);
  const halfGutter = Math.round((w * gutterPct) / 2);

  const leftEnd = Math.max(1, splitX - halfGutter);
  const rightStart = Math.min(w - 1, splitX + halfGutter);

  function cropPixels(src, x0, y0, cw, ch) {
    const channels =
      src.data.length === src.width * src.height * 4 ? 4 :
      src.data.length === src.width * src.height * 3 ? 3 : 1;
    const data = new Uint8Array(cw * ch * 4);
    let di = 0;

    for (let y = 0; y < ch; y++) {
      for (let x = 0; x < cw; x++) {
        const sx = x0 + x;
        const sy = y0 + y;
        const si = (sy * src.width + sx) * channels;

        if (channels === 4) {
          data[di++] = src.data[si];
          data[di++] = src.data[si + 1];
          data[di++] = src.data[si + 2];
          data[di++] = src.data[si + 3];
        } else if (channels === 3) {
          data[di++] = src.data[si];
          data[di++] = src.data[si + 1];
          data[di++] = src.data[si + 2];
          data[di++] = 255;
        } else {
          const v = src.data[sy * src.width + sx];
          data[di++] = v;
          data[di++] = v;
          data[di++] = v;
          data[di++] = 255;
        }
      }
    }

    return { width: cw, height: ch, data };
  }

  return {
    left: cropPixels(image, 0, 0, leftEnd, h),
    right: cropPixels(image, rightStart, 0, w - rightStart, h),
  };
}

function sortBoxesHorizontal(boxes) {
  const items = boxes.map((box) => ({ box, b: boxBounds(box) }));
  if (!items.length) return [];

  const heights = items.map((x) => x.b.height).sort((a, b) => a - b);
  const medianH = heights[Math.floor(heights.length / 2)] || 20;
  const lineTolerance = Math.max(10, medianH * 0.72);

  items.sort((a, b) => a.b.cy - b.b.cy);

  const lines = [];
  for (const item of items) {
    let best = null;
    let bestDist = Infinity;

    for (const line of lines) {
      const dist = Math.abs(item.b.cy - line.cy);
      if (dist < lineTolerance && dist < bestDist) {
        best = line;
        bestDist = dist;
      }
    }

    if (!best) {
      best = { cy: item.b.cy, items: [] };
      lines.push(best);
    }

    best.items.push(item);
    best.cy = best.items.reduce((sum, x) => sum + x.b.cy, 0) / best.items.length;
  }

  lines.sort((a, b) => a.cy - b.cy);

  const out = [];
  for (const line of lines) {
    line.items.sort((a, b) => a.b.cx - b.b.cx);
    out.push(...line.items.map((x) => x.box));
  }
  return out;
}

function sortBoxesVertical(boxes) {
  return [...boxes].sort((a, b) => {
    const aa = boxBounds(a);
    const bb = boxBounds(b);
    const dx = bb.cx - aa.cx;
    if (Math.abs(dx) > Math.max(aa.width, bb.width) * 0.55) return dx;
    return aa.cy - bb.cy;
  });
}

function drawGeneralOverlay(image, boxes, canvas) {
  pixelsToCanvas(image, canvas);
  const ctx = canvas.getContext("2d");

  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  ctx.strokeStyle = "rgba(220, 55, 65, .88)";
  ctx.fillStyle = "rgba(25, 105, 215, .96)";
  ctx.lineWidth = Math.max(2, canvas.width / 600);
  ctx.font = `bold ${Math.max(14, canvas.width / 44)}px system-ui`;

  boxes.forEach((box, index) => {
    const pts = boxPoints(box);
    if (!pts.length) return;

    ctx.beginPath();
    ctx.moveTo(pts[0].x, pts[0].y);
    for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y);
    ctx.closePath();
    ctx.stroke();

    const b = boxBounds(box);
    ctx.fillText(
      String(index + 1),
      Math.max(3, Math.min(canvas.width - 40, b.minX + 3)),
      Math.max(17, b.minY + 16),
    );
  });
}

async function recognizeGeneralBoxes(image, boxes, recognizer, pageName) {
  const results = [];

  for (let i = 0; i < boxes.length; i++) {
    setStatus(
      `${pageName} OCR：區塊 ${i + 1}/${boxes.length}`,
      "PP-OCRv5 recognition",
      55 + 38 * ((i + 1) / Math.max(1, boxes.length)),
    );

    const rec = await recognizer.run(
      normalizeInputToRgb(image),
      [boxes[i]],
      { ordering: { sortByReadingOrder: false } },
    );

    results.push({
      pageName,
      index: i + 1,
      text: rec[0]?.text ?? "",
      confidence: Number(rec[0]?.confidence ?? 0),
    });
  }

  return results;
}

function renderGeneralPages(pages) {
  const root = $("generalPages");
  root.innerHTML = "";

  for (const page of pages) {
    const card = document.createElement("article");
    card.className = "card general-page-card";

    const meta = document.createElement("div");
    meta.className = "general-page-meta";

    const title = document.createElement("h3");
    title.textContent = page.name;

    const pill = document.createElement("span");
    pill.className = "route-pill";
    pill.textContent = `${page.boxes.length} 文字區塊`;

    const canvas = document.createElement("canvas");
    drawGeneralOverlay(page.image, page.boxes, canvas);

    meta.appendChild(title);
    meta.appendChild(pill);
    card.appendChild(meta);
    card.appendChild(canvas);
    root.appendChild(card);
  }
}

function renderGeneralRecognition(items) {
  const root = $("generalRecognition");
  root.innerHTML = "";

  items.forEach((item) => {
    const row = document.createElement("div");
    row.className = "recognition-item";

    const name = document.createElement("div");
    name.className = "col-name";
    name.textContent = `${item.pageName} #${item.index}`;

    const text = document.createElement("div");
    text.className = "rec-text";
    text.textContent = item.text || "（空白）";

    const confidence = document.createElement("div");
    confidence.className = "confidence";
    confidence.textContent = `${(item.confidence * 100).toFixed(1)}%`;

    row.appendChild(name);
    row.appendChild(text);
    row.appendChild(confidence);
    root.appendChild(row);
  });

  const nonEmpty = items.filter((x) => x.text.trim()).length;
  $("generalStats").textContent = `${nonEmpty}/${items.length} 區塊有文字`;
}

async function runGeneralPipeline(mode) {
  $("generalPipeline").classList.remove("hidden");
  $("verticalPipeline").classList.add("hidden");

  let pages = [];

  if (mode === "horizontal" && state.routerAnalysis?.looksLikeSpread) {
    const halves = splitPixelsForBook();
    const uv = await ensureUvDoc();

    setStatus("橫排書籍：展平左頁…", "", 30);
    const leftFlat = (await uv.run(halves.left)).doctrImage;

    setStatus("橫排書籍：展平右頁…", "", 38);
    const rightFlat = (await uv.run(halves.right)).doctrImage;

    pages = [
      { name: "左頁", image: leftFlat },
      { name: "右頁", image: rightFlat },
    ];
  } else if (mode === "horizontal") {
    // A single photographed horizontal page can also benefit from UVDoc.
    const uv = await ensureUvDoc();
    setStatus("橫排單頁：UVDoc 展平…", "", 30);
    const flat = (await uv.run(state.fullInput)).doctrImage;
    pages = [{ name: "單頁", image: flat }];
  } else {
    pages = [{ name: mode === "vertical-single" ? "直排單頁" : "一般圖片", image: state.fullInput }];
  }

  const recognizer = await ensureRecognizer(false);
  const allResults = [];

  for (const page of pages) {
    setStatus(`${page.name}：偵測文字區塊…`, "", 42);
    let boxes = await detectRawImage(page.image, page.name);

    if (mode === "vertical-single") {
      boxes = sortBoxesVertical(boxes);
    } else {
      boxes = sortBoxesHorizontal(boxes);
    }

    page.boxes = boxes;

    const pageResults = await recognizeGeneralBoxes(
      page.image,
      boxes,
      recognizer,
      page.name,
    );

    allResults.push(...pageResults);
  }

  state.generalPages = pages;
  state.generalRecognition = allResults;

  renderGeneralPages(pages);
  renderGeneralRecognition(allResults);

  const textParts = [];
  for (const page of pages) {
    const texts = allResults
      .filter((x) => x.pageName === page.name)
      .map((x) => x.text.trim())
      .filter(Boolean);

    if (texts.length) textParts.push(texts.join("\n"));
  }

  const text = textParts.join("\n\n");
  $("fullText").value = text;
  $("copyTextBtn").disabled = !text;
  $("downloadTextBtn").disabled = !text;

  setStatus(
    "OCR 完成。",
    `使用 ${displayModeName(mode)} 流程；${allResults.length} 個文字區塊。`,
    100,
  );
}

function splitPages() {
  if (!state.bitmap) return;

  const w = state.bitmap.width;
  const h = state.bitmap.height;
  const splitPct = Number($("splitRange").value) / 100;
  const gutterPct = Number($("gutterRange").value) / 100;
  const splitX = w * splitPct;
  const halfGutter = (w * gutterPct) / 2;
  const leftEnd = Math.max(1, splitX - halfGutter);
  const rightStart = Math.min(w - 1, splitX + halfGutter);

  const rc = document.createElement("canvas");
  const lc = document.createElement("canvas");

  drawBitmapCropToCanvas(state.bitmap, rightStart, 0, w - rightStart, h, rc);
  drawBitmapCropToCanvas(state.bitmap, 0, 0, leftEnd, h, lc);

  state.rightInput = canvasToPixels(rc);
  state.leftInput = canvasToPixels(lc);
  state.rightFlat = null;
  state.leftFlat = null;
  state.rightDetection = null;
  state.leftDetection = null;
  state.rightStrips = [];
  state.leftStrips = [];
  state.rightRecognition = [];
  state.leftRecognition = [];

  clearCanvas("rightFlat", "rightFlatEmpty", "等待 UVDoc");
  clearCanvas("leftFlat", "leftFlatEmpty", "等待 UVDoc");
  clearCanvas("rightOverlay", "rightOverlayEmpty", "等待 Detector");
  clearCanvas("leftOverlay", "leftOverlayEmpty", "等待 Detector");
  $("rightStats").textContent = "尚未偵測";
  $("leftStats").textContent = "尚未偵測";
  $("rightStripStats").textContent = "尚未抽欄";
  $("leftStripStats").textContent = "尚未抽欄";
  $("rightStrips").innerHTML = '<div class="empty">等待 V3 抽欄</div>';
  $("leftStrips").innerHTML = '<div class="empty">等待 V3 抽欄</div>';
  $("rightRecStats").textContent = "尚未辨識";
  $("leftRecStats").textContent = "尚未辨識";
  $("rightRecognition").innerHTML = '<div class="empty">等待 recognition</div>';
  $("leftRecognition").innerHTML = '<div class="empty">等待 recognition</div>';
  $("fullText").value = "";
  $("copyTextBtn").disabled = true;
  $("downloadTextBtn").disabled = true;

  setStatus("左右頁切割完成。", "下一步可跑 UVDoc。", 0);
}

async function runUvDoc() {
  if (!state.rightInput || !state.leftInput) splitPages();

  const uv = await ensureUvDoc();

  setStatus("UVDoc：正在展平右頁…", "", 55);
  state.rightFlat = (await uv.run(state.rightInput)).doctrImage;
  pixelsToCanvas(state.rightFlat, $("rightFlat"));
  showCanvas("rightFlat", "rightFlatEmpty");

  setStatus("UVDoc：正在展平左頁…", "", 72);
  state.leftFlat = (await uv.run(state.leftInput)).doctrImage;
  pixelsToCanvas(state.leftFlat, $("leftFlat"));
  showCanvas("leftFlat", "leftFlatEmpty");

  setStatus("UVDoc 完成。", "現在可以跑 Detector + PCA 中心線。", 100);
}

function boxPoints(box) {
  if (Array.isArray(box.points) && box.points.length >= 4) {
    return box.points.map((p) => ({ x: Number(p.x), y: Number(p.y) }));
  }
  if (Array.isArray(box.polygon) && box.polygon.length >= 4) {
    return box.polygon.map((p) => ({ x: Number(p.x), y: Number(p.y) }));
  }
  return [
    { x: box.x, y: box.y },
    { x: box.x + box.width, y: box.y },
    { x: box.x + box.width, y: box.y + box.height },
    { x: box.x, y: box.y + box.height },
  ];
}

function pcaGeometry(points) {
  const n = points.length;
  if (n < 2) return null;

  let cx = 0, cy = 0;
  for (const p of points) { cx += p.x; cy += p.y; }
  cx /= n; cy /= n;

  let a = 0, b = 0, d = 0;
  for (const p of points) {
    const x = p.x - cx;
    const y = p.y - cy;
    a += x * x;
    b += x * y;
    d += y * y;
  }
  a /= n;
  b /= n;
  d /= n;

  // Principal eigenvector for symmetric 2x2 covariance matrix.
  const trace = a + d;
  const disc = Math.sqrt(Math.max(0, (a - d) * (a - d) + 4 * b * b));
  const lambda1 = (trace + disc) / 2;

  let vx, vy;
  if (Math.abs(b) > 1e-8) {
    vx = lambda1 - d;
    vy = b;
  } else if (a >= d) {
    vx = 1; vy = 0;
  } else {
    vx = 0; vy = 1;
  }

  const norm = Math.hypot(vx, vy) || 1;
  vx /= norm; vy /= norm;
  if (vy < 0) { vx = -vx; vy = -vy; }

  const mx = -vy, my = vx;
  let majorMin = Infinity, majorMax = -Infinity;
  let minorMin = Infinity, minorMax = -Infinity;

  for (const p of points) {
    const dx = p.x - cx;
    const dy = p.y - cy;
    const q1 = dx * vx + dy * vy;
    const q2 = dx * mx + dy * my;
    majorMin = Math.min(majorMin, q1);
    majorMax = Math.max(majorMax, q1);
    minorMin = Math.min(minorMin, q2);
    minorMax = Math.max(minorMax, q2);
  }

  const top = { x: cx + vx * majorMin, y: cy + vy * majorMin };
  const bottom = { x: cx + vx * majorMax, y: cy + vy * majorMax };
  const length = majorMax - majorMin;
  const width = minorMax - minorMin;
  const dy = bottom.y - top.y;
  if (Math.abs(dy) < 3) return null;

  const slope = (bottom.x - top.x) / dy;
  const intercept = top.x - slope * top.y;

  return {
    center: { x: cx, y: cy },
    top,
    bottom,
    length,
    width,
    slope,
    intercept,
    y0: Math.min(top.y, bottom.y),
    y1: Math.max(top.y, bottom.y),
  };
}

function median(values) {
  if (!values.length) return 0;
  const x = [...values].sort((a, b) => a - b);
  const m = Math.floor(x.length / 2);
  return x.length % 2 ? x[m] : (x[m - 1] + x[m]) / 2;
}

function lineX(c, y) {
  return c.slope * y + c.intercept;
}

function makeColumns(boxes, imageHeight) {
  const cols = [];

  for (const box of boxes) {
    const points = boxPoints(box);
    const g = pcaGeometry(points);
    if (!g) continue;

    // Same intent as Colab: keep long vertical-ish text regions,
    // including shorter headings, but reject tiny/non-column shapes.
    if (g.length < g.width * 2) continue;
    if (g.length < imageHeight * 0.025) continue;

    cols.push({ ...g, points, rawBox: box });
  }

  if (!cols.length) return [];

  const stableSlopes = cols
    .map((c) => c.slope)
    .filter((s) => Math.abs(s) < 0.5);

  const pageMedianSlope = stableSlopes.length ? median(stableSlopes) : 0;

  for (const c of cols) {
    if (Math.abs(c.slope - pageMedianSlope) > 0.12) {
      c.slope = pageMedianSlope;
      c.intercept = c.center.x - pageMedianSlope * c.center.y;
    }
    c.xRef = lineX(c, imageHeight * 0.5);
  }

  // Traditional vertical reading order: right to left.
  cols.sort((a, b) => b.xRef - a.xRef);
  return cols;
}

function drawOverlay(image, boxes, cols, canvas) {
  pixelsToCanvas(image, canvas);
  const ctx = canvas.getContext("2d");

  ctx.lineJoin = "round";
  ctx.lineCap = "round";

  // Detector polygons: red.
  ctx.strokeStyle = "rgba(220, 55, 65, .82)";
  ctx.lineWidth = Math.max(1.5, canvas.width / 650);

  for (const box of boxes) {
    const pts = boxPoints(box);
    if (!pts.length) continue;
    ctx.beginPath();
    ctx.moveTo(pts[0].x, pts[0].y);
    for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y);
    ctx.closePath();
    ctx.stroke();
  }

  // PCA centerlines: blue.
  ctx.strokeStyle = "rgba(25, 105, 215, .95)";
  ctx.fillStyle = "rgba(25, 105, 215, .95)";
  ctx.lineWidth = Math.max(2.2, canvas.width / 430);
  ctx.font = `bold ${Math.max(15, canvas.width / 42)}px system-ui`;

  cols.forEach((c, i) => {
    const y0 = Math.max(0, c.y0);
    const y1 = Math.min(canvas.height - 1, c.y1);
    const x0 = lineX(c, y0);
    const x1 = lineX(c, y1);

    ctx.beginPath();
    ctx.moveTo(x0, y0);
    ctx.lineTo(x1, y1);
    ctx.stroke();

    const labelX = Math.max(4, Math.min(canvas.width - 50, x0 + 5));
    const labelY = Math.max(20, y0 + 18);
    ctx.fillText(String(i + 1), labelX, labelY);
  });
}

async function detectPage(image, side) {
  const detector = await ensureDetector();
  const options = detectorRuntimeOptions();

  // UVDoc returns caller-owned pixels: { width, height, data }.
  // Low-level DetectionService preprocessing expects paddleocr.js's Image
  // object because it calls Image.resize(), padding(), tensor(), etc.
  // normalizeInputToRgb() safely converts grayscale/RGB/RGBA pixels into
  // the library's RGB Image object before detection.
  const detectorImage = normalizeInputToRgb(image);

  const boxes = await detector.run(detectorImage, {
    ...options,
    onProgress(event) {
      const stageName = {
        preprocess: "前處理",
        infer: "模型推理",
        postprocess: "DB 後處理",
      }[event.stage] || event.stage;

      setStatus(
        `Detector：${side === "right" ? "右頁" : "左頁"} ${stageName}…`,
        event.detectedCount != null ? `偵測到 ${event.detectedCount} 個區域` : "",
        side === "right" ? 45 : 78,
      );
    },
  });

  const cols = makeColumns(boxes, image.height);
  const canvas = $(side === "right" ? "rightOverlay" : "leftOverlay");
  drawOverlay(image, boxes, cols, canvas);

  showCanvas(
    side === "right" ? "rightOverlay" : "leftOverlay",
    side === "right" ? "rightOverlayEmpty" : "leftOverlayEmpty",
  );

  $(side === "right" ? "rightStats" : "leftStats").textContent =
    `紅框 ${boxes.length} · 藍線 ${cols.length}`;

  return { boxes, cols };
}

async function runDetector() {
  if (!state.rightFlat || !state.leftFlat) {
    await runUvDoc();
  }

  setStatus("Detector：右頁…", "使用 PP-OCRv5_mobile_det", 35);
  state.rightDetection = await detectPage(state.rightFlat, "right");

  setStatus("Detector：左頁…", "使用 PP-OCRv5_mobile_det", 68);
  state.leftDetection = await detectPage(state.leftFlat, "left");

  setStatus(
    "中心線完成。",
    "請看藍線是否大致一欄一條；數字已依右 → 左排序。",
    100,
  );
}


function typicalGap(cols, imageHeight) {
  if (!cols || cols.length < 2) return 45;

  const yRef = imageHeight * 0.5;
  const xs = cols.map((c) => lineX(c, yRef));
  let gaps = [];

  for (let i = 0; i < xs.length - 1; i++) {
    const g = Math.abs(xs[i] - xs[i + 1]);
    if (g > 8) gaps.push(g);
  }

  if (!gaps.length) return 45;

  const med = median(gaps);
  const good = gaps.filter((g) => g > med * 0.45 && g < med * 1.8);
  return good.length ? median(good) : med;
}

function rgbaAt(image, x, y) {
  const { width, height, data } = image;
  x = Math.max(0, Math.min(width - 1, x));
  y = Math.max(0, Math.min(height - 1, y));

  const channels =
    data.length === width * height * 4 ? 4 :
    data.length === width * height * 3 ? 3 : 1;

  const i = (y * width + x) * channels;

  if (channels === 4) {
    return [data[i], data[i + 1], data[i + 2], data[i + 3]];
  }
  if (channels === 3) {
    return [data[i], data[i + 1], data[i + 2], 255];
  }

  const v = data[y * width + x];
  return [v, v, v, 255];
}

function bilinearSample(image, x, y) {
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const x1 = Math.min(image.width - 1, x0 + 1);
  const y1 = Math.min(image.height - 1, y0 + 1);
  const tx = x - x0;
  const ty = y - y0;

  const p00 = rgbaAt(image, x0, y0);
  const p10 = rgbaAt(image, x1, y0);
  const p01 = rgbaAt(image, x0, y1);
  const p11 = rgbaAt(image, x1, y1);

  const out = [0, 0, 0, 255];
  for (let c = 0; c < 4; c++) {
    const a = p00[c] * (1 - tx) + p10[c] * tx;
    const b = p01[c] * (1 - tx) + p11[c] * tx;
    out[c] = Math.max(0, Math.min(255, Math.round(a * (1 - ty) + b * ty)));
  }
  return out;
}

function extractV3Strip(image, cols, index) {
  const H = image.height;
  const W = image.width;
  const c = cols[index];

  let y0 = Math.round(c.y0);
  let y1 = Math.round(c.y1);

  const verticalMargin = Math.round(Math.max(8, c.width * 0.45));
  y0 = Math.max(0, y0 - verticalMargin);
  y1 = Math.min(H - 1, y1 + verticalMargin);

  if (y1 - y0 < 3) return null;

  const gap = typicalGap(cols, H);
  const minAllowed = gap * 0.50;
  const maxAllowed = gap * 1.35;

  const rows = [];
  const rowWidths = [];

  for (let y = y0; y <= y1; y++) {
    const xc = lineX(c, y);

    let rightBoundary;
    if (index > 0) {
      const rightNeighbor = cols[index - 1];
      rightBoundary = (xc + lineX(rightNeighbor, y)) / 2;
    } else {
      rightBoundary = xc + gap * 0.48;
    }

    let leftBoundary;
    if (index < cols.length - 1) {
      const leftNeighbor = cols[index + 1];
      leftBoundary = (xc + lineX(leftNeighbor, y)) / 2;
    } else {
      leftBoundary = xc - gap * 0.48;
    }

    let xl = Math.min(leftBoundary, rightBoundary);
    let xr = Math.max(leftBoundary, rightBoundary);
    const wr = xr - xl;

    rows.push({ y, xc, xl, xr, wr });
    if (Number.isFinite(wr) && wr > 1) rowWidths.push(wr);
  }

  let medianWidth = rowWidths.length ? median(rowWidths) : gap;
  medianWidth = Math.max(minAllowed, Math.min(maxAllowed, medianWidth));

  const outW = Math.max(28, Math.round(medianWidth));
  const outH = rows.length;
  const out = new Uint8Array(outW * outH * 4);

  let di = 0;

  for (let row = 0; row < rows.length; row++) {
    let { y, xc, xl, xr, wr } = rows[row];

    // Same V3 safety clamp as the Colab version:
    // if neighbor geometry creates an absurd corridor on a row,
    // fall back to the page's typical column width centered on xc.
    if (!Number.isFinite(wr) || wr < minAllowed || wr > maxAllowed) {
      xl = xc - medianWidth / 2;
      xr = xc + medianWidth / 2;
    }

    for (let x = 0; x < outW; x++) {
      const t = outW === 1 ? 0.5 : x / (outW - 1);
      const sx = xl + t * (xr - xl);
      const sy = y;

      let rgba;
      if (sx < 0 || sx > W - 1 || sy < 0 || sy > H - 1) {
        rgba = [245, 245, 240, 255];
      } else {
        rgba = bilinearSample(image, sx, sy);
      }

      out[di++] = rgba[0];
      out[di++] = rgba[1];
      out[di++] = rgba[2];
      out[di++] = rgba[3];
    }
  }

  return {
    width: outW,
    height: outH,
    data: out,
    sourceIndex: index,
    y0,
    y1,
    typicalGap: gap,
  };
}

function renderStrips(strips, side) {
  const root = $(side === "right" ? "rightStrips" : "leftStrips");
  root.innerHTML = "";

  if (!strips.length) {
    root.innerHTML = '<div class="empty">沒有可抽出的直欄</div>';
    return;
  }

  strips.forEach((strip, i) => {
    const card = document.createElement("div");
    card.className = "strip-card";

    const title = document.createElement("h4");
    title.textContent = `Column ${String(i + 1).padStart(2, "0")}`;

    const wrap = document.createElement("div");
    wrap.className = "strip-canvas-wrap";

    const canvas = document.createElement("canvas");
    pixelsToCanvas(strip, canvas);

    const meta = document.createElement("div");
    meta.className = "strip-meta";
    meta.textContent = `${strip.width} × ${strip.height}`;

    wrap.appendChild(canvas);
    card.appendChild(title);
    card.appendChild(wrap);
    card.appendChild(meta);
    root.appendChild(card);
  });
}

function extractPageStrips(image, detection, side) {
  if (!detection?.cols?.length) return [];

  const strips = [];

  detection.cols.forEach((_, index) => {
    const strip = extractV3Strip(image, detection.cols, index);
    if (strip) strips.push(strip);
  });

  renderStrips(strips, side);

  $(side === "right" ? "rightStripStats" : "leftStripStats").textContent =
    `${strips.length} 欄`;

  return strips;
}

async function runExtraction() {
  if (!state.rightDetection || !state.leftDetection) {
    await runDetector();
  }

  setStatus("V3：抽出右頁直欄…", "scanline corridor + bilinear remap", 35);
  state.rightStrips = extractPageStrips(
    state.rightFlat,
    state.rightDetection,
    "right",
  );

  setStatus("V3：抽出左頁直欄…", "scanline corridor + bilinear remap", 70);
  state.leftStrips = extractPageStrips(
    state.leftFlat,
    state.leftDetection,
    "left",
  );

  setStatus(
    "V3 抽欄完成。",
    "Column 01 現在會顯示在最右邊；請檢查每欄是否完整。",
    100,
  );
}


async function fetchTextCached(url, resourceName) {
  const cache = "caches" in window ? await caches.open(MODEL_CACHE) : null;

  if (cache) {
    const cached = await cache.match(url);
    if (cached) return cached.text();
  }

  const response = await fetch(url, { mode: "cors" });
  if (!response.ok) {
    throw new Error(`${modelName} 下載失敗：HTTP ${response.status}`);
  }

  const text = await response.text();

  if (cache) {
    await cache.put(
      url,
      new Response(text, {
        headers: { "content-type": "text/plain;charset=utf-8" },
      }),
    );
  }

  return text;
}

function parsePpOcrV5Dictionary(text) {
  // Do NOT trim the beginning of this file: PP-OCRv5's dictionary may
  // contain whitespace-like characters as real dictionary entries.
  const dictionary = text.split(/\r?\n/);

  // A trailing newline creates one synthetic empty record. Remove only that.
  if (dictionary.length && dictionary[dictionary.length - 1] === "") {
    dictionary.pop();
  }

  // Current official ppocrv5_dict.txt is observed as 18,383 entries.
  // Paddle's PP-OCRv5 config has use_space_char=true, so the normal ASCII
  // space belongs at the end of the recognition character list.
  if (dictionary[dictionary.length - 1] !== " ") {
    dictionary.push(" ");
  }

  // Do not hard-code 18,384 as a minimum. RecognitionService validates
  // against the actual ONNX output class count at inference time.
  if (dictionary.length < 1000) {
    throw new Error(
      `PP-OCRv5 字典讀取異常：只讀到 ${dictionary.length} 個項目。`,
    );
  }

  console.info(`PP-OCRv5 dictionary entries: ${dictionary.length}`);
  return dictionary;
}

async function ensureRecognizer(forceWasm = false) {
  const key = $("recModel").value;
  const spec = REC_MODELS[key];

  if (!spec) {
    throw new Error(`未知 recognition 模型：${key}`);
  }

  if (
    state.recognizer &&
    state.recPreset === spec.preset &&
    state.recForcedWasm === forceWasm
  ) {
    return state.recognizer;
  }

  setStatus(
    `準備 ${spec.label}…`,
    forceWasm
      ? "Recognition 已切換為 CPU / WASM fallback。"
      : "Recognition 先嘗試 WebGPU；若實際推理失敗會自動重試 CPU。",
    5,
  );
  await nextPaint();

  const [modelBuffer, dictText] = await Promise.all([
    state.recModelBuffer && state.recSpec?.preset === spec.preset
      ? Promise.resolve(state.recModelBuffer)
      : fetchArrayBufferCached(spec.url, spec.label, 5, 68),
    state.recDictionary
      ? Promise.resolve(null)
      : fetchTextCached(DICT_URL, "PP-OCRv5 字典"),
  ]);

  if (!state.recDictionary) {
    state.recDictionary = parsePpOcrV5Dictionary(dictText);
  }

  let session;
  if (forceWasm) {
    session = await createWasmSession(modelBuffer, spec.label);
  } else {
    session = await createSessionAuto(modelBuffer, spec.label);
  }

  state.recModelBuffer = modelBuffer;
  state.recSpec = spec;
  state.recSession = session;
  state.recPreset = spec.preset;
  state.recForcedWasm = forceWasm;
  state.recognizer = new RecognitionService(ort, session, {
    ...getTextRecognitionPresetOptions(spec.preset),
    charactersDictionary: state.recDictionary,
  });

  setStatus(
    `${spec.label} 已準備完成。`,
    forceWasm ? "Recognition = CPU / WASM fallback" : "",
    90,
  );
  return state.recognizer;
}

function fullVerticalBox(image) {
  // Supplying polygon points is intentional.
  // RecognitionService.cropRotated() automatically rotates a crop
  // counter-clockwise when height/width >= 1.5. For a traditional
  // vertical column this maps original top->bottom to horizontal
  // left->right before PP-OCRv5 recognition.
  return {
    x: 0,
    y: 0,
    width: image.width,
    height: image.height,
    points: [
      { x: 0, y: 0 },
      { x: image.width, y: 0 },
      { x: image.width, y: image.height },
      { x: 0, y: image.height },
    ],
  };
}

function stripChannels(image) {
  const px = image.width * image.height;
  if (image.data.length === px * 4) return 4;
  if (image.data.length === px * 3) return 3;
  return 1;
}

function stripGray(image) {
  const channels = stripChannels(image);
  const out = new Uint8Array(image.width * image.height);

  for (let i = 0; i < out.length; i++) {
    if (channels === 1) {
      out[i] = image.data[i];
    } else {
      const si = i * channels;
      const r = image.data[si];
      const g = image.data[si + 1];
      const b = image.data[si + 2];
      out[i] = Math.max(
        0,
        Math.min(255, Math.round(0.299 * r + 0.587 * g + 0.114 * b)),
      );
    }
  }
  return out;
}

function histogramPercentile(hist, total, fraction) {
  const target = total * fraction;
  let seen = 0;
  for (let i = 0; i < 256; i++) {
    seen += hist[i];
    if (seen >= target) return i;
  }
  return 255;
}

function grayBoxBlur3(gray, width, height) {
  const out = new Float32Array(gray.length);

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let sum = 0;
      let count = 0;

      for (let yy = Math.max(0, y - 1); yy <= Math.min(height - 1, y + 1); yy++) {
        for (let xx = Math.max(0, x - 1); xx <= Math.min(width - 1, x + 1); xx++) {
          sum += gray[yy * width + xx];
          count++;
        }
      }

      out[y * width + x] = sum / count;
    }
  }
  return out;
}

function resizePixelsBilinear(image, scale) {
  if (scale === 1) return image;

  const width = Math.max(1, Math.round(image.width * scale));
  const height = Math.max(1, Math.round(image.height * scale));
  const data = new Uint8Array(width * height * 4);

  let di = 0;
  for (let y = 0; y < height; y++) {
    const sy = height === 1 ? 0 : y * (image.height - 1) / (height - 1);

    for (let x = 0; x < width; x++) {
      const sx = width === 1 ? 0 : x * (image.width - 1) / (width - 1);
      const rgba = bilinearSample(image, sx, sy);

      data[di++] = rgba[0];
      data[di++] = rgba[1];
      data[di++] = rgba[2];
      data[di++] = 255;
    }
  }

  return { width, height, data };
}

function enhanceStripForOcr(strip, mode = "normal") {
  const scale = mode === "strong" ? 3 : 2;
  const enlarged = resizePixelsBilinear(strip, scale);

  const width = enlarged.width;
  const height = enlarged.height;
  const gray = stripGray(enlarged);

  const hist = new Uint32Array(256);
  for (const v of gray) hist[v]++;

  const lowFrac = mode === "strong" ? 0.035 : 0.02;
  const highFrac = mode === "strong" ? 0.965 : 0.985;

  let low = histogramPercentile(hist, gray.length, lowFrac);
  let high = histogramPercentile(hist, gray.length, highFrac);

  if (high - low < 45) {
    low = Math.max(0, low - 20);
    high = Math.min(255, high + 20);
  }

  const stretched = new Uint8Array(gray.length);
  const gamma = mode === "strong" ? 1.10 : 1.04;

  for (let i = 0; i < gray.length; i++) {
    let t = (gray[i] - low) / Math.max(1, high - low);
    t = Math.max(0, Math.min(1, t));
    t = Math.pow(t, gamma);
    stretched[i] = Math.round(t * 255);
  }

  const blur = grayBoxBlur3(stretched, width, height);
  const amount = mode === "strong" ? 1.05 : 0.72;
  const output = new Uint8Array(width * height * 4);

  let di = 0;
  for (let i = 0; i < stretched.length; i++) {
    let v = stretched[i] + amount * (stretched[i] - blur[i]);

    if (v > 205) {
      v = 205 + (v - 205) * 1.35;
    }

    v = Math.max(0, Math.min(255, Math.round(v)));
    output[di++] = v;
    output[di++] = v;
    output[di++] = v;
    output[di++] = 255;
  }

  return { width, height, data: output };
}

function coreTextLength(text) {
  return [...String(text || "")
    .replace(/[\s，。；：、,.!?！？「」『』（）()《》〈〉【】〔〕—…·・：；]/g, "")]
    .length;
}

function estimateShortColumnChars(strip) {
  const ratio = strip.height / Math.max(1, strip.width);
  const estimate = Math.max(1, Math.round(ratio * 0.96));

  if (estimate > 12 || strip.height > 360) return null;
  return estimate;
}

function chooseRecognitionCandidate(candidates, strip) {
  const expectedChars = estimateShortColumnChars(strip);
  const normalizedCounts = new Map();

  for (const candidate of candidates) {
    const key = candidate.text.trim();
    if (key) normalizedCounts.set(key, (normalizedCounts.get(key) || 0) + 1);
  }

  const scored = candidates.map((candidate) => {
    const text = candidate.text.trim();
    const charCount = coreTextLength(text);
    let adjustedScore = Number(candidate.confidence || 0);

    const consensus = normalizedCounts.get(text) || 0;
    if (text && consensus >= 2) {
      adjustedScore += 0.035 * (consensus - 1);
    }

    if (expectedChars !== null && text) {
      const diff = Math.abs(charCount - expectedChars);
      adjustedScore -= Math.min(0.22, diff * 0.06);
      if (diff === 0) adjustedScore += 0.025;
    }

    if (!text) adjustedScore -= 0.30;

    return {
      ...candidate,
      charCount,
      adjustedScore,
    };
  });

  scored.sort((a, b) => b.adjustedScore - a.adjustedScore);

  return {
    ...scored[0],
    expectedChars,
    candidates: scored,
  };
}

async function recognizeStripOnce(strip, recognizer, variantName) {
  const image = normalizeInputToRgb(strip);
  const box = fullVerticalBox(image);

  const results = await recognizer.run(image, [box], {
    ordering: { sortByReadingOrder: false },
  });

  const result = results[0];

  return {
    variant: variantName,
    text: result?.text ?? "",
    confidence: Number(result?.confidence ?? 0),
  };
}

async function recognizeStrip(strip, recognizer) {
  const useEnhancement = $("enhanceOcr")?.checked ?? true;

  if (!useEnhancement) {
    const single = await recognizeStripOnce(strip, recognizer, "原圖");
    return {
      ...single,
      expectedChars: estimateShortColumnChars(strip),
      candidates: [single],
    };
  }

  const candidates = [];

  candidates.push(
    await recognizeStripOnce(strip, recognizer, "原圖"),
  );

  const enhanced = enhanceStripForOcr(strip, "normal");
  candidates.push(
    await recognizeStripOnce(enhanced, recognizer, "增強"),
  );

  if (estimateShortColumnChars(strip) !== null) {
    const strong = enhanceStripForOcr(strip, "strong");
    candidates.push(
      await recognizeStripOnce(strong, recognizer, "增強+"),
    );
  }

  return chooseRecognitionCandidate(candidates, strip);
}

function renderRecognition(items, side) {
  const root = $(side === "right" ? "rightRecognition" : "leftRecognition");
  root.innerHTML = "";

  if (!items.length) {
    root.innerHTML = '<div class="empty">沒有辨識結果</div>';
    return;
  }

  items.forEach((item, index) => {
    const row = document.createElement("div");
    row.className = "recognition-item";

    const name = document.createElement("div");
    name.className = "col-name";
    name.textContent = `Column ${String(index + 1).padStart(2, "0")}`;

    const text = document.createElement("div");
    text.className = "rec-text";
    text.textContent = item.text || "（空白）";

    const confidence = document.createElement("div");
    confidence.className = "confidence";
    confidence.textContent = `${(item.confidence * 100).toFixed(1)}%`;

    row.appendChild(name);
    row.appendChild(text);
    row.appendChild(confidence);

    if (Array.isArray(item.candidates) && item.candidates.length > 1) {
      const info = document.createElement("div");
      info.className = "variant-info";

      const parts = item.candidates.map((candidate) => {
        const picked = candidate.variant === item.variant ? "✓ " : "";
        return `${picked}${candidate.variant} ${(candidate.confidence * 100).toFixed(1)}%`;
      });

      const expected =
        item.expectedChars !== null && item.expectedChars !== undefined
          ? ` · 短欄估計約 ${item.expectedChars} 字`
          : "";

      info.innerHTML =
        `<strong>選 ${item.variant}</strong> · ${parts.join(" / ")}${expected}`;

      info.title = item.candidates
        .map(
          (candidate) =>
            `${candidate.variant}: ${candidate.text} (${(candidate.confidence * 100).toFixed(1)}%)`,
        )
        .join("\n");

      row.appendChild(info);
    }

    root.appendChild(row);
  });
}
async function recognizePage(strips, side, recognizer, startProgress, span) {
  const results = [];

  for (let i = 0; i < strips.length; i++) {
    setStatus(
      `Recognition：${side === "right" ? "右頁" : "左頁"} Column ${i + 1}/${strips.length}`,
      $("enhanceOcr")?.checked ? "PP-OCRv5 A/B recognition in browser" : "PP-OCRv5 recognition in browser",
      startProgress + span * ((i + 1) / Math.max(1, strips.length)),
    );

    const result = await recognizeStrip(strips[i], recognizer);
    results.push(result);
  }

  renderRecognition(results, side);

  const nonEmpty = results.filter((x) => x.text.trim()).length;
  $(side === "right" ? "rightRecStats" : "leftRecStats").textContent =
    `${nonEmpty}/${results.length} 欄有文字`;

  return results;
}

function assembleFullText() {
  // Traditional book reading order:
  // right page first, then left page;
  // strips are already sorted right -> left within each page.
  const right = state.rightRecognition
    .map((x) => x.text.trim())
    .filter(Boolean);

  const left = state.leftRecognition
    .map((x) => x.text.trim())
    .filter(Boolean);

  const pageParts = [];
  if (right.length) pageParts.push(right.join("\n"));
  if (left.length) pageParts.push(left.join("\n"));

  const text = pageParts.join("\n\n");
  $("fullText").value = text;
  $("copyTextBtn").disabled = !text;
  $("downloadTextBtn").disabled = !text;

  return text;
}

async function runRecognition() {
  if (!state.rightStrips.length || !state.leftStrips.length) {
    await runExtraction();
  }

  let recognizer = await ensureRecognizer(false);

  try {
    state.rightRecognition = await recognizePage(
      state.rightStrips,
      "right",
      recognizer,
      10,
      40,
    );

    state.leftRecognition = await recognizePage(
      state.leftStrips,
      "left",
      recognizer,
      52,
      40,
    );
  } catch (gpuError) {
    const currentBackend = state.recSpec
      ? state.backends[state.recSpec.label]
      : undefined;

    if (currentBackend !== "webgpu") {
      throw gpuError;
    }

    console.warn(
      "Recognition WebGPU runtime failed. Rebuilding recognizer on WASM and retrying.",
      gpuError,
    );

    setStatus(
      "Recognition GPU 推理失敗，正在自動改用 CPU / WASM…",
      gpuError instanceof Error
        ? `${gpuError.name}: ${gpuError.message}`
        : String(gpuError),
      5,
    );
    await nextPaint();

    try {
      state.recSession?.release?.();
    } catch {}
    state.recognizer = null;
    state.recSession = null;
    state.recPreset = null;

    recognizer = await ensureRecognizer(true);

    // Retry both pages from the beginning so output stays deterministic.
    state.rightRecognition = await recognizePage(
      state.rightStrips,
      "right",
      recognizer,
      10,
      40,
    );

    state.leftRecognition = await recognizePage(
      state.leftStrips,
      "left",
      recognizer,
      52,
      40,
    );
  }

  const text = assembleFullText();

  setStatus(
    "OCR 辨識完成。",
    text
      ? (
          state.recForcedWasm
            ? "Recognition 使用 CPU / WASM fallback；全文已組合。"
            : "Recognition 使用 WebGPU；全文已組合。"
        )
      : "模型完成推理，但沒有得到文字。",
    100,
  );
}
async function withBusy(fn) {
  const buttons = ["splitBtn", "uvBtn", "detBtn", "extractBtn", "recBtn", "allBtn"];
  buttons.forEach((id) => $(id).disabled = true);

  try {
    await fn();
  } catch (error) {
    console.error("BookOCR processing error:", error);

    const name = error instanceof Error ? error.name : "Error";
    const message = error instanceof Error ? error.message : String(error);
    const stackLine =
      error instanceof Error && error.stack
        ? error.stack.split("\n").slice(1, 3).join(" | ")
        : "";

    setStatus(
      "處理失敗。",
      `${name}: ${message}${stackLine ? ` ｜ ${stackLine}` : ""}`,
      0,
    );
  } finally {
    if (state.bitmap) {
      buttons.forEach((id) => $(id).disabled = false);
    }
  }
}
async function loadFile(file) {
  state.file = file;
  state.bitmap?.close?.();
  state.bitmap = await createImageBitmap(file, { imageOrientation: "from-image" });
  state.fullInput = bitmapToPixels(state.bitmap, 2600);
  state.routedMode = null;
  state.routerAnalysis = null;
  state.generalPages = [];
  state.generalRecognition = [];

  $("routerCard").classList.add("hidden");
  $("generalPages").innerHTML = "";
  $("generalRecognition").innerHTML = '<div class="empty">等待 Auto OCR</div>';
  $("generalStats").textContent = "尚未辨識";

  $("photoPreview").src = URL.createObjectURL(file);
  $("photoPreview").style.display = "block";
  $("dropHint").style.display = "none";

  ["splitBtn", "uvBtn", "detBtn", "extractBtn", "recBtn", "allBtn"].forEach((id) => $(id).disabled = false);

  const selected = $("ocrMode").value;
  if (selected === "traditional") {
    updateModeUi("traditional", null);
    splitPages();
  } else {
    updateModeUi(selected === "auto" ? "general" : selected, null);
    setStatus(
      selected === "auto" ? "圖片已載入，等待 Auto OCR。" : `圖片已載入：${displayModeName(selected)}。`,
      selected === "auto" ? "按「Auto OCR 到文字」即可自動判斷。" : "",
      0,
    );
  }
}

$("fileInput").addEventListener("change", (event) => {
  const file = event.target.files?.[0];
  if (file) loadFile(file);
});

const dropZone = $("dropZone");
["dragenter", "dragover"].forEach((name) => {
  dropZone.addEventListener(name, (event) => {
    event.preventDefault();
    dropZone.classList.add("drag");
  });
});
["dragleave", "drop"].forEach((name) => {
  dropZone.addEventListener(name, (event) => {
    event.preventDefault();
    dropZone.classList.remove("drag");
  });
});
dropZone.addEventListener("drop", (event) => {
  const file = event.dataTransfer?.files?.[0];
  if (file?.type.startsWith("image/")) loadFile(file);
});

$("splitRange").addEventListener("input", () => {
  $("splitValue").textContent = `${Number($("splitRange").value).toFixed(1)}%`;
  if (state.bitmap) splitPages();
});
$("gutterRange").addEventListener("input", () => {
  $("gutterValue").textContent = `${Number($("gutterRange").value).toFixed(1)}%`;
  if (state.bitmap) splitPages();
});

$("splitBtn").addEventListener("click", () => splitPages());
$("uvBtn").addEventListener("click", () => withBusy(runUvDoc));
$("detBtn").addEventListener("click", () => withBusy(runDetector));
$("extractBtn").addEventListener("click", () => withBusy(runExtraction));
$("recBtn").addEventListener("click", () => withBusy(runRecognition));
$("allBtn").addEventListener("click", () => withBusy(async () => {
  if (!state.fullInput) {
    throw new Error("請先上傳圖片。");
  }

  const mode = await resolveRunMode();

  if (mode === "traditional") {
    splitPages();
    await runUvDoc();
    await runDetector();
    await runExtraction();
    await runRecognition();
  } else {
    await runGeneralPipeline(mode);
  }
}));

$("ocrMode").addEventListener("change", () => {
  const selected = $("ocrMode").value;
  state.routedMode = null;
  state.routerAnalysis = null;
  $("routerCard").classList.add("hidden");

  if (selected === "auto") {
    $("readingOrderLabel").textContent = "Auto 判斷中";
    $("readingOrderHint").textContent = "上傳後按 Auto OCR，先分析版面再選流程。";
    $("verticalPipeline").classList.add("hidden");
    $("generalPipeline").classList.add("hidden");
    $("verticalSteps").classList.add("hidden");
  } else {
    updateModeUi(selected, null);
  }

  if (state.bitmap && selected === "traditional") {
    splitPages();
  }

  setStatus(
    selected === "auto"
      ? "已切換 Auto。"
      : `已切換：${displayModeName(selected)}。`,
    "重新按 Auto OCR 到文字即可重跑。",
    0,
  );
});

$("recModel").addEventListener("change", () => {
  state.recognizer = null;
  state.recSession = null;
  state.recPreset = null;
  state.recModelBuffer = null;
  state.recSpec = null;
  state.recForcedWasm = false;

  const key = $("recModel").value;
  setStatus(
    key === "mobile"
      ? "已切換到 mobile recognizer。"
      : "已切換到重型 server recognizer。",
    key === "mobile"
      ? "下一次辨識會使用較輕的 mobile 模型。"
      : "server 模型較大，建立 WebGPU session 可能需要更久。",
    0,
  );
});

$("copyTextBtn").addEventListener("click", async () => {
  const text = $("fullText").value;
  if (!text) return;
  await navigator.clipboard.writeText(text);
  setStatus("文字已複製。", "", 100);
});

$("downloadTextBtn").addEventListener("click", () => {
  const text = $("fullText").value;
  if (!text) return;

  const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "bookocr.txt";
  a.click();
  URL.revokeObjectURL(url);
});

$("clearCacheBtn").addEventListener("click", async () => {
  if ("caches" in window) await caches.delete(MODEL_CACHE);
  state.uvdoc = null;
  state.uvSession = null;
  state.detector = null;
  state.detSession = null;
  state.recognizer = null;
  state.recSession = null;
  state.recPreset = null;
  state.recDictionary = null;
  state.recModelBuffer = null;
  state.recSpec = null;
  state.recForcedWasm = false;
  state.backends = {};
  updateRuntimeBadge();
  setStatus(
    "模型快取已清除。",
    "下一次執行會重新下載 UVDoc、Detector，以及你選的 Recognition 模型。",
    0,
  );
});

$("splitValue").textContent = `${Number($("splitRange").value).toFixed(1)}%`;
$("gutterValue").textContent = `${Number($("gutterRange").value).toFixed(1)}%`;


probeWebGpu().then((available) => {
  updateRuntimeBadge();
  console.info(
    available
      ? "WebGPU adapter detected. GPU will be preferred."
      : "No WebGPU adapter. WASM/CPU fallback will be used.",
  );
});


$("verticalPipeline").classList.add("hidden");
$("generalPipeline").classList.add("hidden");
$("verticalSteps").classList.add("hidden");
$("readingOrderLabel").textContent = "Auto 判斷中";
