/**
 * GLM-Image Studio - Main Logic
 * Integrates T2I, I2I, and I2T (Thinking Mode)
 */

// --- GLOBAL STATE ---
let loraFiles = [];
let timerInterval;
let startTime;
let currentMode = "t2i";
let uploadedImageDims = { w: 0, h: 0 }; // Dimensioni originali immagine caricata
let i2tFullResponse = ""; // Buffer per accumulare il testo I2T

// --- DOM ELEMENTS REFERENCE ---

// Sliders standard
const sliderW = document.getElementById('width');
const sliderH = document.getElementById('height');
const valW = document.getElementById('val-w');
const valH = document.getElementById('val-h');
const sliderSteps = document.getElementById('steps');
const valSteps = document.getElementById('val-steps');
const sliderCfg = document.getElementById('guidance');
const valCfg = document.getElementById('val-cfg');

// Sliders I2T (Nuovi)
const sliderTopK = document.getElementById('topk');
const valTopK = document.getElementById('val-topk');
const sliderTemp = document.getElementById('temp');
const valTemp = document.getElementById('val-temp');

// Sliders I2I Specific
const sliderStrength = document.getElementById('strength');
const valStrength = document.getElementById('val-strength');
const sliderMix = document.getElementById('mix-ratio');
const valMix = document.getElementById('val-mix');
const i2iParamsPanel = document.getElementById('i2i-specific-params');
const colMix = document.getElementById('col-mix');

// Inputs & Buttons
const promptBox = document.getElementById('prompt');
const seedInput = document.getElementById('seed');
const randomizeCheckbox = document.getElementById('randomize');
const btnGenerate = document.getElementById('btn-generate');
const btnStop = document.getElementById('btn-stop');
const btnExit = document.getElementById('btn-exit');
const btnOrig = document.querySelector('.btn-orig'); // Tasto Original Ratio

// LoRA
const loraContainer = document.getElementById('lora-container');
const loraFolderInput = document.getElementById('lora-folder');
const btnScan = document.getElementById('btn-scan');
const btnAddLora = document.getElementById('btn-add-lora');

// Panels
const uploadPanel = document.getElementById('upload-panel');
const paramsPanel = document.getElementById('params-panel');
// const i2tParamsPanel = document.getElementById('i2t-params-panel'); // Removed
// const i2tParamsPanel = document.getElementById('i2t-params-panel'); // Removed

// Upload Elements
// Upload Elements
const uploadZone1 = document.getElementById('upload-zone-1');
const fileInput = document.getElementById('file-input');
const uploadedPathInput = document.getElementById('uploaded-path');
const preview1 = document.getElementById('preview-1');
const btnDel1 = document.getElementById('btn-del-1');

// Upload Elements 2
const slot2Container = document.getElementById('slot-2-container');
const swapContainer = document.getElementById('swap-container');
const uploadZone2 = document.getElementById('upload-zone-2');
const fileInput2 = document.getElementById('file-input-2');
const uploadedPathInput2 = document.getElementById('uploaded-path-2');
const preview2 = document.getElementById('preview-2');
const btnDel2 = document.getElementById('btn-del-2');

// Output Areas
const imgArea = document.getElementById('image-area');
const placeholder = document.querySelector('.placeholder');
const resultImg = document.getElementById('result-img');
const textResultArea = document.getElementById('text-result-area');
const thinkingBox = document.getElementById('thinking-box');
const answerBox = document.getElementById('answer-box');

// System Logs
const statusBar = document.getElementById('status-bar');
const statusText = document.getElementById('status-text');
const timerDisplay = document.getElementById('timer');
const consoleDiv = document.getElementById('console');


// --- INITIALIZATION ---
document.addEventListener('DOMContentLoaded', () => {
    // Setup Slider Listeners (aggiornano il numerino a fianco)
    // --- EVENT LISTENERS (Inputs) ---
    sliderW.oninput = () => valW.innerText = sliderW.value;
    sliderH.oninput = () => valH.innerText = sliderH.value;
    sliderSteps.oninput = () => valSteps.innerText = sliderSteps.value;
    sliderCfg.oninput = () => valCfg.innerText = sliderCfg.value;

    // New Unified Params Listeners
    sliderTopK.oninput = () => valTopK.innerText = sliderTopK.value;
    sliderTemp.oninput = () => valTemp.innerText = sliderTemp.value;

    // I2I Specific listeners
    if (sliderStrength) sliderStrength.oninput = () => valStrength.innerText = sliderStrength.value;
    if (sliderMix) sliderMix.oninput = () => valMix.innerText = sliderMix.value;

    // Imposta default path per LoRA
    if (loraFolderInput) loraFolderInput.value = "/app/loras";

    // Scansione iniziale LoRA (con leggero ritardo per stabilità)
    setTimeout(scanLoras, 500);

    // Aggiungi un paio di slot vuoti
    addLoraSlot();
    addLoraSlot();

    // Imposta stato iniziale UI
    switchTab('t2i', document.querySelector('.nav-btn.active'));
});

// --- HELPER FUNCTIONS ---

function setupSlider(el, display) {
    if (el && display) {
        el.addEventListener('input', () => display.innerText = el.value);
    }
}

function formatDateYYMMDD(timestamp) {
    const d = new Date(timestamp * 1000);
    const pad = (n) => n.toString().padStart(2, '0');
    const yy = d.getFullYear().toString().slice(-2);
    const mm = pad(d.getMonth() + 1);
    const dd = pad(d.getDate());
    const hh = pad(d.getHours());
    const min = pad(d.getMinutes());
    const ss = pad(d.getSeconds());
    return `${yy}-${mm}-${dd} ${hh}:${min}:${ss}`;
}

function log(msg, isError = false, isSuccess = false) {
    const div = document.createElement('div');
    div.className = 'log-line';
    if (isError) div.className += ' log-err';
    if (isSuccess) div.className += ' log-info';
    div.innerText = msg;
    consoleDiv.appendChild(div);
    consoleDiv.scrollTop = consoleDiv.scrollHeight;
}

function startTimer() {
    startTime = Date.now();
    timerDisplay.innerText = "0.0s";
    timerInterval = setInterval(() => {
        timerDisplay.innerText = ((Date.now() - startTime) / 1000).toFixed(1) + "s";
    }, 100);
}

function stopTimer() { clearInterval(timerInterval); }

let currentResultPath = "";

// Separate Prompt Buffers
const promptBuffers = {
    t2i: "",
    i2i: "",
    i2t: ""
};

// Separate Image Buffers
const imageBuffers = {
    t2i: "",
    i2i: "",
    i2t: ""
};

function toggleSystemLog() {
    const drawer = document.getElementById('system-log-drawer');
    const btn = document.getElementById('btn-log-toggle');
    if (!drawer || !btn) return;

    drawer.classList.toggle('collapsed');
    const isCollapsed = drawer.classList.contains('collapsed');

    // Rotate arrow or change icon
    btn.innerText = isCollapsed ? '▼' : '▲';
}

function showImage(url) {
    placeholder.style.display = 'none';
    imgArea.style.display = 'flex';
    // Timestamp per evitare cache del browser
    // Timestamp per evitare cache del browser
    currentResultPath = url; // Save relative path

    // Save to buffer
    if (currentMode && imageBuffers.hasOwnProperty(currentMode)) {
        imageBuffers[currentMode] = url;
    }

    resultImg.src = url + "?t=" + Date.now();
    resultImg.style.display = 'block';

    // Show Action Buttons
}

// --- TAB SWITCHING LOGIC ---

window.switchTab = function (mode, btn) {
    try {
        // Save current prompt to buffer
        if (currentMode && promptBuffers.hasOwnProperty(currentMode)) {
            promptBuffers[currentMode] = promptBox.value;
        }

        currentMode = mode;

        // Restore new prompt from buffer
        promptBox.value = promptBuffers[mode] || "";

        // Restore Image from buffer
        const savedImg = imageBuffers[mode];
        if (savedImg) {
            currentResultPath = savedImg;
            resultImg.src = savedImg + "?t=" + Date.now();
            resultImg.style.display = 'block';
            placeholder.style.display = 'none';
            imgArea.style.display = 'flex';
        } else {
            // No image for this mode -> Show Placeholder
            currentResultPath = "";
            resultImg.style.display = 'none';
            resultImg.src = "";
            placeholder.style.display = 'block';
            imgArea.style.display = 'flex'; // Keep container generic style
        }

        // Gestione classe attiva bottoni
        // Se chiamato da script (senza btn), trova quello giusto
        if (!btn) {
            if (mode === 't2i') btn = document.querySelectorAll('.nav-btn')[0];
            if (mode === 'i2i') btn = document.querySelectorAll('.nav-btn')[1];
            if (mode === 'i2t') btn = document.querySelectorAll('.nav-btn')[2];
        }

        document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
        if (btn) btn.classList.add('active');

        // Reset visualizzazione pannelli
        uploadPanel.classList.add('hidden');
        paramsPanel.classList.add('hidden');
        imgArea.classList.add('hidden');
        textResultArea.classList.add('hidden');
        if (i2iParamsPanel) i2iParamsPanel.classList.add('hidden');

        // Hide all history buttons
        const hTop = document.getElementById('btn-hist-top');
        const hI2I = document.getElementById('btn-hist-i2i');
        const hI2T = document.getElementById('btn-hist-i2t');

        if (hTop) hTop.classList.add('hidden');
        if (hI2I) hI2I.classList.add('hidden');
        if (hI2T) hI2T.classList.add('hidden');
        if (colMix) colMix.classList.add('hidden'); // Reset Mix Slider

        const slot2Container = document.getElementById('slot-2-container');
        const swapContainer = document.getElementById('swap-container');

        if (mode === 't2i') {
            // SETUP T2I
            paramsPanel.classList.remove('hidden');
            imgArea.classList.remove('hidden');

            let btn = document.getElementById('btn-hist-top');
            if (btn) {
                btn.classList.remove('hidden');
                console.log("Shown btn-hist-top");
            } else {
                console.error("btn-hist-top missing");
            }

            btnGenerate.innerText = "🚀 GENERATE IMAGE";
            promptBox.placeholder = "Describe your imagination here...";

            if (btnOrig) {
                btnOrig.style.opacity = "0.3";
                btnOrig.style.cursor = "not-allowed";
            }

        } else if (mode === 'i2i') {
            // SETUP I2I
            uploadPanel.classList.remove('hidden');
            paramsPanel.classList.remove('hidden');
            imgArea.classList.remove('hidden');
            if (hI2I) hI2I.classList.remove('hidden'); // Show I2I Hist Btn

            // Show second upload slot & swap for I2I
            if (slot2Container) slot2Container.classList.remove('hidden');
            if (swapContainer) swapContainer.classList.remove('hidden');
            if (uploadZone2) uploadZone2.style.display = 'flex'; // Ensure flex layout

            if (colMix) colMix.classList.remove('hidden'); // Show Mix Slider
            if (i2iParamsPanel) i2iParamsPanel.classList.remove('hidden');

            btnGenerate.innerText = "🎨 RESTYLE IMAGE";
            promptBox.placeholder = "Describe how to modify the image...";

            if (btnOrig) {
                btnOrig.style.opacity = "1";
                btnOrig.style.cursor = "pointer";
            }

        } else if (mode === 'i2t') {
            // SETUP I2T
            uploadPanel.classList.remove('hidden');
            paramsPanel.classList.remove('hidden');
            textResultArea.classList.remove('hidden');
            imgArea.classList.remove('hidden');
            if (hI2T) hI2T.classList.remove('hidden'); // Show I2T Hist Btn

            // Show second upload slot & swap for I2T (Multi-Image Mode)
            if (slot2Container) slot2Container.classList.remove('hidden');
            if (swapContainer) swapContainer.classList.remove('hidden');
            if (uploadZone2) uploadZone2.style.display = 'flex';

            if (colMix) colMix.classList.remove('hidden'); // Show Mix Slider for I2T too
            if (i2iParamsPanel) i2iParamsPanel.classList.remove('hidden'); // Show Params

            btnGenerate.innerText = "🧠 ANALYZE IMAGE(S)";
            promptBox.placeholder = "Ask a question about the image(s)...";

            if (btnOrig) {
                btnOrig.style.opacity = "1";
                btnOrig.style.cursor = "pointer";
            }
        }

        // Trigger update to sync dynamic elements (like I2I sliders visibility)
        if (typeof updatePreviews === 'function') updatePreviews();

        // Refresh history view for the new mode if it's open (or ready to be opened)
        if (typeof refreshHistoryView === 'function') {
            currentPage = 0;
            refreshHistoryView();
        }
        // Aggiorna visibilità parametri per I2T
        if (typeof updateParamVisibility === 'function') updateParamVisibility();

        // Convalida input
        if (typeof validateInputs === 'function') validateInputs();
    } catch (e) {
        console.error("Error in switchTab:", e);
        if (typeof log === 'function') log("UI Error (SwitchTab): " + e.message, true);
    }
}

// --- CROSS-TAB LOGIC ---
window.sendToI2I = function () {
    if (!currentResultPath) return;
    const fullPath = "/app" + currentResultPath; // Reconstruct server path

    uploadedPathInput.value = fullPath;
    uploadText.innerText = "✅ Generated Image";

    // Create preview
    previewContainer.innerHTML = '';
    const img = document.createElement('img');
    img.className = 'uploaded-preview';
    img.src = currentResultPath;

    img.onload = () => {
        uploadedImageDims = { w: img.naturalWidth, h: img.naturalHeight };
    };

    previewContainer.appendChild(img);
    previewContainer.classList.remove('hidden');

    switchTab('i2i');
    log("Image sent to I2I workspace.");
}

window.sendToI2T = function () {
    if (!currentResultPath) return;
    const fullPath = "/app" + currentResultPath;

    uploadedPathInput.value = fullPath;
    uploadText.innerText = "✅ Generated Image";

    // Create preview
    previewContainer.innerHTML = '';
    const img = document.createElement('img');
    img.className = 'uploaded-preview';
    img.src = currentResultPath;

    img.onload = () => {
        uploadedImageDims = { w: img.naturalWidth, h: img.naturalHeight };
    };

    previewContainer.appendChild(img);
    previewContainer.classList.remove('hidden');

    switchTab('i2t');
    log("Image sent to I2T workspace.");
}

window.sendPromptTo = function (targetMode) {
    const text = answerBox.innerText;
    if (!text) return;
    // WRITES TO TARGET BUFFER DIRECTLY
    if (promptBuffers) promptBuffers[targetMode] = text;
    switchTab(targetMode);
    log(`Analysis result sent to ${targetMode.toUpperCase()}.`);
}

window.sendThinkingTo = function (targetMode) {
    const text = thinkingBox.innerText;
    if (!text) return;
    // WRITES TO TARGET BUFFER DIRECTLY
    if (promptBuffers) promptBuffers[targetMode] = text;
    switchTab(targetMode);
    log(`Thinking process sent to ${targetMode.toUpperCase()}.`);
}


window.toggleThinking = function () {
    if (!thinkingBox) return; // Safety Check
    const isCollapsed = thinkingBox.classList.toggle('collapsed');
    const icon = document.getElementById('thinking-toggle-icon');
    if (icon) icon.innerText = isCollapsed ? "▶" : "▼";
}

window.setRatio = function (wRatio, hRatio) {
    const currentW = parseInt(sliderW.value);
    const newH = Math.round(currentW * (hRatio / wRatio));
    // Clamp between 512 and 2048
    const clampedH = Math.max(512, Math.min(2048, newH));
    sliderH.value = clampedH;
    valH.innerText = clampedH;
}

window.setOriginalRatio = function () {
    // In T2I do nothing
    if (currentMode === 't2i') return;

    if (uploadedImageDims.w > 0) {
        sliderW.value = uploadedImageDims.w;
        valW.innerText = uploadedImageDims.w;
        sliderH.value = uploadedImageDims.h;
        valH.innerText = uploadedImageDims.h;
        log(`Resized to Original: ${uploadedImageDims.w}x${uploadedImageDims.h}`);
    } else {
        log("No image loaded for Original Ratio", true);
    }
}

// --- LORA LOGIC ---

async function scanLoras() {
    // FIXED: Use correct default path
    const folderPath = loraFolderInput ? loraFolderInput.value.trim() : "/app/loras";

    try {
        log(`Scanning LoRAs in: ${folderPath}...`);
        const res = await fetch('/api/scan_loras', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ folder: folderPath })
        });
        const data = await res.json();

        if (data.files && data.files.length > 0) {
            loraFiles = data.files;
            // Update potential existing selects
            document.querySelectorAll('.lora-select').forEach(sel => populateSelect(sel, sel.value));
            log(`✅ Found ${loraFiles.length} LoRAs.`);
        } else {
            log("⚠️ No LoRAs found in folder.", true);
            loraFiles = [];
            document.querySelectorAll('.lora-select').forEach(sel => populateSelect(sel, ""));
        }
    } catch (e) {
        log("❌ Scan Error: " + e, true);
    }
}

function populateSelect(sel, selectedValue = "") {
    sel.innerHTML = '<option value="">(None)</option>';
    loraFiles.forEach(f => {
        const opt = document.createElement('option');
        opt.value = f;
        opt.innerText = f;
        if (f === selectedValue) opt.selected = true;
        sel.appendChild(opt);
    });
}

function addLoraSlot() {
    const row = document.createElement('div');
    row.className = 'lora-row';

    const sel = document.createElement('select');
    sel.className = 'lora-select';
    populateSelect(sel);

    const num = document.createElement('input');
    num.type = 'number';
    num.className = 'lora-num';
    num.step = '0.1';
    num.value = '1.0';

    const btnDel = document.createElement('button');
    btnDel.className = 'lora-del';
    btnDel.innerText = '×';
    btnDel.onclick = () => row.remove();

    row.append(sel, num, btnDel);
    loraContainer.appendChild(row);
}

// Event Listeners LoRA
btnScan.onclick = scanLoras;
btnAddLora.onclick = addLoraSlot;

// --- UPLOAD LOGIC ---

function updatePreviews() {
    try {
        // SLOT 1
        const p1 = uploadedPathInput.value;
        if (p1) {
            let url = p1.startsWith('/app/') ? p1.substring(4) : p1;
            preview1.src = url + "?t=" + Date.now();
            preview1.classList.remove('hidden');
            btnDel1.classList.remove('hidden');
            uploadZone1.classList.add('has-image');

            // Get dims if needed (async)
            preview1.onload = () => uploadedImageDims = { w: preview1.naturalWidth, h: preview1.naturalHeight };
        } else {
            preview1.src = "";
            preview1.classList.add('hidden');
            btnDel1.classList.add('hidden');
            uploadZone1.classList.remove('has-image');
        }

        // SLOT 2
        const p2 = uploadedPathInput2.value;
        if (p2) {
            let url = p2.startsWith('/app/') ? p2.substring(4) : p2;
            preview2.src = url + "?t=" + Date.now();
            preview2.classList.remove('hidden');
            btnDel2.classList.remove('hidden');
            uploadZone2.classList.add('has-image');
        } else {
            preview2.src = "";
            preview2.classList.add('hidden');
            btnDel2.classList.add('hidden');
            uploadZone2.classList.remove('has-image');
        }

        // MODE VISIBILITY
        // In I2I and I2T we want both slots enabled.
        // T2I hides the entire uploadPanel, so we don't need detailed logic here.
        if (currentMode === 't2i') {
            // Just in case
        } else {
            slot2Container.classList.remove('hidden');
            swapContainer.classList.remove('hidden');
        }

        // Toggle I2I specific params visibility
        // Showing for both I2I and I2T now
        if (i2iParamsPanel) {
            if (currentMode === 'i2i' || currentMode === 'i2t') {
                i2iParamsPanel.classList.remove('hidden');
                // Ensure mix slider is visible if we have 2 images?
                // Logic for Mix Slider visibility is usually handled elsewhere or via CSS depending on slot usage?
                // script.js line 209: if (colMix) colMix.classList.add('hidden'); // Reset Mix Slider
                // We need to ensure logic below re-enables it if needed.
            } else {
                i2iParamsPanel.classList.add('hidden');
            }
        }

        if (typeof validateInputs === 'function') validateInputs();
    } catch (e) {
        console.error("Error in updatePreviews:", e);
        if (typeof log === 'function') log("UI Error (Previews): " + e.message, true);
    }
}

window.deleteImage = function (slot) {
    if (slot === 1) {
        // If slot 2 has image, shift it to slot 1
        if (uploadedPathInput2.value) {
            uploadedPathInput.value = uploadedPathInput2.value;
            uploadedPathInput2.value = "";
            log("Image 1 deleted. Image 2 shifted to 1.");
        } else {
            uploadedPathInput.value = "";
            log("Image 1 deleted.");
        }
    } else if (slot === 2) {
        uploadedPathInput2.value = "";
        log("Image 2 deleted.");
    }
    updatePreviews();
}

window.swapUploadedImages = function () {
    const p1 = uploadedPathInput.value;
    const p2 = uploadedPathInput2.value;

    // Swap paths
    uploadedPathInput.value = p2;
    uploadedPathInput2.value = p1;

    // Update previews
    updatePreviews();
    log("Images swapped.");
}



async function handleFileUploadGeneral(file, pathInput) {
    const formData = new FormData();
    formData.append("file", file);
    // Visual feedback handled by updatePreviews or loading state if added

    try {
        const res = await fetch('/api/upload_image', { method: 'POST', body: formData });
        const data = await res.json();

        if (data.path) {
            pathInput.value = data.path;
            log("Uploaded: " + file.name);
            updatePreviews();
        }
    } catch (e) {
        log("Upload failed: " + e, true);
    }
}

// Handler Slot 1
async function handleFileUpload(file) {
    handleFileUploadGeneral(file, uploadedPathInput);
}
// Handler Slot 2
async function handleFileUpload2(file) {
    handleFileUploadGeneral(file, uploadedPathInput2);
}

// Upload Drag & Drop handlers 1
// Upload Drag & Drop handlers 1
// uploadZone1.onclick -> Handled by Browse Button
uploadZone1.ondragover = (e) => { e.preventDefault(); uploadZone1.classList.add('dragover'); };
uploadZone1.ondragleave = () => uploadZone1.classList.remove('dragover');
uploadZone1.ondrop = (e) => {
    e.preventDefault();
    uploadZone1.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) handleFileUpload(e.dataTransfer.files[0]);
};
fileInput.onchange = () => { if (fileInput.files.length > 0) handleFileUpload(fileInput.files[0]); };

// Upload Drag & Drop handlers 2
if (uploadZone2) {
    // uploadZone2.onclick -> Handled by Browse Button
    uploadZone2.ondragover = (e) => { e.preventDefault(); uploadZone2.classList.add('dragover'); };
    uploadZone2.ondragleave = () => uploadZone2.classList.remove('dragover');
    uploadZone2.ondrop = (e) => {
        console.log("Slot 2 Dropped");
        e.preventDefault();
        uploadZone2.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) handleFileUpload2(e.dataTransfer.files[0]);
    };
    fileInput2.onchange = () => {
        console.log("Slot 2 File Selected", fileInput2.files);
        if (fileInput2.files.length > 0) handleFileUpload2(fileInput2.files[0]);
    };
} else {
    console.error("Slot 2 Element not found during init");
}


// --- GENERATION HANDLERS ---

let isProcessing = false;

function setGenerationState(processing) {
    if (!btnGenerate) return; // Safety Check
    isProcessing = processing;

    if (processing) {
        // STOP STATE
        btnGenerate.innerText = "🛑 STOP generation process";
        btnGenerate.title = "Click to Abort";
        btnGenerate.classList.add("btn-stop-active");
        btnGenerate.disabled = false; // Always enabled to allow stop
        statusText.innerText = "Processing...";
    } else {
        // IDLE STATE (Revert)
        // Check mode to restore correct text
        if (currentMode === 'i2t') btnGenerate.innerText = "🔍 ANALYZE IMAGE";
        else if (currentMode === 'i2i') btnGenerate.innerText = "🎨 TRANSFORM IMAGE";
        else btnGenerate.innerText = "🚀 GENERATE IMAGE";

        btnGenerate.classList.remove("btn-stop-active");
        btnGenerate.title = "";
        btnGenerate.disabled = false;

        // Re-validate to ensure disable if empty inputs
        validateInputs();
    }
}

// Main Button Click
btnGenerate.onclick = async () => {
    // IF PROCESSING -> STOP
    if (isProcessing) {
        if (confirm("Abort current process?")) {
            await fetch('/api/stop', { method: 'POST' });
            log("🛑 Abort Signal Sent", true);
            // State will be reset by the DONE/ERR handler or manually here to be safe
        }
        return;
    }

    // NORMAL GENERATION START
    if (currentMode === 'i2t') {
        handleI2T();
    } else {
        handleGeneration(); // T2I e I2I condividono la logica
    }
};

// Parsing output Thinking for I2T
function parseThinking(fullText) {
    let thought = "";
    let answer = fullText;

    // Extract <think>
    const thinkMatch = fullText.match(/<think>([\s\S]*?)<\/think>/);
    if (thinkMatch) {
        thought = thinkMatch[1].trim();
        answer = answer.replace(thinkMatch[0], "").trim();
    }

    // Extract <answer> or clean tags
    const answerMatch = answer.match(/<answer>([\s\S]*?)<\/answer>/);
    if (answerMatch) {
        answer = answerMatch[1].trim();
    } else {
        answer = answer.replace(/<\/?answer>/g, "").trim();
    }

    return { thought, answer };
}

async function handleI2T() {
    setGenerationState(true); // SET STOP MODE
    consoleDiv.innerHTML = ''; // Clear Log

    // Reset UI
    thinkingBox.innerText = "";
    answerBox.innerText = "";
    i2tFullResponse = "";

    startTimer();

    const payload = {
        image_path: uploadedPathInput.value,
        image_path_2: uploadedPathInput2.value || null, // Optional 2nd image
        prompt: promptBox.value,
        top_k: parseInt(sliderTopK.value),
        temperature: parseFloat(sliderTemp.value),
        strength: parseFloat(sliderStrength.value),
        mix_ratio: parseFloat(sliderMix.value)
    };

    try {
        const response = await fetch('/api/analyze', {
            method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload)
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            const chunk = decoder.decode(value);
            const lines = chunk.split('\n\n');
            lines.forEach(line => {
                if (line.startsWith('data: ')) {
                    const content = line.substring(6);
                    if (content.startsWith('TXT|')) {
                        // I2T text stream
                        i2tFullResponse += content.substring(4) + "\n";
                        const parsed = parseThinking(i2tFullResponse);
                        thinkingBox.innerText = parsed.thought;
                        answerBox.innerText = parsed.answer;

                        // Auto-scroll
                        thinkingBox.scrollTop = thinkingBox.scrollHeight;
                        answerBox.scrollTop = answerBox.scrollHeight;

                    } else if (content.startsWith('LOG|')) {
                        log(content.substring(4));
                        statusText.innerText = "Processing...";
                    } else if (content.startsWith('DONE|')) {
                        stopTimer();
                        log(content.substring(5), false, true);
                        statusText.innerText = "✅ Analysis Complete";
                        setGenerationState(false); // RESET
                    } else if (content.startsWith('ERR|')) {
                        stopTimer();
                        log(content.substring(4), true);
                        statusText.innerText = "❌ Error";
                        setGenerationState(false); // RESET
                    }
                }
            });
        }
    } catch (e) {
        stopTimer();
        log("Connection Error: " + e, true);
        setGenerationState(false); // RESET
    }
}

async function handleGeneration() {
    setGenerationState(true); // SET STOP MODE
    consoleDiv.innerHTML = ''; // Clear Log
    startTimer();

    // Collect Selected LoRAs
    const loras = [];
    document.querySelectorAll('.lora-row').forEach(row => {
        const sel = row.querySelector('select');
        const num = row.querySelector('input');
        if (sel.value) {
            loras.push({
                folder: loraFolderInput.value, // Use valid folder
                filename: sel.value,
                strength: parseFloat(num.value)
            });
        }
    });

    const payload = {
        mode: currentMode,
        init_image: uploadedPathInput.value,
        init_image_2: uploadedPathInput2.value ? uploadedPathInput2.value : null, // SECOND IMAGE
        prompt: promptBox.value,
        width: parseInt(sliderW.value),
        height: parseInt(sliderH.value),
        steps: parseInt(sliderSteps.value),
        guidance: parseFloat(sliderCfg.value),
        seed: parseInt(seedInput.value),
        randomize: randomizeCheckbox.checked,
        loras: loras,
        top_k: parseFloat(sliderTopK.value), // Unified
        temperature: parseFloat(sliderTemp.value), // Unified
        strength: sliderStrength ? parseFloat(sliderStrength.value) : 0.75,
        mix_ratio: sliderMix ? parseFloat(sliderMix.value) : 0.5
    };

    try {
        const response = await fetch('/api/generate', {
            method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload)
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            const chunk = decoder.decode(value);
            const lines = chunk.split('\n\n');
            lines.forEach(line => {
                if (line.startsWith('data: ')) {
                    const content = line.substring(6);
                    if (content.startsWith('LOG|')) {
                        log(content.substring(4));
                        statusText.innerText = content.substring(4);
                    } else if (content.startsWith('IMG|')) {
                        showImage(content.substring(4));
                    } else if (content.startsWith('DONE|')) {
                        stopTimer();
                        log(content.substring(5), false, true);
                        statusText.innerText = "✅ Complete";
                        setGenerationState(false); // RESET
                        // Refresh history
                        setTimeout(loadHistory, 1000);
                    } else if (content.startsWith('ERR|')) {
                        stopTimer();
                        log(content.substring(4), true);
                        statusText.innerText = "❌ Error";
                        setGenerationState(false); // RESET
                    }
                }
            });
        }
    } catch (e) {
        stopTimer();
        log("Connection Error: " + e, true);
        setGenerationState(false); // RESET
    }
}


// --- HISTORY LOGIC ---
const historyPanel = document.getElementById('history-panel');
const historyList = document.getElementById('history-list');
const historyToggleBtn = document.getElementById('history-toggle-btn');

let lastHistoryData = []; // Cache to avoid re-fetching on size change

window.toggleHistory = function () {
    if (historyPanel.classList.contains('collapsed')) {
        historyPanel.classList.remove('collapsed');
        if (historyToggleBtn) historyToggleBtn.style.display = 'none';
        loadHistory();
    } else {
        historyPanel.classList.add('collapsed');
        if (historyToggleBtn) historyToggleBtn.style.display = 'flex';
    }
}

async function loadHistory() {
    try {
        const res = await fetch('/api/history');
        const data = await res.json();
        lastHistoryData = data.history || []; // Store in cache
        refreshHistoryView(); // Trigger render
    } catch (e) {
        console.error("Failed to load history", e);
    }
}

function refreshHistoryView() {
    // Prevent caching with timestamp
    fetch('/api/history?t=' + Date.now())
        .then(res => res.json())
        .then(data => {
            try {
                console.log("History Loaded:", data.history);
                lastHistoryData = data.history || [];
                if (typeof renderHistoryRobust === 'function') {
                    renderHistoryRobust(lastHistoryData);
                } else if (typeof renderHistoryNew === 'function') {
                    renderHistoryNew(lastHistoryData); // Use New
                } else {
                    renderHistory(lastHistoryData);
                }
            } catch (e) {
                console.error("Error rendering history:", e);
                if (typeof log === 'function') log("UI Error (History): " + e.message, true);
            }
        })
        .catch(err => {
            console.error(err);
            console.error("Failed to load history", err);
        });
}

function renderHistory(items) {
    historyList.innerHTML = '';
    items.forEach(item => {
        const div = document.createElement('div');
        div.className = 'history-item';

        const dateStr = new Date(item.timestamp * 1000).toLocaleString();
        const params = item.params || {};
        const mode = params.mode ? params.mode.toUpperCase() : "UNK";

        div.innerHTML = `
            <img src="${item.image}" loading="lazy" alt="${item.filename}">
            <div class="history-info">
                <span class="mode-badge">${mode}</span>
                <span style="float:right; font-size:10px">${dateStr.split(',')[1]}</span>
            </div>
        `;

        div.onclick = () => restoreHistoryState(item);
        historyList.appendChild(div);
    });
}

function restoreHistoryState(item) {
    const p = item.params;
    console.log("Restoring history item:", item); // DEBUG
    if (!p) {
        console.warn("No parameters found in history item");
        statusText.innerText = "⚠️ No parameters in history file";
        return;
    }

    statusText.innerText = "♻️ Restoring parameters...";

    // Restore Prompt
    if (p.prompt) promptBox.value = p.prompt;

    // Restore Sliders
    if (p.width) { sliderW.value = p.width; valW.innerText = p.width; }
    if (p.height) { sliderH.value = p.height; valH.innerText = p.height; }
    if (p.steps) { sliderSteps.value = p.steps; valSteps.innerText = p.steps; }

    // Restore Source Images (I2I / I2T)
    if (p.source_image_1) {
        uploadedPathInput.value = p.source_image_1;
    }
    if (p.source_image_2 && uploadedPathInput2) {
        uploadedPathInput2.value = p.source_image_2;
    }

    // Refresh Previews
    if (typeof updatePreviews === 'function') updatePreviews();

    // Restore I2T Text Fields (Thinking & Answer)
    if (currentMode === 'i2t') {
        const out = item.output || {};
        // Support both nested V2 structure and potential flat structure
        const source = out.text_content || out;

        if (source.thinking_process) {
            thinkingBox.innerText = source.thinking_process;
        } else {
            thinkingBox.innerText = "";
        }

        if (source.final_answer) {
            answerBox.innerText = source.final_answer;
        } else {
            answerBox.innerText = "";
        }
    }

    // Check key naming (guidance vs guidance_scale)
    const cfg = p.guidance || p.guidance_scale;
    if (cfg) { sliderCfg.value = cfg; valCfg.innerText = cfg; }

    if (p.seed) { seedInput.value = p.seed; randomizeCheckbox.checked = false; }
    if (p.top_k) { sliderTopK.value = p.top_k; valTopK.innerText = p.top_k; }
    if (p.temperature) { sliderTemp.value = p.temperature; valTemp.innerText = p.temperature; }


    // Persist restored prompt to buffer
    promptBuffers[currentMode] = promptBox.value;

    // Show Image
    showImage(item.image);

    // Log success
    log(`Restored settings from: ${item.filename}`);
    console.log("Parameters applied:", p);
}



// --- SYSTEM CONTROLS ---

// Button Stop Removed in favor of Unified Button

btnExit.onclick = async () => {
    if (confirm("Shutdown server?")) {
        try {
            await fetch('/api/exit', { method: 'POST' });
        } catch (e) {
            console.log("Server shutdown initiated (Network disconnected)");
        }
        document.body.innerHTML = "<h1 style='color:white;text-align:center;margin-top:20%'>System Shutdown.</h1>";
    }
};

function renderHistoryNew(items) {
    historyList.innerHTML = '';

    // Filter items:
    // 1. Must have params & mode
    // 2. Mode must match currentMode
    const validItems = items.filter(item => {
        const p = item.params;
        if (!p || Object.keys(p).length === 0) return false;

        let itemMode = p.mode ? p.mode.toLowerCase() : 'unk';
        return itemMode === currentMode;
    });

    if (validItems.length === 0) {
        historyList.innerHTML = '<div style="padding:15px; color:#888; text-align:center; font-size:12px;">No history for this mode.</div>';
        return;
    }

    validItems.forEach(item => {
        const div = document.createElement('div');
        div.className = 'history-item';

        const dateStr = new Date(item.timestamp * 1000).toLocaleString();
        const mode = item.params.mode.toUpperCase();
        const promptTxt = item.params.prompt || "No Prompt";

        div.innerHTML = `
            <div class="history-thumb">
                <img src="${item.image}" loading="lazy" alt="${item.filename}">
            </div>
            <div class="history-info">
                <span class="mode-badge" style="background:#555; padding:2px 5px; border-radius:3px; font-size:9px;">${mode}</span>
                <div style="font-size:11px; margin-top:4px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; max-width:140px; color:#ccc;" title="${promptTxt}">
                    ${promptTxt}
                </div>
                <span style="font-size:9px; color:#666; margin-top:4px;">${dateStr.split(',')[1]}</span>
            </div>
        `;

        div.onclick = () => restoreHistoryState(item);
        historyList.appendChild(div);
    });
}

// --- PAGINATION STATE ---
let currentPage = 0;
let pageSize = 8; // Default for 128px
let historyIconSize = 128;
let showAllHistory = false; // New State

function toggleHistoryFilter() {
    showAllHistory = !showAllHistory;
    const btn = document.getElementById('btn-history-filter');
    if (btn) {
        btn.style.background = showAllHistory ? "var(--accent)" : "";
        btn.style.color = showAllHistory ? "white" : "";
    }
    currentPage = 0;
    refreshHistoryView();
}

function setHistorySize(size) {
    historyIconSize = size;
    // Update Page Size based on icon size
    if (historyIconSize === 128) pageSize = 10;
    else pageSize = 20;

    currentPage = 0; // Reset to first page
    refreshHistoryView();
}

function changePage(delta) {
    const totalPages = Math.ceil(lastValidItemsCount / pageSize);
    const newPage = currentPage + delta;

    if (newPage >= 0 && newPage < totalPages) {
        currentPage = newPage;
        refreshHistoryView();
    }
}

// Global to track total items for pagination limits
let lastValidItemsCount = 0;

function renderHistoryRobust(items) {
    historyList.innerHTML = '';
    historyList.scrollTop = 0;

    // 0. SORT DESCENDING (Newest First)
    items.sort((a, b) => b.timestamp - a.timestamp);

    // 1. FILTER
    const validItems = items.filter(item => {
        const p = item.params;
        if (!p || Object.keys(p).length === 0) return false;

        // Show All Logic
        if (showAllHistory) return true;

        const itemMode = (p.params && p.params.mode ? p.params.mode : (p.mode || 'unk')).toString().toUpperCase();
        const curMode = (currentMode || 'UNK').toString().toUpperCase();

        // DEBUG: Log ignored items to help debugging
        if (itemMode !== curMode) {
            // console.log(`[Filter] Hidden: ${itemMode} != ${curMode} for ${item.filename}`);
        }

        return itemMode === curMode;
    });

    console.log(`[History] Filtered for ${currentMode.toUpperCase()}: ${validItems.length} items found.`);
    if (items.length > 0) {
        // console.log("Sample Item 0:", items[0]);
    }

    // FIX: Update global count for pagination
    lastValidItemsCount = validItems.length;

    if (validItems.length === 0) {
        historyList.innerHTML = '<div style="padding:20px; color:#999; text-align:center; font-size:12px;">No history found.</div>';
        // Disable buttons if empty
        if (document.getElementById('btn-page-prev')) document.getElementById('btn-page-prev').disabled = true;
        if (document.getElementById('btn-page-next')) document.getElementById('btn-page-next').disabled = true;
        return;
    }

    // 2. PAGINATE
    const totalPages = Math.ceil(validItems.length / pageSize);
    if (currentPage >= totalPages) currentPage = totalPages - 1;
    if (currentPage < 0) currentPage = 0;

    // Update Buttons State
    const btnPrev = document.getElementById('btn-page-prev');
    const btnNext = document.getElementById('btn-page-next');
    if (btnPrev) {
        btnPrev.disabled = (currentPage <= 0);
        btnPrev.style.opacity = (currentPage <= 0) ? "0.3" : "1";
        btnPrev.style.cursor = (currentPage <= 0) ? "default" : "pointer";
    }
    if (btnNext) {
        btnNext.disabled = (currentPage >= totalPages - 1);
        btnNext.style.opacity = (currentPage >= totalPages - 1) ? "0.3" : "1";
        btnNext.style.cursor = (currentPage >= totalPages - 1) ? "default" : "pointer";
    }

    const startIdx = currentPage * pageSize;
    const endIdx = startIdx + pageSize;
    const pageItems = validItems.slice(startIdx, endIdx);

    // 3. RENDER PAGE
    pageItems.forEach(item => {
        const div = document.createElement('div');
        constsizeClass = historyIconSize;
        div.className = `history-item size-${historyIconSize}`;

        const dateStr = new Date(item.timestamp * 1000).toLocaleString();
        const itemMode = (item.params.mode || "UNK").toUpperCase();
        const curModeUC = currentMode.toUpperCase();
        const isForeign = itemMode !== curModeUC;
        const promptTxt = item.params.prompt || "No Prompt";

        // Visual cue for foreign items
        if (isForeign) {
            div.style.opacity = "0.7";
            div.style.border = "1px dashed #444";
        }

        div.innerHTML = `
            <div class="history-thumb">
                <img src="${item.image}" loading="lazy" alt="${item.filename}">
            </div>
            <div class="history-info" style="position:relative;">
                <span class="mode-badge" style="background:${isForeign ? '#777' : '#444'}; padding:2px 5px; border-radius:3px; font-size:9px;">${itemMode}</span>
                <span style="display:block; font-size:10px; color:#fff; font-weight:bold; margin-top:5px;">${formatDateYYMMDD(item.timestamp)}</span>
            </div>
        `;

        // ACTIONS
        const actionsDiv = document.createElement('div');
        actionsDiv.style.cssText = "display: flex; flex-direction: column; gap: 8px; margin-top: 6px;";

        if (isForeign) {
            // LIMITED ACTIONS FOR FOREIGN ITEMS
            // Allow loading to input if supported (see below)

            // Prevent main interaction (Parameter Restore)
            div.onclick = (e) => { e.stopPropagation(); statusText.innerText = "ℹ️ Foreign item: Reference only."; };

        } else {
            // STANDARD ACTIONS
            // Add Delete button only for Standard items
            const delRow = document.createElement('div');
            const btnDel = document.createElement('button');
            btnDel.innerText = "🗑️";
            btnDel.title = "Delete this image";
            btnDel.style.cssText = "background: #500; border: none; border-radius: 3px; cursor: pointer; color: white; padding: 2px 5px; font-size: 12px; line-height:1;";
            btnDel.onclick = (e) => {
                e.stopPropagation();
                deleteHistoryItem(item.filename);
            };
            delRow.appendChild(btnDel);
            actionsDiv.appendChild(delRow);

            div.onclick = () => restoreHistoryState(item);
        }

        // UNIVERSAL MOVE BUTTONS (For I2I/I2T modes)
        // Applies to BOTH Standard and Foreign items
        if (historyIconSize === 128 && currentMode !== 't2i') {
            const moveRow = document.createElement('div');
            moveRow.style.cssText = "display: flex; gap: 5px; flex-wrap: wrap; margin-bottom: 4px;"; // Added margin for spacing

            const btn1 = document.createElement('button');
            btn1.innerText = "➜ 1";
            btn1.title = "Load into Input Image 1";
            btn1.style.cssText = "background: #242; border: none; border-radius: 3px; cursor: pointer; color: white; padding: 2px 6px; font-size: 10px; font-weight:bold;";
            btn1.onclick = (e) => {
                e.stopPropagation();
                loadToInput(item.image, 1);
            };
            moveRow.appendChild(btn1);

            const btn2 = document.createElement('button');
            btn2.innerText = "➜ 2";
            btn2.title = "Load into Input Image 2";
            btn2.style.cssText = "background: #242; border: none; border-radius: 3px; cursor: pointer; color: white; padding: 2px 6px; font-size: 10px; font-weight:bold;";
            btn2.onclick = (e) => {
                e.stopPropagation();
                loadToInput(item.image, 2);
            };
            moveRow.appendChild(btn2);

            // Prepend moveRow
            if (actionsDiv.firstChild) {
                actionsDiv.insertBefore(moveRow, actionsDiv.firstChild);
            } else {
                actionsDiv.appendChild(moveRow);
            }

            // Add [All] button in a separate row if source_image_2 exists
            if (item.params && item.params.source_image_2) {
                const allRow = document.createElement('div');
                allRow.style.cssText = "display: flex; margin-bottom: 4px;";

                const btnAll = document.createElement('button');
                btnAll.innerText = "All";
                btnAll.title = "Load Both Source Images";
                btnAll.style.cssText = "background: #226; border: none; border-radius: 3px; cursor: pointer; color: white; padding: 4px 6px; font-size: 10px; font-weight:bold; width: 100%;";
                btnAll.onclick = (e) => {
                    e.stopPropagation();
                    if (item.params.source_image_1) loadToInput(item.params.source_image_1, 1);
                    loadToInput(item.params.source_image_2, 2);
                };
                allRow.appendChild(btnAll);

                // Insert after moveRow
                if (moveRow.nextSibling) {
                    actionsDiv.insertBefore(allRow, moveRow.nextSibling);
                } else {
                    actionsDiv.appendChild(allRow);
                }
            }
        }

        div.querySelector('.history-info').appendChild(actionsDiv);
        historyList.appendChild(div);
    });

    // 4. ADD PAGE INDICATOR/CONTROLS AT BOTTOM
    const controlsDiv = document.createElement('div');
    controlsDiv.style.padding = "10px";
    controlsDiv.style.textAlign = "center";
    controlsDiv.style.borderTop = "1px solid #333";
    controlsDiv.style.marginTop = "auto";
    controlsDiv.innerHTML = `
        <span style="font-size:11px; color:#888;">Page ${currentPage + 1} / ${totalPages}</span>
        <div style="font-size:10px; color:#555;">(${validItems.length} total)</div>
    `;
    historyList.appendChild(controlsDiv);
}

// --- HELPER FUNCTIONS FOR HISTORY ACTIONS ---
function deleteHistoryItem(filename) {
    if (!confirm(`Are you sure you want to delete ${filename}?`)) return;

    fetch('/api/delete_history', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename: filename })
    })
        .then(r => r.json())
        .then(data => {
            if (data.status === 'deleted') {
                // Remove from local cache immediately for speed or reload
                loadHistory(); // Reload to refresh list
            } else {
                alert("Error deleting: " + data.error);
            }
        })
        .catch(e => console.error(e));
}

function loadToInput(webPath, slot) {
    // webPath is like "/outputs/foo.png"
    // Backend expects absolute path: "/app/outputs/foo.png"

    if (webPath.startsWith("http") || webPath.startsWith("//")) {
        alert("Cannot load external or placeholder image as input.");
        return;
    }

    let serverPath = webPath;
    if (!webPath.startsWith('/app')) {
        serverPath = "/app" + webPath;
    }

    // Determine which input slot
    if (slot === 1) {
        if (typeof uploadedPathInput !== 'undefined') {
            uploadedPathInput.value = serverPath;
        }
    } else if (slot === 2) {
        if (typeof uploadedPathInput2 !== 'undefined') {
            uploadedPathInput2.value = serverPath;
        }
    }

    // Trigger update logic
    if (typeof updatePreviews === 'function') updatePreviews();

    // Provide visual feedback
    const btn = event.target;
    const oldText = btn.innerText;
    btn.innerText = "✓";
    setTimeout(() => btn.innerText = oldText, 1000);
}


function toggleHistorySize() {
    // Toggle logic: 128 -> 64 -> 128
    historyIconSize = (historyIconSize === 128) ? 64 : 128;
    setHistorySize(historyIconSize);

    // Update button icon/text if needed (optional)
    const btn = document.getElementById('btn-size-toggle');
    if (btn) btn.innerText = (historyIconSize === 128) ? "🔽" : "🔼";
}

function updateParamVisibility() {
    const isI2T = (currentMode === 'i2t');

    // IDs to disable/enable
    const ids = ['width', 'height', 'steps', 'guidance', 'seed', 'randomize'];
    ids.forEach(id => {
        const el = document.getElementById(id);
        if (el) {
            el.disabled = isI2T;
            el.style.opacity = isI2T ? "0.5" : "1";
            el.style.cursor = isI2T ? "not-allowed" : "pointer";
            // Also dim labels if possible? Maybe too complex for generic ID
        }
    });

    // Handle LoRA Section (Class)
    const loraSection = document.querySelector('.lora-section');
    if (loraSection) {
        if (isI2T) {
            loraSection.style.pointerEvents = "none";
            loraSection.style.opacity = "0.4";
            loraSection.style.filter = "grayscale(100%)";
        } else {
            loraSection.style.pointerEvents = "auto";
            loraSection.style.opacity = "1";
            loraSection.style.filter = "none";
        }
    }

    // Handle Aspect Ratio Buttons
    const ratioBtns = document.querySelectorAll('.ratios button');
    ratioBtns.forEach(btn => {
        btn.disabled = isI2T;
        btn.style.opacity = isI2T ? "0.5" : "1";
        btn.style.cursor = isI2T ? "not-allowed" : "pointer";
    });
}

function validateInputs() {
    let isValid = false;

    if (currentMode === 't2i') {
        // T2I requires prompt
        isValid = promptBox.value.trim().length > 0;
    } else if (currentMode === 'i2i' || currentMode === 'i2t') {
        // I2I/I2T requires at least one image
        isValid = !!uploadedPathInput.value;
    }

    if (isValid) {
        btnGenerate.disabled = false;
        btnGenerate.style.opacity = "1";
        btnGenerate.style.cursor = "pointer";
        btnGenerate.title = "";
    } else {
        btnGenerate.disabled = true;
        btnGenerate.style.opacity = "0.5";
        btnGenerate.style.cursor = "not-allowed";
        btnGenerate.title = (currentMode === 't2i') ? "Please enter a prompt" : "Please upload an image";
    }
}

// Attach listener to prompt
promptBox.addEventListener('input', validateInputs);

// Verify on load
// setTimeout to ensure elements are loaded
setTimeout(validateInputs, 500);
