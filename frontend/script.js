// ============================================================
// Configuration
// ============================================================
const API_BASE = "https://neuroscan-backend-oqcj.onrender.com"; // Flask backend

// ============================================================
// Element refs
// ============================================================
const fileInput = document.getElementById('fileInput');
const dropZone = document.getElementById('dropZone');
const uploadWrap = document.getElementById('uploadWrap');
const stageWrap = document.getElementById('stageWrap');
const mriImg = document.getElementById('mriImg');
const overlayCanvas = document.getElementById('overlayCanvas');
const scanFrame = document.getElementById('scanFrame');
const runBtn = document.getElementById('runBtn');
const resetBtn = document.getElementById('resetBtn');
const imgMeta = document.getElementById('img-meta');
const scanStatus = document.getElementById('scanStatus');
const scanRes = document.getElementById('scanRes');
const scanlineFx = document.getElementById('scanlineFx');
const layerToggles = document.getElementById('layerToggles');
const serverStatus = document.getElementById('serverStatus');
const downloadReportBtn = document.getElementById('downloadReportBtn');
const historyToggleBtn = document.getElementById('historyToggleBtn');
const historyPanel = document.getElementById('historyPanel');
const historyEmpty = document.getElementById('historyEmpty');
const historyList = document.getElementById('historyList');
const historyCount = document.getElementById('historyCount');

const dotUpload = document.getElementById('dot-upload');
const dotCnn = document.getElementById('dot-cnn');
const dotUnet = document.getElementById('dot-unet');
const dotRisk = document.getElementById('dot-risk');
const dotExplain = document.getElementById('dot-explain');

const cnnEmpty = document.getElementById('cnnEmpty');
const cnnResults = document.getElementById('cnnResults');
const unetEmpty = document.getElementById('unetEmpty');
const unetResults = document.getElementById('unetResults');
const riskEmpty = document.getElementById('riskEmpty');
const riskBadge = document.getElementById('riskBadge');

const explainPanel = document.getElementById('explainPanel');
const explainEmpty = document.getElementById('explainEmpty');
const explainBody = document.getElementById('explainBody');
const chatLog = document.getElementById('chatLog');
const chatInput = document.getElementById('chatInput');
const chatSendBtn = document.getElementById('chatSendBtn');

// ============================================================
// State
// ============================================================
let currentFile = null;
let lastResult = null;        // raw JSON from /predict
let chatHistory = [];         // [{role, content}, ...] excluding the first explanation
let maskImage = null;         // decoded mask <img> element for canvas drawing
let layerState = { mask: false, bbox: false };

let chatSending = false;


// ============================================================
// Backend health check
// ============================================================
async function checkServer() {
  try {
    const res = await fetch(`${API_BASE}/health`, { method: 'GET' });
    const data = await res.json();
    if (data.models_ready) {
      serverStatus.textContent = 'backend connected · models ready';
      serverStatus.className = 'server-status ok';
    } else {
      serverStatus.textContent = 'backend connected · models not trained yet';
      serverStatus.className = 'server-status down';
    }
  } catch (e) {
    serverStatus.textContent = 'backend not reachable — start app.py';
    serverStatus.className = 'server-status down';
  }
}
checkServer();

// ============================================================
// Upload handling
// ============================================================
fileInput.addEventListener('change', (e) => {
  if (e.target.files && e.target.files[0]) loadImage(e.target.files[0]);
});
dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.style.borderColor = 'var(--cyan)'; });
dropZone.addEventListener('dragleave', () => { dropZone.style.borderColor = ''; });
dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.style.borderColor = '';
  if (e.dataTransfer.files && e.dataTransfer.files[0]) loadImage(e.dataTransfer.files[0]);
});

function loadImage(file) {
  currentFile = file;
  const url = URL.createObjectURL(file);
  mriImg.src = url;
  mriImg.onload = () => {
    uploadWrap.classList.add('hidden');
    stageWrap.classList.remove('hidden');
    imgMeta.textContent = file.name.length > 22 ? file.name.slice(0, 19) + '...' : file.name;
    scanRes.textContent = mriImg.naturalWidth + '×' + mriImg.naturalHeight;
    sizeCanvas();
    dotUpload.classList.add('done');
    resetAnalysis();
  };
}

function sizeCanvas() {
  const r = scanFrame.getBoundingClientRect();
  overlayCanvas.width = r.width;
  overlayCanvas.height = r.height;
}
window.addEventListener('resize', () => {
  if (!stageWrap.classList.contains('hidden')) {
    sizeCanvas();
    if (lastResult) drawOverlay();
  }
});

resetBtn.addEventListener('click', () => {
  stageWrap.classList.add('hidden');
  uploadWrap.classList.remove('hidden');
  fileInput.value = '';
  currentFile = null;
  dotUpload.classList.remove('done');
  resetAnalysis();
});

function resetAnalysis() {
  lastResult = null;
  maskImage = null;
  chatHistory = [];
  layerState = { mask: false, bbox: false };

  [...layerToggles.querySelectorAll('.layer-chip')].forEach(c => {
    if (c.dataset.layer !== 'mri') c.classList.remove('on');
  });

  dotCnn.classList.remove('active', 'done');
  dotUnet.classList.remove('active', 'done');
  dotRisk.classList.remove('active', 'done');
  dotExplain.classList.remove('active', 'done');

  cnnResults.classList.add('hidden'); cnnEmpty.classList.remove('hidden');
  unetResults.classList.add('hidden'); unetEmpty.classList.remove('hidden');
  riskBadge.classList.add('hidden'); riskEmpty.classList.remove('hidden');

  explainPanel.classList.add('hidden');
  explainBody.classList.add('hidden');
  explainEmpty.classList.remove('hidden');
  chatLog.innerHTML = '';

  downloadReportBtn.disabled = true;

  scanStatus.textContent = 'Ready for analysis';
  const ctx = overlayCanvas.getContext('2d');
  ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
}

// ============================================================
// Layer toggles
// ============================================================
layerToggles.addEventListener('click', (e) => {
  const chip = e.target.closest('.layer-chip');
  if (!chip || chip.dataset.layer === 'mri' || !lastResult) return;
  chip.classList.toggle('on');
  layerState[chip.dataset.layer] = chip.classList.contains('on');
  drawOverlay();
});

// ============================================================
// Run pipeline (calls real backend)
// ============================================================
runBtn.addEventListener('click', runPipeline);

async function runPipeline() {
  if (!currentFile) return;

  runBtn.disabled = true;
  runBtn.classList.add('running');
  runBtn.textContent = 'Running…';
  scanlineFx.classList.remove('hidden');

  dotCnn.classList.add('active');
  dotUnet.classList.add('active');
  scanStatus.textContent = 'Sending scan to backend (CNN + U-Net)…';

  const formData = new FormData();
  formData.append('image', currentFile);

  try {
    const res = await fetch(`${API_BASE}/predict`, { method: 'POST', body: formData });
    const data = await res.json();

    if (!res.ok) {
      scanStatus.textContent = 'Error: ' + (data.error || 'prediction failed');
      resetRunButton();
      return;
    }

    lastResult = data;
    downloadReportBtn.disabled = !data.scan_id;

    // Stage 1: classification
    renderClassification(data.classification);
    dotCnn.classList.remove('active'); dotCnn.classList.add('done');

    // Stage 2: segmentation
    await renderSegmentation(data.segmentation);
    layerState.mask = true; layerState.bbox = true;
    layerToggles.querySelector('[data-layer="mask"]').classList.add('on');
    layerToggles.querySelector('[data-layer="bbox"]').classList.add('on');
    drawOverlay();
    dotUnet.classList.remove('active'); dotUnet.classList.add('done');

    // Stage 3: risk
    renderRisk(data.risk);
    dotRisk.classList.add('active');
    await new Promise(r => setTimeout(r, 300));
    dotRisk.classList.remove('active'); dotRisk.classList.add('done');

    scanStatus.textContent = 'Analysis complete';
    explainPanel.classList.remove('hidden');

    // Automatically kick off the Mistral explanation
    await requestExplanation();

  } catch (err) {
    scanStatus.textContent = 'Could not reach backend — is app.py running?';
  }

  scanlineFx.classList.add('hidden');
  resetRunButton();
}

function resetRunButton() {
  runBtn.disabled = false;
  runBtn.classList.remove('running');
  runBtn.textContent = 'Re-run pipeline';
}

// ============================================================
// Render: classification
// ============================================================
function renderClassification(items) {
  cnnEmpty.classList.add('hidden');
  cnnResults.classList.remove('hidden');
  cnnResults.innerHTML = '';

  items.forEach((it, i) => {
    const row = document.createElement('div');
    row.className = 'cls-item' + (i === 0 ? ' top' : '');
    row.innerHTML = `
      <div class="lbl"><span class="name">${it.name}</span><span class="pct">${(it.prob * 100).toFixed(1)}%</span></div>
      <div class="bar-track"><div class="bar-fill" style="width:0%"></div></div>`;
    cnnResults.appendChild(row);
    requestAnimationFrame(() => { row.querySelector('.bar-fill').style.width = (it.prob * 100) + '%'; });
  });
}

// ============================================================
// Render: segmentation
// ============================================================
function renderSegmentation(seg) {
  return new Promise((resolve) => {
    unetEmpty.classList.add('hidden');
    unetResults.classList.remove('hidden');

    if (!seg.present) {
      document.getElementById('stat-region').textContent = 'None';
      document.getElementById('stat-area').textContent = '0 mm²';
      document.getElementById('stat-diam').textContent = '—';
      document.getElementById('stat-maskconf').textContent = '—';
      maskImage = null;
      resolve();
      return;
    }

    document.getElementById('stat-region').textContent = seg.region;
    document.getElementById('stat-area').textContent = seg.area_mm2 + ' mm²';
    document.getElementById('stat-diam').textContent = seg.diam_mm + ' mm';
    document.getElementById('stat-maskconf').textContent = (seg.mask_confidence * 100).toFixed(1) + '%';

    if (seg.mask_base64) {
      const img = new Image();
      img.onload = () => { maskImage = img; resolve(); };
      img.src = 'data:image/png;base64,' + seg.mask_base64;
    } else {
      maskImage = null;
      resolve();
    }
  });
}

// ============================================================
// Canvas overlay drawing
// ============================================================
function drawOverlay() {
  const ctx = overlayCanvas.getContext('2d');
  ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
  if (!lastResult || !lastResult.segmentation || !lastResult.segmentation.present) return;

  const seg = lastResult.segmentation;
  const canvasSize = seg.mask_canvas_size || 256;
  const scaleX = overlayCanvas.width / canvasSize;
  const scaleY = overlayCanvas.height / canvasSize;

  if (layerState.mask && maskImage) {
    // Draw mask as a tinted cyan overlay using an offscreen canvas
    const off = document.createElement('canvas');
    off.width = canvasSize; off.height = canvasSize;
    const offCtx = off.getContext('2d');
    offCtx.drawImage(maskImage, 0, 0, canvasSize, canvasSize);
    const imgData = offCtx.getImageData(0, 0, canvasSize, canvasSize);
    for (let i = 0; i < imgData.data.length; i += 4) {
      const v = imgData.data[i]; // grayscale mask value
      if (v > 127) {
        imgData.data[i] = 79; imgData.data[i + 1] = 209; imgData.data[i + 2] = 197; imgData.data[i + 3] = 130;
      } else {
        imgData.data[i + 3] = 0;
      }
    }
    offCtx.putImageData(imgData, 0, 0);
    ctx.drawImage(off, 0, 0, overlayCanvas.width, overlayCanvas.height);
  }

  if (layerState.bbox && seg.bbox) {
    const { x, y, w, h } = seg.bbox;
    ctx.save();
    ctx.strokeStyle = 'rgba(240,168,96,0.85)';
    ctx.lineWidth = 1.3;
    ctx.setLineDash([5, 4]);
    ctx.strokeRect(x * scaleX, y * scaleY, w * scaleX, h * scaleY);
    ctx.restore();
    ctx.font = '11px IBM Plex Mono, monospace';
    ctx.fillStyle = 'rgba(240,168,96,0.9)';
    ctx.fillText('tumor region', x * scaleX, Math.max(12, y * scaleY - 6));
  }
}

// ============================================================
// Render: risk
// ============================================================
function renderRisk(risk) {
  riskEmpty.classList.add('hidden');
  riskBadge.classList.remove('hidden');

  const icons = {
    low: '<path d="M5 13l4 4L19 7" stroke-linecap="round" stroke-linejoin="round"/>',
    moderate: '<path d="M12 9v4M12 17h.01" stroke-linecap="round"/><path d="M10.3 4.9L2.9 18a1.5 1.5 0 001.3 2.2h15.6a1.5 1.5 0 001.3-2.2L13.7 4.9a1.5 1.5 0 00-2.6 0z" stroke-linejoin="round"/>',
    elevated: '<path d="M12 8v5M12 17h.01" stroke-linecap="round"/><circle cx="12" cy="12" r="9"/>'
  };
  const band = risk.band || 'low';

  riskBadge.className = 'risk-badge ' + band;
  riskBadge.innerHTML = `
    <div class="ricon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">${icons[band]}</svg></div>
    <div class="rtext">
      <div class="rk">Risk band</div>
      <div class="rv">${risk.label}</div>
      <div class="rsub">${risk.explanation}</div>
    </div>`;
}

// ============================================================
// Mistral: explanation + follow-up chat
// ============================================================
async function requestExplanation() {
  dotExplain.classList.add('active');
  explainEmpty.classList.add('hidden');
  explainBody.classList.remove('hidden');

  appendChatMessage(
    'assistant',
    'Reading the result and preparing an explanation…',
    true
  );

  try {
    console.log('Calling /explain...');
    console.log('API:', `${API_BASE}/explain`);
    console.log('Result:', lastResult);

    const res = await fetch(`${API_BASE}/explain`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        result: lastResult
      })
    });

    console.log('Explain HTTP status:', res.status);

    const text = await res.text();

    console.log('Explain raw response:', text);

    removeLoadingMessage();

    let data;

    try {
      data = JSON.parse(text);
    } catch (parseError) {
      throw new Error(
        `Backend returned invalid JSON. HTTP ${res.status}: ${text}`
      );
    }

    if (!res.ok) {
      throw new Error(
        data.message ||
        data.error ||
        `HTTP ${res.status}`
      );
    }

    if (data.error) {
      appendChatMessage(
        'assistant',
        'Could not generate explanation: ' +
        (data.message || 'Unknown error')
      );
    } else {
      appendMarkdownMessage('assistant', data.message);

      chatHistory.push({
        role: 'assistant',
        content: data.message
      });
    }

  } catch (err) {

    removeLoadingMessage();

    console.error('EXPLAIN ERROR:', err);

    appendChatMessage(
      'assistant',
      'Explanation error: ' + err.message
    );
  }

  dotExplain.classList.remove('active');
  dotExplain.classList.add('done');
}

// ============================================================
// Chat Enter key handler
// ============================================================

chatInput.addEventListener('keydown', function (e) {

    if (e.key !== 'Enter') {
        return;
    }

    // Stop browser default action
    e.preventDefault();

    // Stop other keyboard handlers
    e.stopImmediatePropagation();

    // Ignore repeated Enter key
    if (e.repeat) {
        return;
    }

    // Don't send while another request is running
    if (chatSending) {
        return;
    }

    // Send message
    sendFollowUp();

}, true);

chatSendBtn.addEventListener('click', function (e) {
    e.preventDefault();
    e.stopPropagation();

    if (chatSending) return;

    sendFollowUp();
});

async function sendFollowUp() {

    // Prevent multiple requests
    if (chatSending) return;

    const question = chatInput.value.trim();

    // Nothing to send
    if (!question || !lastResult) return;

    // Lock chat immediately
    chatSending = true;
    chatSendBtn.disabled = true;

    // Show user message
    appendChatMessage('user', question);

    chatHistory.push({
        role: 'user',
        content: question
    });

    // Clear input
    chatInput.value = '';

    // Show loading message
    appendChatMessage(
        'assistant',
        'Thinking…',
        true
    );

    try {

        console.log('Calling /chat...');
        console.log('API:', `${API_BASE}/chat`);
        console.log('Question:', question);
        console.log('History:', chatHistory.slice(0, -1));
        console.log('Result:', lastResult);

        const res = await fetch(`${API_BASE}/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                result: lastResult,
                history: chatHistory.slice(0, -1),
                question: question
            })
        });

        console.log('Chat HTTP status:', res.status);

        const text = await res.text();

        console.log('Chat raw response:', text);

        removeLoadingMessage();

        let data;

        try {
            data = JSON.parse(text);
        } catch (parseError) {
            throw new Error(
                `Backend returned invalid JSON. HTTP ${res.status}: ${text}`
            );
        }

        if (!res.ok) {
            throw new Error(
                data.message ||
                data.error ||
                `HTTP ${res.status}`
            );
        }

        if (data.error) {

            appendChatMessage(
                'assistant',
                'Could not get a response: ' +
                (data.message || 'Unknown error')
            );

        } else {

            appendMarkdownMessage(
                'assistant',
                data.message
            );

            chatHistory.push({
                role: 'assistant',
                content: data.message
            });
        }

    } catch (err) {

        removeLoadingMessage();

        console.error('CHAT ERROR:', err);

        appendChatMessage(
            'assistant',
            'Chat error: ' + err.message
        );

    } finally {

        chatSending = false;
        chatSendBtn.disabled = false;

        // Return focus to input
        chatInput.focus();
    }
}

function appendChatMessage(role, text, loading = false) {
  const el = document.createElement('div');
  el.className = 'chat-msg ' + role + (loading ? ' loading' : '');
  if (loading) el.dataset.loading = 'true';
  el.innerHTML = `<span class="role-label">${role === 'user' ? 'You' : 'Mistral'}</span>${escapeHtml(text)}`;
  chatLog.appendChild(el);
  chatLog.scrollTop = chatLog.scrollHeight;
}
function appendMarkdownMessage(role, text) {
  const el = document.createElement('div');

  el.className = 'chat-msg ' + role;

  const roleLabel = document.createElement('span');
  roleLabel.className = 'role-label';
  roleLabel.textContent = role === 'user' ? 'You' : 'Mistral';

  const content = document.createElement('div');
  content.className = 'markdown-content';

  if (typeof marked !== 'undefined') {
    content.innerHTML = marked.parse(text);
  } else {
    content.textContent = text;
  }

  el.appendChild(roleLabel);
  el.appendChild(content);

  chatLog.appendChild(el);
  chatLog.scrollTop = chatLog.scrollHeight;
}

function removeLoadingMessage() {
  const loadingEl = chatLog.querySelector('[data-loading="true"]');
  if (loadingEl) loadingEl.remove();
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}
function renderExplanationInChat(explanationText) {
  const log = document.getElementById("chatLog");

  log.innerHTML = `
    <div class="chat-msg assistant">
      ${marked.parse(explanationText)}
    </div>
  `;

  log.scrollTop = log.scrollHeight;
}

// ============================================================
// Download PDF report
// ============================================================
downloadReportBtn.addEventListener('click', async () => {
  if (!lastResult || !lastResult.scan_id) return;

  const originalText = downloadReportBtn.innerHTML;
  downloadReportBtn.disabled = true;
  downloadReportBtn.textContent = 'Generating report…';

  try {
    const res = await fetch(`${API_BASE}/report/${lastResult.scan_id}`);
    if (!res.ok) {
      throw new Error('Report generation failed');
    }
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `NeuroScan_Report_${lastResult.scan_id}.pdf`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  } catch (err) {
    alert('Could not generate the PDF report. Check that the backend is running.');
  }

  downloadReportBtn.disabled = false;
  downloadReportBtn.innerHTML = originalText;
});

// ============================================================
// Scan history panel
// ============================================================
let historyLoaded = false;

historyToggleBtn.addEventListener('click', async () => {
  historyPanel.classList.toggle('hidden');
  if (!historyPanel.classList.contains('hidden') && !historyLoaded) {
    await loadHistory();
  }
});

async function loadHistory() {
  try {
    const res = await fetch(`${API_BASE}/history`);
    const data = await res.json();
    historyLoaded = true;

    const scans = data.scans || [];
    historyCount.textContent = scans.length ? `${scans.length} scan${scans.length === 1 ? '' : 's'}` : '';

    if (scans.length === 0) {
      historyEmpty.classList.remove('hidden');
      historyList.classList.add('hidden');
      return;
    }

    historyEmpty.classList.add('hidden');
    historyList.classList.remove('hidden');
    historyList.innerHTML = '';

    scans.forEach(scan => {
      const item = document.createElement('div');
      item.className = 'history-item';
      const date = new Date(scan.created_at * 1000);
      const dateStr = date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
      const riskBand = scan.risk_band || 'unknown';

      item.innerHTML = `
        <div class="hi-left">
          <span class="hi-class">${scan.top_class || 'Unknown'}</span>
          <span class="hi-date">${dateStr}</span>
        </div>
        <span class="hi-risk ${riskBand}">${riskBand}</span>
      `;
      item.addEventListener('click', () => loadPastScan(scan.id));
      historyList.appendChild(item);
    });
  } catch (err) {
    historyEmpty.textContent = 'Could not reach the backend to load history.';
    historyEmpty.classList.remove('hidden');
    historyList.classList.add('hidden');
  }
}

async function loadPastScan(scanId) {
  try {
    const res = await fetch(`${API_BASE}/history/${scanId}`);
    const data = await res.json();
    if (!res.ok) {
      alert('Could not load this scan.');
      return;
    }

    // Load the stored original image into the scan viewer
    uploadWrap.classList.add('hidden');
    stageWrap.classList.remove('hidden');
    mriImg.src = 'data:image/png;base64,' + data.original_image_b64;

    await new Promise(resolve => { mriImg.onload = resolve; });
    sizeCanvas();
    imgMeta.textContent = `Scan ${scanId}`;
    scanRes.textContent = mriImg.naturalWidth + '×' + mriImg.naturalHeight;
    dotUpload.classList.add('done');

    resetAnalysis();

    lastResult = data.result;
    lastResult.scan_id = scanId;
    downloadReportBtn.disabled = false;

    renderClassification(data.result.classification);
    dotCnn.classList.add('done');

    await renderSegmentation(data.result.segmentation);
    layerState.mask = true; layerState.bbox = true;
    layerToggles.querySelector('[data-layer="mask"]').classList.add('on');
    layerToggles.querySelector('[data-layer="bbox"]').classList.add('on');
    drawOverlay();
    dotUnet.classList.add('done');

    renderRisk(data.result.risk);
    dotRisk.classList.add('done');

    scanStatus.textContent = 'Loaded from history';

    // Replay the saved chat history instead of calling /explain again
    explainPanel.classList.remove('hidden');
    explainBody.classList.remove('hidden');
    explainEmpty.classList.add('hidden');
    chatLog.innerHTML = '';
    chatHistory = [];

    (data.chat_history || []).forEach(msg => {
      appendChatMessage(msg.role, msg.content);
      chatHistory.push({ role: msg.role, content: msg.content });
    });
    dotExplain.classList.add('done');

    historyPanel.classList.add('hidden');
  } catch (err) {
    alert('Could not load this scan from history.');
  }
}

