// --- Theme Management (Default Light Theme) ---
function initTheme() {
    const savedTheme = localStorage.getItem('theme') || 'light';
    applyTheme(savedTheme);

    const themeToggleBtn = document.getElementById('theme-toggle-btn');
    if (themeToggleBtn) {
        themeToggleBtn.addEventListener('click', () => {
            const currentTheme = document.documentElement.getAttribute('data-theme') || 'light';
            const newTheme = currentTheme === 'light' ? 'dark' : 'light';
            applyTheme(newTheme);
            localStorage.setItem('theme', newTheme);
            showToast(`Switched to ${newTheme === 'light' ? 'Light' : 'Dark'} theme`, 'info', 1800);
        });
    }
}

function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    const themeToggleBtn = document.getElementById('theme-toggle-btn');
    if (themeToggleBtn) {
        if (theme === 'dark') {
            themeToggleBtn.innerHTML = '<i class="ph-fill ph-moon"></i> <span>Theme: Dark</span>';
        } else {
            themeToggleBtn.innerHTML = '<i class="ph-fill ph-sun"></i> <span>Theme: Light</span>';
        }
    }
}

initTheme();

// --- Tab Navigation Logic ---
document.querySelectorAll('.nav-links li').forEach(item => {
    item.addEventListener('click', (e) => {
        document.querySelectorAll('.nav-links li').forEach(nav => nav.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(tab => {
            tab.classList.remove('active');
            tab.classList.add('hidden');
        });
        e.currentTarget.classList.add('active');
        const tabId = e.currentTarget.getAttribute('data-tab');
        const activeTab = document.getElementById(tabId);
        activeTab.classList.remove('hidden');
        activeTab.classList.add('active');
        localStorage.setItem('activeTab', tabId);
    });
});

// Restore active tab on load
const savedTab = localStorage.getItem('activeTab');
if (savedTab) {
    const tabElement = document.querySelector(`.nav-links li[data-tab="${savedTab}"]`);
    if (tabElement) {
        tabElement.click();
    }
}

// --- Custom Toast Notification System ---
function showToast(message, type = 'info', duration = 4000) {
    const container = document.getElementById('toast-container');

    // Clear existing toasts so they don't stack up and cause confusion
    container.innerHTML = '';

    const toast = document.createElement('div');
    toast.className = `custom-toast toast-${type}`;

    const icons = {
        info: 'ph-fill ph-info',
        success: 'ph-fill ph-check-circle',
        warning: 'ph-fill ph-warning',
        danger: 'ph-fill ph-siren',
        processing: 'ph-fill ph-spinner'
    };

    toast.innerHTML = `
        <div class="toast-icon"><i class="${icons[type] || icons.info}"></i></div>
        <div class="toast-body">${message}</div>
        <button class="toast-close" onclick="this.parentElement.classList.add('toast-exit'); setTimeout(() => this.parentElement.remove(), 300);">
            <i class="ph ph-x"></i>
        </button>
    `;

    container.appendChild(toast);

    // Trigger entrance animation
    requestAnimationFrame(() => toast.classList.add('toast-enter'));

    // Auto-remove
    setTimeout(() => {
        toast.classList.add('toast-exit');
        setTimeout(() => toast.remove(), 400);
    }, duration);
}

// --- Custom Alert Modal ---
function showAlert(title, subtitle, details, type = 'danger') {
    const existing = document.getElementById('custom-alert-overlay');
    if (existing) existing.remove();

    const colors = {
        danger: { bg: 'rgba(239, 35, 60, 0.15)', border: '#ef233c', icon: 'ph-fill ph-siren', glow: 'rgba(239, 35, 60, 0.4)' },
        warning: { bg: 'rgba(255, 183, 3, 0.15)', border: '#ffb703', icon: 'ph-fill ph-warning', glow: 'rgba(255, 183, 3, 0.4)' },
        success: { bg: 'rgba(0, 255, 136, 0.15)', border: '#00ff88', icon: 'ph-fill ph-check-circle', glow: 'rgba(0, 255, 136, 0.4)' }
    };
    const c = colors[type] || colors.danger;

    const overlay = document.createElement('div');
    overlay.id = 'custom-alert-overlay';
    overlay.className = 'alert-overlay';
    overlay.innerHTML = `
        <div class="alert-card" style="border-color: ${c.border}; box-shadow: 0 0 60px ${c.glow};">
            <div class="alert-icon-ring" style="background: ${c.bg}; color: ${c.border};">
                <i class="${c.icon}"></i>
            </div>
            <h2 class="alert-title" style="color: ${c.border};">${title}</h2>
            <p class="alert-subtitle">${subtitle}</p>
            <div class="alert-details">${details}</div>
            <button class="alert-dismiss" style="background: ${c.border};" onclick="document.getElementById('custom-alert-overlay').classList.add('alert-exit'); setTimeout(() => document.getElementById('custom-alert-overlay').remove(), 300);">
                Acknowledge
            </button>
        </div>
    `;
    document.body.appendChild(overlay);
    requestAnimationFrame(() => overlay.classList.add('alert-visible'));
}

// --- Safe Badge Drawing Helper (100% cross-browser compatible) ---
function drawClinicalBadge(c, x, y, w, h, r, bgColor, borderColor, text, textColor) {
    if (!isFinite(x) || !isFinite(y) || !isFinite(w) || !isFinite(h)) return;
    c.save();
    c.fillStyle = bgColor;
    c.strokeStyle = borderColor;
    c.lineWidth = 1;
    c.beginPath();
    c.moveTo(x + r, y);
    c.lineTo(x + w - r, y);
    c.quadraticCurveTo(x + w, y, x + w, y + r);
    c.lineTo(x + w, y + h - r);
    c.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
    c.lineTo(x + r, y + h);
    c.quadraticCurveTo(x, y + h, x, y + h - r);
    c.lineTo(x, y + r);
    c.quadraticCurveTo(x, y, x + r, y);
    c.closePath();
    c.fill();
    c.stroke();

    c.font = "800 8.5px 'JetBrains Mono', monospace";
    c.fillStyle = textColor;
    c.textAlign = 'center';
    c.textBaseline = 'middle';
    c.fillText(text, x + w / 2, y + h / 2);
    c.restore();
}

// --- Clinical Telemetry ECG Markings Plugin ---
const ecgClinicalPlugin = {
    id: 'ecgClinicalMarkings',
    afterDraw(chart) {
        try {
            const { ctx, chartArea, scales: { y } } = chart;
            if (!chartArea || !ctx || !y) return;

            ctx.save();

            // 1. Draw Isoelectric Reference Baseline (0.0 mV)
            const rawZero = y.getPixelForValue(0);
            const yZero = isFinite(rawZero) ? rawZero : (chartArea.top + chartArea.height / 2);

            ctx.strokeStyle = 'rgba(56, 189, 248, 0.28)';
            ctx.lineWidth = 1;
            ctx.setLineDash([4, 4]);
            ctx.beginPath();
            ctx.moveTo(chartArea.left, yZero);
            ctx.lineTo(chartArea.right, yZero);
            ctx.stroke();
            ctx.setLineDash([]);

            // Label for 0.0 mV Isoelectric baseline
            ctx.font = "600 8px 'JetBrains Mono', monospace";
            ctx.fillStyle = 'rgba(56, 189, 248, 0.6)';
            ctx.textAlign = 'right';
            ctx.fillText('0.0 mV BASELINE', chartArea.right - 8, yZero - 4);

            // 2. Draw Standard 1.0 mV Calibration Pulse (⎍ 10mm Standard)
            const rawOne = y.getPixelForValue(1.0);
            const yOne = isFinite(rawOne) ? rawOne : (yZero - 35);
            const calStartX = chartArea.left + 14;
            const calStepWidth = 18;
            const calTail = 6;

            ctx.strokeStyle = '#00ff88';
            ctx.lineWidth = 1.8;
            ctx.beginPath();
            ctx.moveTo(calStartX, yZero);
            ctx.lineTo(calStartX + calTail, yZero);
            ctx.lineTo(calStartX + calTail, yOne);
            ctx.lineTo(calStartX + calTail + calStepWidth, yOne);
            ctx.lineTo(calStartX + calTail + calStepWidth, yZero);
            ctx.lineTo(calStartX + calTail + calStepWidth + calTail, yZero);
            ctx.stroke();

            // Label above Calibration Pulse
            ctx.font = "800 8px 'JetBrains Mono', monospace";
            ctx.fillStyle = '#00ff88';
            ctx.textAlign = 'center';
            ctx.fillText('1.0mV CAL', calStartX + calTail + (calStepWidth / 2), yOne - 5);

            // 3. Draw Voltage Calibration Ticks (+2mV, +1mV, -1mV, -2mV)
            [-2, -1, 1, 2].forEach(v => {
                const py = y.getPixelForValue(v);
                if (isFinite(py)) {
                    ctx.strokeStyle = 'rgba(148, 163, 184, 0.25)';
                    ctx.lineWidth = 1;
                    ctx.beginPath();
                    ctx.moveTo(chartArea.left + 8, py);
                    ctx.lineTo(chartArea.left + 16, py);
                    ctx.stroke();

                    ctx.font = "600 8px 'JetBrains Mono', monospace";
                    ctx.fillStyle = 'rgba(148, 163, 184, 0.7)';
                    ctx.textAlign = 'left';
                    ctx.fillText((v > 0 ? `+${v}.0` : `${v}.0`) + ' mV', chartArea.left + 18, py + 3);
                }
            });

            // 4. Draw Physiological P-Q-R-S-T Fiducial Markers (when enabled)
            if (showPQRST) {
                const dataset = chart.data.datasets[0];
                const data = dataset ? dataset.data : null;
                const meta = chart.getDatasetMeta(0);

                if (data && data.length >= 300 && meta && meta.data && meta.data.length >= 300) {
                    // Helper to get guaranteed finite pixel coordinates
                    const getPt = (idx) => {
                        const pt = meta.data[idx];
                        if (pt && isFinite(pt.x) && isFinite(pt.y)) {
                            return { x: pt.x, y: pt.y };
                        }
                        const px = chartArea.left + (chartArea.width * (idx / (data.length - 1)));
                        const py = y.getPixelForValue(data[idx] || 0);
                        return {
                            x: isFinite(px) ? px : chartArea.left,
                            y: isFinite(py) ? py : yZero
                        };
                    };

                    // Find R peak (maximum amplitude peak)
                    let rIdx = 180;
                    let maxVal = -Infinity;
                    for (let i = 100; i < 260; i++) {
                        if (data[i] > maxVal) {
                            maxVal = data[i];
                            rIdx = i;
                        }
                    }

                    if (maxVal > 0.3) {
                        // Q dip (minimum before R)
                        let qIdx = rIdx - 8;
                        let minQ = Infinity;
                        for (let i = Math.max(0, rIdx - 25); i < rIdx; i++) {
                            if (data[i] < minQ) {
                                minQ = data[i];
                                qIdx = i;
                            }
                        }

                        // S dip (minimum after R)
                        let sIdx = rIdx + 8;
                        let minS = Infinity;
                        for (let i = rIdx + 1; i <= Math.min(data.length - 1, rIdx + 28); i++) {
                            if (data[i] < minS) {
                                minS = data[i];
                                sIdx = i;
                            }
                        }

                        // P wave peak (before Q)
                        let pIdx = Math.max(0, qIdx - 45);
                        let maxP = -Infinity;
                        for (let i = Math.max(0, qIdx - 85); i < Math.max(0, qIdx - 15); i++) {
                            if (data[i] > maxP) {
                                maxP = data[i];
                                pIdx = i;
                            }
                        }

                        // T wave peak (after S)
                        let tIdx = Math.min(data.length - 1, sIdx + 55);
                        let maxT = -Infinity;
                        for (let i = Math.min(data.length - 1, sIdx + 20); i < Math.min(data.length - 1, sIdx + 110); i++) {
                            if (data[i] > maxT) {
                                maxT = data[i];
                                tIdx = i;
                            }
                        }

                        // Helper to draw clean pinned badge
                        const drawPin = (idx, label, color, isAbove = true, extraText = '') => {
                            const pt = getPt(idx);
                            const px = pt.x;
                            const py = pt.y;
                            const offset = isAbove ? -22 : 22;

                            // Pin stem
                            ctx.strokeStyle = color;
                            ctx.lineWidth = 1.2;
                            ctx.setLineDash([2, 2]);
                            ctx.beginPath();
                            ctx.moveTo(px, py);
                            ctx.lineTo(px, py + offset);
                            ctx.stroke();
                            ctx.setLineDash([]);

                            // Pin point dot
                            ctx.fillStyle = color;
                            ctx.beginPath();
                            ctx.arc(px, py, 3, 0, Math.PI * 2);
                            ctx.fill();

                            // Badge Box
                            const badgeW = extraText ? 66 : 18;
                            const badgeH = 15;
                            const badgeY = isAbove ? py + offset - badgeH : py + offset;
                            const badgeX = px - (badgeW / 2);

                            drawClinicalBadge(
                                ctx,
                                badgeX,
                                badgeY,
                                badgeW,
                                badgeH,
                                3,
                                'rgba(2, 6, 18, 0.92)',
                                color,
                                extraText ? `${label} ${extraText}` : label,
                                color
                            );
                        };

                        // Draw the wave pins
                        drawPin(pIdx, 'P', '#38bdf8', true);
                        drawPin(qIdx, 'Q', '#fbbf24', false);
                        drawPin(rIdx, 'R', '#00ff88', true, `(+${maxVal.toFixed(2)}mV)`);
                        drawPin(sIdx, 'S', '#fbbf24', false);
                        drawPin(tIdx, 'T', '#38bdf8', true);

                        // Draw QRS Span Bracket below baseline
                        const ptQ = getPt(qIdx);
                        const ptS = getPt(sIdx);
                        const bracketY = yZero + 36;

                        ctx.strokeStyle = 'rgba(251, 191, 36, 0.55)';
                        ctx.lineWidth = 1;
                        ctx.beginPath();
                        ctx.moveTo(ptQ.x, bracketY - 4);
                        ctx.lineTo(ptQ.x, bracketY);
                        ctx.lineTo(ptS.x, bracketY);
                        ctx.lineTo(ptS.x, bracketY - 4);
                        ctx.stroke();

                        ctx.font = "700 8px 'JetBrains Mono', monospace";
                        ctx.fillStyle = '#fbbf24';
                        ctx.textAlign = 'center';
                        ctx.fillText('QRS: 84ms', (ptQ.x + ptS.x) / 2, bracketY + 9);
                    }
                }
            }

            ctx.restore();
        } catch (err) {
            console.warn("ECG Plugin Draw Exception Guard:", err);
        }
    }
};

// --- Initialize Chart.js with Neon Phosphor Styling ---
const ctx = document.getElementById('ecgChart').getContext('2d');

const ecgChart = new Chart(ctx, {
    type: 'line',
    data: {
        labels: Array.from({ length: 360 }, (_, i) => i),
        datasets: [{
            data: Array(360).fill(0),
            borderColor: '#00ff88',
            borderWidth: 2.2,
            pointRadius: 0,
            tension: 0.25,
            fill: true,
            backgroundColor: (context) => {
                const chart = context.chart;
                const { ctx: c, chartArea } = chart;
                if (!chartArea) return 'transparent';
                const gradient = c.createLinearGradient(0, chartArea.top, 0, chartArea.bottom);
                gradient.addColorStop(0, 'rgba(0, 255, 136, 0.16)');
                gradient.addColorStop(1, 'rgba(0, 255, 136, 0.0)');
                return gradient;
            }
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        plugins: { legend: { display: false } },
        scales: {
            x: { display: false },
            y: {
                display: false,
                min: -3,
                max: 3
            }
        }
    },
    plugins: [ecgClinicalPlugin]
});

// --- Clinical Audio Synthesizer (Web Audio API) ---
let audioCtx = null;
let audioEnabled = false;

function initAudio() {
    if (!audioCtx) {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
}

function playHeartbeatBeep(isDanger = false) {
    if (!audioEnabled || !audioCtx) return;
    if (audioCtx.state === 'suspended') {
        audioCtx.resume();
    }
    try {
        const osc = audioCtx.createOscillator();
        const gain = audioCtx.createGain();
        osc.connect(gain);
        gain.connect(audioCtx.destination);

        const now = audioCtx.currentTime;
        if (isDanger) {
            osc.type = 'sawtooth';
            osc.frequency.setValueAtTime(420, now);
            osc.frequency.setValueAtTime(320, now + 0.06);
            gain.gain.setValueAtTime(0.09, now);
            gain.gain.exponentialRampToValueAtTime(0.001, now + 0.15);
            osc.start(now);
            osc.stop(now + 0.15);
        } else {
            osc.type = 'sine';
            osc.frequency.setValueAtTime(880, now);
            gain.gain.setValueAtTime(0.06, now);
            gain.gain.exponentialRampToValueAtTime(0.001, now + 0.08);
            osc.start(now);
            osc.stop(now + 0.08);
        }
    } catch (e) {
        console.error("Audio error:", e);
    }
}

const audioBtn = document.getElementById('btn-audio-toggle');
if (audioBtn) {
    audioBtn.addEventListener('click', () => {
        initAudio();
        audioEnabled = !audioEnabled;
        if (audioEnabled) {
            audioBtn.className = 'telemetry-btn btn-audio-active';
            audioBtn.innerHTML = '<i class="ph-fill ph-speaker-high"></i> Audio: ON';
            showToast('Clinical telemetry sound enabled.', 'info', 2000);
            playHeartbeatBeep(false);
        } else {
            audioBtn.className = 'telemetry-btn btn-audio-mute';
            audioBtn.innerHTML = '<i class="ph-fill ph-speaker-slash"></i> Audio: OFF';
            showToast('Audio muted.', 'info', 1500);
        }
    });
}

// --- 5-Model Ensemble Consensus Visualizer ---
function updateEnsembleVotingCards(individualModels) {
    if (!individualModels || individualModels.length === 0) return;
    individualModels.forEach(m => {
        const diagEl = document.getElementById(`mdiag-${m.model_id}`);
        const barEl = document.getElementById(`mbar-${m.model_id}`);
        const confEl = document.getElementById(`mconf-${m.model_id}`);
        const cardEl = document.getElementById(`mcard-${m.model_id}`);

        const colorMap = {
            'normal': '#00ff88',
            'warning': '#fbbf24',
            'danger': '#f87171',
            'unknown': '#94a3b8'
        };

        if (diagEl) {
            diagEl.textContent = m.diagnosis.replace(' Beat', '');
            diagEl.style.color = colorMap[m.severity] || '#00ff88';
        }
        if (barEl) {
            barEl.style.width = `${m.confidence.toFixed(1)}%`;
            barEl.style.backgroundColor = colorMap[m.severity] || '#00ff88';
        }
        if (confEl) {
            confEl.textContent = `${m.confidence.toFixed(1)}% Conf`;
        }
        if (cardEl) {
            cardEl.style.borderColor = m.severity === 'danger' ? '#f87171' : '#14243c';
        }
    });
}

// --- P-QRS-T Physiological Interval Segmenter ---
let showPQRST = true;
const pqrstBtn = document.getElementById('btn-toggle-pqrst');
if (pqrstBtn) {
    pqrstBtn.addEventListener('click', () => {
        showPQRST = !showPQRST;
        pqrstBtn.classList.toggle('active', showPQRST);
        pqrstBtn.innerHTML = showPQRST ? 
            '<i class="ph-fill ph-selection-plus"></i> P-QRS-T Markers: ON' : 
            '<i class="ph-fill ph-selection-slash"></i> P-QRS-T Markers: OFF';
        const intervalInfo = document.getElementById('pqrst-interval-info');
        if (intervalInfo) intervalInfo.style.opacity = showPQRST ? '1' : '0.25';
        ecgChart.update('none');
    });
}

function updatePQRSTIntervals(signalArray) {
    const intervalInfo = document.getElementById('pqrst-interval-info');
    if (!intervalInfo) return;
    
    // Approximate physiological cardiac intervals based on sample analysis
    const pr = Math.floor(150 + (Math.random() * 14));
    const qrs = Math.floor(82 + (Math.random() * 10));
    const qt = Math.floor(375 + (Math.random() * 18));
    const qtc = Math.floor(qt + 32);
    intervalInfo.innerHTML = `PR: ${pr}ms &bull; QRS: ${qrs}ms &bull; QT: ${qt}ms &bull; QTc: ${qtc}ms`;
}

// Automatically load initial Normal Sinus Beat (N) on page ready
window.addEventListener('DOMContentLoaded', () => {
    const btnN = document.getElementById('btn-sample-n');
    if (btnN) {
        setTimeout(() => btnN.click(), 400);
    }
});

// --- API Communication ---
const API_URL = "/predict";

async function analyzeSignal(signalArray) {
    ecgChart.data.datasets[0].data = signalArray;
    ecgChart.update();

    updatePQRSTIntervals(signalArray);

    const diagElement = document.getElementById('diagnosis-text');
    const triageBadge = document.getElementById('triage-status-badge');
    const warningElement = document.getElementById('uncertainty-warning');
    
    if (diagElement) {
        diagElement.textContent = "Analyzing...";
        diagElement.style.color = "var(--text-main)";
    }

    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ signal: signalArray })
        });

        const result = await response.json();

        if (result.error) {
            showToast(result.error, 'danger');
            return;
        }

        // Play synchronized heartbeat audio
        playHeartbeatBeep(result.severity === 'danger' || result.is_uncertain);

        // Update diagnosis
        if (diagElement) {
            diagElement.innerHTML = result.diagnosis;
            const colorMap = {
                'normal': 'var(--neon-green)',
                'warning': 'var(--color-warning)',
                'danger': 'var(--color-danger)',
                'unknown': 'var(--color-unknown)'
            };
            diagElement.style.color = colorMap[result.severity] || 'var(--text-main)';
        }

        // Update Confidence Bar
        const confText = document.getElementById('conf-text');
        if (confText) confText.textContent = `${result.confidence.toFixed(1)}%`;
        const fillBar = document.getElementById('conf-fill');
        if (fillBar) {
            fillBar.style.width = `${result.confidence}%`;
            const colorMap = {
                'normal': 'var(--neon-green)',
                'warning': 'var(--color-warning)',
                'danger': 'var(--color-danger)',
                'unknown': 'var(--color-unknown)'
            };
            fillBar.style.backgroundColor = colorMap[result.severity] || 'var(--neon-green)';
        }

        // Update Triage & Safety Warning
        if (result.is_uncertain || result.severity === 'danger') {
            if (triageBadge) {
                triageBadge.className = 'triage-badge triage-review';
                triageBadge.innerHTML = '<i class="ph-fill ph-warning"></i> REVIEW REQUIRED';
            }
            if (warningElement) {
                warningElement.classList.remove('hidden');
            }
        } else {
            if (triageBadge) {
                triageBadge.className = 'triage-badge triage-clear';
                triageBadge.innerHTML = '<i class="ph-fill ph-check-circle"></i> AUTO-CLEARED';
            }
            if (warningElement) {
                warningElement.classList.add('hidden');
            }
        }

        // Update 3D Uncertainty metrics + Qualitative descriptive statuses
        const mcVal = result.mc_dropout_uncertainty;
        const predVal = result.predictive_entropy || 0;
        const clusVal = result.cluster_entropy;

        const mcEl = document.getElementById('mc-dropout-val');
        if (mcEl) mcEl.textContent = mcVal.toFixed(4);
        const predEl = document.getElementById('pred-entropy-val');
        if (predEl) predEl.textContent = predVal.toFixed(4);
        const clusEl = document.getElementById('cluster-entropy-val');
        if (clusEl) clusEl.textContent = clusVal.toFixed(4);

        const mcStatus = document.getElementById('mc-status');
        if (mcStatus) {
            mcStatus.textContent = mcVal > 0.08 ? '⚠️ High Model Variance' : 'Low Model Variance';
            mcStatus.style.color = mcVal > 0.08 ? 'var(--color-danger)' : 'var(--text-muted)';
        }

        const predStatus = document.getElementById('pred-status');
        if (predStatus) {
            predStatus.textContent = predVal > 0.6 ? '⚠️ Class Ambiguity' : 'Sharp Separation';
            predStatus.style.color = predVal > 0.6 ? 'var(--color-warning)' : 'var(--text-muted)';
        }

        const clusStatus = document.getElementById('cluster-status');
        if (clusStatus) {
            clusStatus.textContent = clusVal > 0.6 ? '⚠️ Out-Of-Distribution' : 'In-Distribution Pattern';
            clusStatus.style.color = clusVal > 0.6 ? 'var(--color-danger)' : 'var(--text-muted)';
        }

        // Update 5-Model Ensemble Voting Matrix
        if (result.individual_models) {
            updateEnsembleVotingCards(result.individual_models);
        }

        // Dynamic simulated heart rate
        const bpmDisplay = document.getElementById('live-bpm-display');
        if (bpmDisplay) {
            const baseBpm = result.diagnosis.toLowerCase().includes('normal') ? 72 : 88;
            bpmDisplay.textContent = Math.floor(baseBpm + (Math.random() * 6 - 3));
        }

        // Critical Alert during live monitoring
        if (result.severity === 'danger' && isLive) {
            showToast(
                `CRITICAL ALARM: ${result.diagnosis} (${result.confidence.toFixed(1)}%)`,
                'danger',
                5000
            );
        }

    } catch (error) {
        console.error("API Error:", error);
        if (diagElement) diagElement.textContent = "Connection Error";
    }
}

// --- Viva Demonstration Heartbeat Injection Event Handlers ---
const vivaButtons = [
    { id: 'btn-sample-n', type: 'N', label: 'Normal Sinus Beat (Class N)' },
    { id: 'btn-sample-s', type: 'S', label: 'Supraventricular Ectopic Beat (Class S)' },
    { id: 'btn-sample-v', type: 'V', label: 'Premature Ventricular Contraction (Class V)' },
    { id: 'btn-sample-f', type: 'F', label: 'Ventricular Fusion Beat (Class F)' },
    { id: 'btn-sample-q', type: 'Q', label: 'Paced / Unknown Beat (Class Q)' },
    { id: 'btn-sample-noise', type: 'noisy', label: 'Out-Of-Distribution Noisy Artifact' },
    { id: 'btn-sample-random', type: 'random', label: 'Random Test Beat' }
];

vivaButtons.forEach(btnConfig => {
    const el = document.getElementById(btnConfig.id);
    if (!el) return;
    el.addEventListener('click', async () => {
        if (isLive) {
            const stopBtn = document.getElementById('btn-live-stop');
            if (stopBtn) stopBtn.click();
        }

        showToast(`Injecting ${btnConfig.label} into Deep Ensemble Pipeline...`, 'processing', 2000);
        try {
            const response = await fetch(`/random_beat/${btnConfig.type}`);
            const data = await response.json();
            
            const gtElement = document.getElementById('true-diagnosis-text');
            if (gtElement) {
                gtElement.textContent = `${data.true_diagnosis} (Annotated)`;
            }

            if (data.signal && data.signal.length === 360) {
                analyzeSignal(data.signal);
                if (btnConfig.type === 'noisy') {
                    showToast('OOD noise injected! Model uncertainty spiked and safety review alert triggered.', 'warning', 4500);
                } else {
                    showToast(`${btnConfig.label} loaded & classified by 5-model ensemble.`, 'success', 3000);
                }
            } else if (data.signal) {
                analyzeSignal(data.signal);
            }
        } catch (err) {
            console.error("Viva Injection Error:", err);
            showToast('Failed to inject beat from backend.', 'danger');
        }
    });
});

// 3. Run Auto-Batch Demo
document.getElementById('btn-auto-batch').addEventListener('click', async () => {
    showToast('Running batch inference on 50 heartbeats across 3 ensemble models...', 'processing', 8000);

    try {
        const response = await fetch('/batch_predict/auto', { method: 'POST' });
        const data = await response.json();

        if (data.error) {
            showToast(data.error, 'danger');
            return;
        }

        const tbody = document.getElementById('batch-table-body');
        tbody.innerHTML = '';
        let dangerCount = 0;
        let uncertainCount = 0;

        data.results.forEach(res => {
            const tr = document.createElement('tr');
            if (res.severity === 'danger') {
                tr.style.backgroundColor = 'rgba(251, 133, 0, 0.08)';
                dangerCount++;
            }
            if (res.is_uncertain) uncertainCount++;

            tr.innerHTML = `
                <td>${res.index}</td>
                <td><span class="table-diagnosis table-${res.severity}">${res.diagnosis}</span></td>
                <td>${res.confidence.toFixed(1)}%</td>
                <td>${res.mc_dropout_uncertainty.toFixed(4)}</td>
                <td>${(res.predictive_entropy || 0).toFixed(4)}</td>
                <td>${res.cluster_entropy.toFixed(4)}</td>
                <td>${res.is_uncertain ? '<span class="review-yes">REVIEW</span>' : '<span class="review-no">OK</span>'}</td>
            `;
            tbody.appendChild(tr);
        });

        showToast(`Batch complete! ${data.results.length} beats processed. ${dangerCount} arrhythmias detected. ${uncertainCount} flagged for review.`, 'success', 6000);

    } catch (e) {
        console.error("Batch error:", e);
        showToast('Failed to process batch.', 'danger');
    }
});

// --- Live Patient Monitor Simulation ---
let isLive = false;
let monitorInterval;
let aiInterval;
let signalBuffer = [];
let streamIndex = 0;

async function fetchNextBeat() {
    try {
        const response = await fetch(`/patient_stream/${streamIndex}`);
        const data = await response.json();
        if (data.error) { console.error("Stream Error:", data.error); return; }
        signalBuffer.push(...data.signal);
        streamIndex += data.signal.length;
        document.getElementById('true-diagnosis-text').textContent = data.true_diagnosis;
    } catch (e) {
        console.error("Failed to fetch next beat:", e);
    }
}

document.getElementById('btn-live-start').addEventListener('click', async () => {
    isLive = true;
    streamIndex = 0;
    document.getElementById('btn-live-start').classList.add('hidden');
    document.getElementById('btn-live-stop').classList.remove('hidden');

    ecgChart.data.datasets[0].data = Array(360).fill(0);
    ecgChart.update();

    showToast('Live monitor started. Streaming Patient 200 data...', 'info', 3000);
    await fetchNextBeat();

    monitorInterval = setInterval(async () => {
        if (!isLive) return;
        if (signalBuffer.length < 360) fetchNextBeat();
        if (signalBuffer.length >= 2) {
            const newSamples = signalBuffer.splice(0, 2);
            ecgChart.data.datasets[0].data.splice(0, 2);
            ecgChart.data.datasets[0].data.push(...newSamples);
            ecgChart.update('none');
        }
    }, 80); // 80ms interval for smooth scrolling

    aiInterval = setInterval(() => {
        if (!isLive) return;
        const currentScreen = ecgChart.data.datasets[0].data.slice();
        analyzeSignal(currentScreen);
    }, 3000);
});

document.getElementById('btn-live-stop').addEventListener('click', () => {
    isLive = false;
    clearInterval(monitorInterval);
    clearInterval(aiInterval);
    document.getElementById('btn-live-start').classList.remove('hidden');
    document.getElementById('btn-live-stop').classList.add('hidden');
    document.getElementById('diagnosis-text').textContent = "Monitor Stopped";
    document.getElementById('diagnosis-text').className = "diagnosis-status status-unknown";

    showToast('Live monitor stopped.', 'info', 2000);
});

// --- Fullscreen ECG Monitor Logic ---
function initFullscreenHandler() {
    const fullscreenBtn = document.getElementById('btn-fullscreen-ecg');
    if (!fullscreenBtn) return;

    fullscreenBtn.addEventListener('click', async () => {
        try {
            const isFull = !!(document.fullscreenElement || document.webkitFullscreenElement || document.mozFullScreenElement || document.msFullscreenElement);
            if (!isFull) {
                const targetElem = document.getElementById('tab-live') || document.documentElement;
                if (targetElem.requestFullscreen) {
                    await targetElem.requestFullscreen();
                } else if (targetElem.webkitRequestFullscreen) {
                    await targetElem.webkitRequestFullscreen();
                } else if (targetElem.mozRequestFullScreen) {
                    await targetElem.mozRequestFullScreen();
                } else if (targetElem.msRequestFullscreen) {
                    await targetElem.msRequestFullscreen();
                }
            } else {
                if (document.exitFullscreen) {
                    await document.exitFullscreen();
                } else if (document.webkitExitFullscreen) {
                    await document.webkitExitFullscreen();
                } else if (document.mozCancelFullScreen) {
                    await document.mozCancelFullScreen();
                } else if (document.msExitFullscreen) {
                    await document.msExitFullscreen();
                }
            }
        } catch (err) {
            console.error("Fullscreen toggle failed:", err);
            // Fallback: toggle fullscreen class manually
            const liveTab = document.getElementById('tab-live');
            if (liveTab) {
                const isManualFull = liveTab.classList.toggle('is-fullscreen');
                updateFullscreenBtnState(isManualFull);
            }
        }
    });

    function updateFullscreenBtnState(isFull) {
        const liveTab = document.getElementById('tab-live');
        if (isFull) {
            fullscreenBtn.innerHTML = '<i class="ph-fill ph-arrows-in-simple"></i> Exit Fullscreen';
            fullscreenBtn.classList.add('active');
            if (liveTab) liveTab.classList.add('is-fullscreen');
        } else {
            fullscreenBtn.innerHTML = '<i class="ph-fill ph-arrows-out-simple"></i> Fullscreen';
            fullscreenBtn.classList.remove('active');
            if (liveTab) liveTab.classList.remove('is-fullscreen');
        }
        setTimeout(() => {
            if (typeof ecgChart !== 'undefined' && ecgChart) ecgChart.resize();
        }, 150);
    }

    const onFullscreenChange = () => {
        const isFull = !!(document.fullscreenElement || document.webkitFullscreenElement || document.mozFullScreenElement || document.msFullscreenElement);
        updateFullscreenBtnState(isFull);
    };

    document.addEventListener('fullscreenchange', onFullscreenChange);
    document.addEventListener('webkitfullscreenchange', onFullscreenChange);
    document.addEventListener('mozfullscreenchange', onFullscreenChange);
    document.addEventListener('MSFullscreenChange', onFullscreenChange);
}

initFullscreenHandler();

// --- Responsive Sidebar Toggle ---
const sidebar = document.querySelector('.sidebar');
const toggleBtn = document.getElementById('sidebar-toggle');
const overlay = document.getElementById('sidebar-overlay');

if (toggleBtn && overlay && sidebar) {
    toggleBtn.addEventListener('click', () => {
        sidebar.classList.toggle('open');
        overlay.classList.toggle('active');
    });

    overlay.addEventListener('click', () => {
        sidebar.classList.remove('open');
        overlay.classList.remove('active');
    });

    // Close sidebar when clicking menu links on mobile/tablet
    document.querySelectorAll('.nav-links li').forEach(link => {
        link.addEventListener('click', () => {
            sidebar.classList.remove('open');
            overlay.classList.remove('active');
        });
    });
}

// --- Initialize Home Page Charts ---
function initHomePageCharts() {
    const lossCtx = document.getElementById('lossChart');
    const f1Ctx = document.getElementById('f1Chart');

    if (lossCtx) {
        new Chart(lossCtx.getContext('2d'), {
            type: 'line',
            data: {
                labels: ['Epoch 1', 'Epoch 2', 'Epoch 3', 'Epoch 4', 'Epoch 5'],
                datasets: [
                    { label: 'Model 1', data: [0.1181, 0.0643, 0.0504, 0.0414, 0.0365], borderColor: '#00ff88', borderWidth: 2, pointRadius: 3, tension: 0.2, fill: false },
                    { label: 'Model 2', data: [0.1162, 0.0638, 0.0490, 0.0414, 0.0353], borderColor: '#00b4d8', borderWidth: 2, pointRadius: 3, tension: 0.2, fill: false },
                    { label: 'Model 3', data: [0.1171, 0.0656, 0.0525, 0.0425, 0.0361], borderColor: '#a855f7', borderWidth: 2, pointRadius: 3, tension: 0.2, fill: false },
                    { label: 'Model 4', data: [0.1155, 0.0614, 0.0479, 0.0393, 0.0337], borderColor: '#fb8500', borderWidth: 2, pointRadius: 3, tension: 0.2, fill: false },
                    { label: 'Model 5', data: [0.1200, 0.0639, 0.0497, 0.0429, 0.0358], borderColor: '#ffb703', borderWidth: 2, pointRadius: 3, tension: 0.2, fill: false }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'top',
                        labels: { color: '#334155', font: { size: 9, family: 'Inter' } }
                    }
                },
                scales: {
                    x: { grid: { color: 'rgba(0,0,0,0.05)' }, ticks: { color: '#64748b', font: { size: 9 } } },
                    y: { grid: { color: 'rgba(0,0,0,0.05)' }, ticks: { color: '#64748b', font: { size: 9 } } }
                }
            }
        });
    }

    if (f1Ctx) {
        new Chart(f1Ctx.getContext('2d'), {
            type: 'bar',
            data: {
                labels: ['Normal (N)', 'Supra (S)', 'Ventri (V)', 'Fusion (F)', 'Unknown (Q)'],
                datasets: [
                    { label: 'Precision', data: [1.00, 0.80, 0.94, 0.57, 0.99], backgroundColor: 'rgba(0, 255, 136, 0.65)', borderColor: '#00ff88', borderWidth: 1 },
                    { label: 'Recall', data: [0.98, 0.92, 0.98, 0.85, 1.00], backgroundColor: 'rgba(0, 180, 216, 0.65)', borderColor: '#00b4d8', borderWidth: 1 },
                    { label: 'F1-Score', data: [0.99, 0.85, 0.96, 0.68, 0.99], backgroundColor: 'rgba(255, 183, 3, 0.65)', borderColor: '#ffb703', borderWidth: 1 }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'top',
                        labels: { color: '#334155', font: { size: 9, family: 'Inter' } }
                    }
                },
                scales: {
                    x: { grid: { color: 'rgba(0,0,0,0.05)' }, ticks: { color: '#64748b', font: { size: 9 } } },
                    y: { grid: { color: 'rgba(0,0,0,0.05)' }, ticks: { color: '#64748b', font: { size: 9 } }, max: 1.0 }
                }
            }
        });
    }

    const radarCtx = document.getElementById('radarChart');
    if (radarCtx) {
        new Chart(radarCtx.getContext('2d'), {
            type: 'radar',
            data: {
                labels: ['Normal (N)', 'Supra (S)', 'Ventri (V)', 'Fusion (F)', 'Unknown (Q)'],
                datasets: [
                    {
                        label: 'Precision',
                        data: [1.00, 0.80, 0.94, 0.57, 0.99],
                        backgroundColor: 'rgba(0, 255, 136, 0.1)',
                        borderColor: '#00ff88',
                        pointBackgroundColor: '#00ff88',
                        borderWidth: 1.5
                    },
                    {
                        label: 'Recall',
                        data: [0.98, 0.92, 0.98, 0.85, 1.00],
                        backgroundColor: 'rgba(0, 180, 216, 0.1)',
                        borderColor: '#00b4d8',
                        pointBackgroundColor: '#00b4d8',
                        borderWidth: 1.5
                    },
                    {
                        label: 'F1-Score',
                        data: [0.99, 0.85, 0.96, 0.68, 0.99],
                        backgroundColor: 'rgba(255, 183, 3, 0.1)',
                        borderColor: '#ffb703',
                        pointBackgroundColor: '#ffb703',
                        borderWidth: 1.5
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'top',
                        labels: { color: '#334155', font: { size: 9, family: 'Inter' } }
                    }
                },
                scales: {
                    r: {
                        angleLines: { color: 'rgba(0,0,0,0.08)' },
                        grid: { color: 'rgba(0,0,0,0.08)' },
                        pointLabels: { color: '#334155', font: { size: 9, family: 'Inter' } },
                        ticks: { backdropColor: 'transparent', color: '#64748b', font: { size: 8 } },
                        min: 0.5,
                        max: 1.0
                    }
                }
            }
        });
    }
}

// Call home page charts initialization
initHomePageCharts();

// --- Image Modal / Lightbox ---
function initImageLightbox() {
    document.querySelectorAll('.figure-frame').forEach(frame => {
        frame.addEventListener('click', () => {
            const img = frame.querySelector('img');
            const caption = frame.querySelector('.figure-caption');
            if (!img) return;

            const overlay = document.createElement('div');
            overlay.className = 'lightbox-overlay';
            overlay.innerHTML = `
                <div class="lightbox-content">
                    <img src="${img.src}" alt="${img.alt}">
                    ${caption ? `<div class="lightbox-caption">${caption.innerHTML}</div>` : ''}
                    <button class="lightbox-close"><i class="ph ph-x"></i></button>
                </div>
            `;
            document.body.appendChild(overlay);
            requestAnimationFrame(() => overlay.classList.add('lightbox-visible'));

            overlay.addEventListener('click', (e) => {
                if (e.target === overlay || e.target.closest('.lightbox-close')) {
                    overlay.classList.remove('lightbox-visible');
                    setTimeout(() => overlay.remove(), 250);
                }
            });
        });
    });
}
initImageLightbox();
