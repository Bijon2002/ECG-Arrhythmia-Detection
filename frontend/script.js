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
    });
});

// --- Custom Toast Notification System ---
function showToast(message, type = 'info', duration = 4000) {
    const container = document.getElementById('toast-container');
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
        danger: { bg: 'rgba(251, 133, 0, 0.15)', border: '#fb8500', icon: 'ph-fill ph-siren', glow: 'rgba(251, 133, 0, 0.4)' },
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

// --- Initialize Chart.js ---
const ctx = document.getElementById('ecgChart').getContext('2d');

const ecgChart = new Chart(ctx, {
    type: 'line',
    data: {
        labels: Array.from({ length: 360 }, (_, i) => i),
        datasets: [{
            data: Array(360).fill(0),
            borderColor: '#00ff88',
            borderWidth: 1.5,
            pointRadius: 0,
            tension: 0.3,
            fill: true,
            backgroundColor: (context) => {
                const chart = context.chart;
                const { ctx: c, chartArea } = chart;
                if (!chartArea) return 'transparent';
                const gradient = c.createLinearGradient(0, chartArea.top, 0, chartArea.bottom);
                gradient.addColorStop(0, 'rgba(0, 255, 136, 0.15)');
                gradient.addColorStop(1, 'rgba(0, 255, 136, 0.0)');
                return gradient;
            }
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 400 },
        plugins: { legend: { display: false } },
        scales: {
            x: { display: false },
            y: {
                display: true,
                min: -3,
                max: 3,
                grid: {
                    color: 'rgba(255, 255, 255, 0.03)',
                    drawBorder: false
                },
                ticks: { color: 'rgba(255,255,255,0.2)', font: { family: 'JetBrains Mono', size: 10 } }
            }
        }
    }
});

// --- API Communication ---
const API_URL = "/predict";

async function analyzeSignal(signalArray) {
    ecgChart.data.datasets[0].data = signalArray;
    ecgChart.update();

    document.getElementById('diagnosis-text').textContent = "Analyzing...";
    document.getElementById('diagnosis-text').className = "diagnosis-status status-unknown";

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

        // Update diagnosis
        const diagElement = document.getElementById('diagnosis-text');
        diagElement.innerHTML = result.diagnosis;
        diagElement.className = `diagnosis-status status-${result.severity}`;

        // Update Confidence Bar
        document.getElementById('conf-text').textContent = `${result.confidence.toFixed(1)}%`;
        const fillBar = document.getElementById('conf-fill');
        fillBar.style.width = `${result.confidence}%`;
        fillBar.style.backgroundColor = `var(--color-${result.severity})`;

        // Update Warning
        const warningElement = document.getElementById('uncertainty-warning');
        if (result.is_uncertain) {
            diagElement.textContent += " (High Entropy)";
            warningElement.classList.remove('hidden');
        } else {
            warningElement.classList.add('hidden');
        }

        // Update metrics
        document.getElementById('mc-dropout-val').textContent = result.mc_dropout_uncertainty.toFixed(4);
        document.getElementById('pred-entropy-val').textContent = result.predictive_entropy.toFixed(4);
        document.getElementById('cluster-entropy-val').textContent = result.cluster_entropy.toFixed(4);

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
        document.getElementById('diagnosis-text').textContent = "Connection Error";
    }
}

// --- Event Listeners ---

// 1. Load Normal Beat
document.getElementById('btn-sample-normal').addEventListener('click', async () => {
    showToast('Loading normal heartbeat from MIT-BIH test set...', 'processing', 2000);
    try {
        const response = await fetch('/random_beat/normal');
        const data = await response.json();
        document.getElementById('true-diagnosis-text').textContent = data.true_diagnosis;
        if (data.signal.length !== 360) {
            showToast(`Expected 360 samples, got ${data.signal.length}`, 'warning');
            return;
        }
        analyzeSignal(data.signal);
        showToast(`Normal beat loaded successfully. Ground truth: ${data.true_diagnosis}`, 'success', 3000);
    } catch (err) {
        console.error(err);
        showToast('Failed to load sample from backend.', 'danger');
    }
});

// 2. Load Abnormal Beat
document.getElementById('btn-sample-abnormal').addEventListener('click', async () => {
    showToast('Loading abnormal heartbeat from MIT-BIH test set...', 'processing', 2000);
    try {
        const response = await fetch('/random_beat/abnormal');
        const data = await response.json();
        document.getElementById('true-diagnosis-text').textContent = data.true_diagnosis;
        if (data.signal.length !== 360) {
            showToast(`Expected 360 samples, got ${data.signal.length}`, 'warning');
            return;
        }
        analyzeSignal(data.signal);
        showToast(`Abnormal beat loaded. Ground truth: ${data.true_diagnosis}`, 'warning', 3000);
    } catch (err) {
        console.error(err);
        showToast('Failed to load sample from backend.', 'danger');
    }
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

    // Request fullscreen on the right-col monitor
    const monitorElement = document.querySelector('.right-col');
    if (monitorElement && monitorElement.requestFullscreen) {
        monitorElement.requestFullscreen().catch(err => console.log(err));
    }

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
    }, 50);

    aiInterval = setInterval(() => {
        if (!isLive) return;
        const currentScreen = ecgChart.data.datasets[0].data.slice();
        analyzeSignal(currentScreen);
    }, 1000);
});

document.getElementById('btn-live-stop').addEventListener('click', () => {
    isLive = false;
    clearInterval(monitorInterval);
    clearInterval(aiInterval);
    document.getElementById('btn-live-start').classList.remove('hidden');
    document.getElementById('btn-live-stop').classList.add('hidden');
    document.getElementById('diagnosis-text').textContent = "Monitor Stopped";
    document.getElementById('diagnosis-text').className = "diagnosis-status status-unknown";

    if (document.fullscreenElement) {
        document.exitFullscreen().catch(err => console.log(err));
    }
    showToast('Live monitor stopped.', 'info', 2000);
});

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
                        labels: { color: 'rgba(255,255,255,0.7)', font: { size: 9, family: 'Inter' } }
                    }
                },
                scales: {
                    x: { grid: { color: 'rgba(255,255,255,0.03)' }, ticks: { color: 'rgba(255,255,255,0.5)', font: { size: 9 } } },
                    y: { grid: { color: 'rgba(255,255,255,0.03)' }, ticks: { color: 'rgba(255,255,255,0.5)', font: { size: 9 } } }
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
                        labels: { color: 'rgba(255,255,255,0.7)', font: { size: 9, family: 'Inter' } }
                    }
                },
                scales: {
                    x: { grid: { color: 'rgba(255,255,255,0.03)' }, ticks: { color: 'rgba(255,255,255,0.5)', font: { size: 9 } } },
                    y: { grid: { color: 'rgba(255,255,255,0.03)' }, ticks: { color: 'rgba(255,255,255,0.5)', font: { size: 9 } }, max: 1.0 }
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
                        labels: { color: 'rgba(255,255,255,0.7)', font: { size: 9, family: 'Inter' } }
                    }
                },
                scales: {
                    r: {
                        angleLines: { color: 'rgba(255,255,255,0.06)' },
                        grid: { color: 'rgba(255,255,255,0.06)' },
                        pointLabels: { color: 'rgba(255,255,255,0.6)', font: { size: 9, family: 'Inter' } },
                        ticks: { backdropColor: 'transparent', color: 'rgba(255,255,255,0.4)', font: { size: 8 } },
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
