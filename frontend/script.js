// --- Initialize Chart.js ---
const ctx = document.getElementById('ecgChart').getContext('2d');

// Create a glowing gradient line for the chart
const gradient = ctx.createLinearGradient(0, 0, 800, 0);
gradient.addColorStop(0, 'rgba(0, 255, 136, 0.8)');
gradient.addColorStop(1, 'rgba(0, 255, 136, 1)');

const ecgChart = new Chart(ctx, {
    type: 'line',
    data: {
        labels: Array.from({ length: 360 }, (_, i) => i),
        datasets: [{
            label: 'ECG Signal',
            data: Array(360).fill(0),
            borderColor: gradient,
            borderWidth: 2,
            tension: 0.4, // Smooth curves
            pointRadius: 0, // Hide dots
            fill: false,
            shadowOffsetX: 0,
            shadowOffsetY: 0,
            shadowBlur: 10,
            shadowColor: 'rgba(0, 255, 136, 0.5)'
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: {
            duration: 800,
            easing: 'easeOutQuart'
        },
        plugins: {
            legend: { display: false }
        },
        scales: {
            x: { display: false },
            y: {
                display: true,
                min: -3,
                max: 3,
                grid: {
                    color: 'rgba(255, 255, 255, 0.05)',
                    drawBorder: false
                },
                ticks: { color: 'rgba(255,255,255,0.3)' }
            }
        }
    }
});

// --- API Communication ---
const API_URL = "/predict";

async function analyzeSignal(signalArray) {
    // Update Graph
    ecgChart.data.datasets[0].data = signalArray;
    ecgChart.update();

    // Show Loading State
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
            alert(result.error);
            return;
        }

        // Update UI with results
        const diagElement = document.getElementById('diagnosis-text');
        diagElement.innerHTML = `${result.icon} ${result.diagnosis}`;
        diagElement.className = `diagnosis-status status-${result.severity}`;

        // Update Confidence Bar
        document.getElementById('conf-text').textContent = `${result.confidence.toFixed(1)}%`;
        const fillBar = document.getElementById('conf-fill');
        fillBar.style.width = `${result.confidence}%`;
        fillBar.style.backgroundColor = `var(--color-${result.severity})`;

        // Update Warning
        const warningElement = document.getElementById('uncertainty-warning');
        if (result.is_uncertain) {
            diagElement.textContent += " (Low Confidence / High Entropy)";
            warningElement.classList.remove('hidden');
        } else {
            warningElement.classList.add('hidden');
        }

        // Update Mathematical Uncertainty Metrics
        document.getElementById('mc-dropout-val').textContent = result.mc_dropout_uncertainty.toFixed(4);
        document.getElementById('pred-entropy-val').textContent = result.predictive_entropy.toFixed(4);
        document.getElementById('cluster-entropy-val').textContent = result.cluster_entropy.toFixed(4);

        // Critical Alert Notification
        if (result.severity === 'danger' && isLive) {
            Toastify({
                text: `CRITICAL ALARM!\nDetected: ${result.diagnosis}\nConfidence: ${result.confidence.toFixed(1)}%`,
                duration: 5000,
                close: true,
                gravity: "top",
                position: "center",
                style: {
                    background: "linear-gradient(to right, #ff0000, #ff4d4d)",
                    color: "white",
                    fontWeight: "bold",
                    borderRadius: "8px",
                    boxShadow: "0 4px 12px rgba(255, 0, 0, 0.4)"
                }
            }).showToast();
        }

    } catch (error) {
        console.error("API Error:", error);
        document.getElementById('diagnosis-text').textContent = "Connection Error";
    }
}

// --- Event Listeners ---

// 1. Load Sample Button (Now fetches a random beat from the massive test set!)
document.getElementById('btn-sample').addEventListener('click', async () => {
    try {
        const response = await fetch('/random_beat');
        const data = await response.json();

        const signal = data.signal;
        document.getElementById('true-diagnosis-text').textContent = data.true_diagnosis;

        if (signal.length !== 360) {
            alert(`Expected 360 samples, got ${signal.length}`);
            return;
        }
        analyzeSignal(signal);
    } catch (err) {
        console.error(err);
        alert("Failed to load sample from backend.");
    }
});

// 2. Upload CSV File (Batch Processing)
document.getElementById('file-upload').addEventListener('change', async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    Toastify({
        text: "Processing Batch Dataset...",
        duration: 3000,
        gravity: "top", position: "center"
    }).showToast();

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/batch_predict', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();

        if (data.error) {
            alert(data.error);
            return;
        }

        // Populate table
        const tbody = document.getElementById('batch-table-body');
        tbody.innerHTML = '';
        data.results.forEach(res => {
            const tr = document.createElement('tr');

            // Highlight dangerous beats
            if (res.severity === 'danger') {
                tr.style.backgroundColor = 'rgba(255, 0, 0, 0.2)';
            }

            tr.innerHTML = `
                <td>${res.index}</td>
                <td style="color: var(--color-${res.severity}); font-weight: bold;">${res.diagnosis}</td>
                <td>${res.confidence.toFixed(1)}%</td>
                <td>${res.mc_dropout_uncertainty.toFixed(4)}</td>
                <td>${res.cluster_entropy.toFixed(4)}</td>
                <td>${res.is_uncertain ? '⚠️ YES' : 'NO'}</td>
            `;
            tbody.appendChild(tr);
        });

        // Show Modal
        document.getElementById('batch-modal').classList.remove('hidden');

    } catch (e) {
        console.error("Batch error:", e);
        alert("Failed to process batch.");
    }

    // Reset file input
    event.target.value = '';
});

// Close Modal
document.getElementById('close-modal').addEventListener('click', () => {
    document.getElementById('batch-modal').classList.add('hidden');
});

// --- Live Patient Monitor Simulation ---
let isLive = false;
let monitorInterval;
let aiInterval;
let signalBuffer = []; // Holds incoming data from API
let streamIndex = 0; // Tracks position in the raw WFDB file

async function fetchNextBeat() {
    try {
        const response = await fetch(`/patient_stream/${streamIndex}`);
        const data = await response.json();

        if (data.error) {
            console.error("Stream Error:", data.error);
            return;
        }

        signalBuffer.push(...data.signal);
        streamIndex += data.signal.length; // Advance the pointer for next fetch

        // Also keep track of the ground truth for UI
        document.getElementById('true-diagnosis-text').textContent = data.true_diagnosis;
    } catch (e) {
        console.error("Failed to fetch next beat:", e);
    }
}

document.getElementById('btn-live-start').addEventListener('click', async () => {
    isLive = true;
    streamIndex = 0; // Reset patient stream to start
    document.getElementById('btn-live-start').classList.add('hidden');
    document.getElementById('btn-live-stop').classList.remove('hidden');

    // Clear graph
    ecgChart.data.datasets[0].data = Array(360).fill(0);
    ecgChart.update();

    // Fetch initial beats to fill buffer
    await fetchNextBeat();

    // Animation Loop: Scroll left by 2 samples every 50ms (Slower medical pace)
    monitorInterval = setInterval(async () => {
        if (!isLive) return;

        if (signalBuffer.length < 360) {
            // Fetch more data in background if buffer is getting low
            fetchNextBeat();
        }

        if (signalBuffer.length >= 2) {
            // Take 2 samples from buffer (much slower than 5)
            const newSamples = signalBuffer.splice(0, 2);

            // Shift chart left by removing oldest 2 and adding newest 2
            ecgChart.data.datasets[0].data.splice(0, 2);
            ecgChart.data.datasets[0].data.push(...newSamples);
            ecgChart.update('none'); // Update without animation for buttery smooth scrolling
        }
    }, 50);

    // AI Loop: Send the current screen to AI every 1 second
    aiInterval = setInterval(() => {
        if (!isLive) return;
        // Copy the current 360 samples displayed on the screen
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
});
