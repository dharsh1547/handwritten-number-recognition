/**
 * NeuroDigit ML - High Precision Handwritten Digit Recognition Engine
 * Powered by Deep Learning CNN & Classical SVM on MNIST
 */

document.addEventListener('DOMContentLoaded', () => {
    // Canvas Elements
    const canvas = document.getElementById('digit-canvas');
    const ctx = canvas.getContext('2d');
    const canvasOverlay = document.getElementById('canvas-overlay');
    const visionCanvas = document.getElementById('vision-canvas');
    const visionCtx = visionCanvas.getContext('2d');
    
    // Tools & Controls
    const brushSlider = document.getElementById('brush-size');
    const brushVal = document.getElementById('brush-val');
    const toolPen = document.getElementById('tool-pen');
    const toolEraser = document.getElementById('tool-eraser');
    const clearBtn = document.getElementById('clear-btn');
    const predictBtn = document.getElementById('predict-btn');
    const liveModeToggle = document.getElementById('live-mode-toggle');
    const presetButtons = document.querySelectorAll('.btn-preset');

    // Model Switcher Elements
    const modelCnnBtn = document.getElementById('model-cnn-btn');
    const modelSvmBtn = document.getElementById('model-svm-btn');
    const activeModelTitle = document.getElementById('active-model-title');
    const specArch = document.getElementById('spec-arch');
    const specInput = document.getElementById('spec-input');
    const datasetBadge = document.getElementById('dataset-badge');

    // Display & Analytics Elements
    const predictedDigitDisplay = document.getElementById('predicted-digit-display');
    const confidenceBadge = document.getElementById('confidence-badge');
    const confidenceVal = document.getElementById('confidence-val');
    const statusText = document.getElementById('status-text');
    const probList = document.getElementById('prob-list');
    const inferenceTime = document.getElementById('inference-time');

    // App State
    let selectedModel = 'cnn'; // 'cnn' | 'svm'
    let isDrawing = false;
    let currentTool = 'pen'; // 'pen' | 'eraser'
    let brushSize = parseInt(brushSlider.value, 10);
    let points = [];
    let debounceTimer = null;
    let hasDrawn = false;

    // HiDPI Scaling for main drawing canvas
    const dpr = window.devicePixelRatio || 1;
    canvas.width = 280 * dpr;
    canvas.height = 280 * dpr;
    ctx.scale(dpr, dpr);
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    // Initialize 28x28 Vision Canvas
    visionCanvas.width = 28;
    visionCanvas.height = 28;
    visionCtx.imageSmoothingEnabled = false;

    // Initialize Probability Bars (Digits 0-9)
    function initProbabilityBars() {
        probList.innerHTML = '';
        for (let i = 0; i < 10; i++) {
            const row = document.createElement('div');
            row.className = 'prob-row';
            row.id = `prob-row-${i}`;
            row.innerHTML = `
                <span class="prob-digit" id="digit-label-${i}">${i}</span>
                <div class="prob-bar-track">
                    <div class="prob-bar-fill" id="prob-fill-${i}"></div>
                </div>
                <span class="prob-val" id="prob-val-${i}">0.0%</span>
            `;
            probList.appendChild(row);
        }
    }

    initProbabilityBars();

    // Model Switching Handlers
    modelCnnBtn.addEventListener('click', () => {
        selectedModel = 'cnn';
        modelCnnBtn.classList.add('active');
        modelSvmBtn.classList.remove('active');
        activeModelTitle.textContent = 'Convolutional Neural Network (CNN)';
        specArch.textContent = 'Conv2D (32/64) + MaxPool + Dropout';
        specInput.textContent = '28x28 Grayscale (Center-of-Mass)';
        datasetBadge.textContent = 'Dataset: MNIST (60,000)';
        if (hasDrawn) triggerRecognition();
    });

    modelSvmBtn.addEventListener('click', () => {
        selectedModel = 'svm';
        modelSvmBtn.classList.add('active');
        modelCnnBtn.classList.remove('active');
        activeModelTitle.textContent = 'Support Vector Machine (SVC)';
        specArch.textContent = 'SVC (RBF Kernel, C=10.0)';
        specInput.textContent = '8x8 Scaled Vector (64 Dim)';
        datasetBadge.textContent = 'Dataset: Digits (1,797)';
        if (hasDrawn) triggerRecognition();
    });

    // Get pointer coordinates relative to canvas
    function getCanvasCoordinates(e) {
        const bounds = canvas.getBoundingClientRect();
        let clientX = e.clientX;
        let clientY = e.clientY;

        if (e.touches && e.touches.length > 0) {
            clientX = e.touches[0].clientX;
            clientY = e.touches[0].clientY;
        }

        return {
            x: (clientX - bounds.left),
            y: (clientY - bounds.top)
        };
    }

    // Drawing Handlers
    function startDrawing(e) {
        e.preventDefault();
        isDrawing = true;
        hasDrawn = true;
        canvasOverlay.classList.add('hidden');

        const pt = getCanvasCoordinates(e);
        points = [pt];

        ctx.beginPath();
        ctx.moveTo(pt.x, pt.y);
        ctx.strokeStyle = (currentTool === 'eraser') ? '#000000' : '#ffffff';
        ctx.lineWidth = (currentTool === 'eraser') ? brushSize * 1.6 : brushSize;
    }

    function draw(e) {
        if (!isDrawing) return;
        e.preventDefault();

        const pt = getCanvasCoordinates(e);
        points.push(pt);

        if (points.length >= 3) {
            const xc = (points[points.length - 2].x + points[points.length - 1].x) / 2;
            const yc = (points[points.length - 2].y + points[points.length - 1].y) / 2;
            ctx.quadraticCurveTo(points[points.length - 2].x, points[points.length - 2].y, xc, yc);
            ctx.stroke();
        } else {
            ctx.lineTo(pt.x, pt.y);
            ctx.stroke();
        }

        // Live Recognition (debounce trigger)
        if (liveModeToggle.checked) {
            clearTimeout(debounceTimer);
            debounceTimer = setTimeout(triggerRecognition, 200);
        }
    }

    function stopDrawing(e) {
        if (!isDrawing) return;
        isDrawing = false;
        ctx.closePath();

        if (liveModeToggle.checked) {
            clearTimeout(debounceTimer);
            debounceTimer = setTimeout(triggerRecognition, 120);
        }
    }

    // Canvas Mouse Listeners
    canvas.addEventListener('mousedown', startDrawing);
    window.addEventListener('mousemove', draw);
    window.addEventListener('mouseup', stopDrawing);

    // Canvas Touch Listeners
    canvas.addEventListener('touchstart', startDrawing, { passive: false });
    canvas.addEventListener('touchmove', draw, { passive: false });
    canvas.addEventListener('touchend', stopDrawing, { passive: false });

    // Brush Tool Controls
    brushSlider.addEventListener('input', (e) => {
        brushSize = parseInt(e.target.value, 10);
        brushVal.textContent = `${brushSize}px`;
    });

    toolPen.addEventListener('click', () => {
        currentTool = 'pen';
        toolPen.classList.add('active');
        toolEraser.classList.remove('active');
    });

    toolEraser.addEventListener('click', () => {
        currentTool = 'eraser';
        toolEraser.classList.add('active');
        toolPen.classList.remove('active');
    });

    // Clear Canvas
    function clearCanvas() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        visionCtx.clearRect(0, 0, 28, 28);
        hasDrawn = false;
        canvasOverlay.classList.remove('hidden');

        // Reset UI State
        predictedDigitDisplay.textContent = '-';
        predictedDigitDisplay.classList.remove('active');
        confidenceBadge.className = 'badge badge-waiting';
        confidenceBadge.textContent = 'Awaiting Input';
        confidenceVal.textContent = '--%';
        statusText.textContent = 'Draw any single digit (0–9) on the canvas to begin.';
        inferenceTime.textContent = '< 10 ms';

        // Clear Probabilities
        for (let i = 0; i < 10; i++) {
            const fill = document.getElementById(`prob-fill-${i}`);
            const val = document.getElementById(`prob-val-${i}`);
            const label = document.getElementById(`digit-label-${i}`);
            if (fill) {
                fill.style.width = '0%';
                fill.classList.remove('top');
            }
            if (val) {
                val.textContent = '0.0%';
                val.classList.remove('top');
            }
            if (label) label.classList.remove('top');
        }
    }

    clearBtn.addEventListener('click', clearCanvas);
    predictBtn.addEventListener('click', triggerRecognition);

    // Send prediction request to Flask API
    async function triggerRecognition() {
        if (!hasDrawn) return;

        const startTime = performance.now();
        const dataUrl = canvas.toDataURL('image/png');

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    image: dataUrl,
                    model_type: selectedModel
                })
            });

            const data = await response.json();
            const elapsed = Math.round(performance.now() - startTime);
            inferenceTime.textContent = `${elapsed} ms`;

            if (data.empty) {
                confidenceBadge.className = 'badge badge-waiting';
                confidenceBadge.textContent = 'Empty';
                statusText.textContent = data.message;
                return;
            }

            if (data.error) {
                console.error(data.error);
                statusText.textContent = `Error: ${data.error}`;
                return;
            }

            renderResults(data);
        } catch (err) {
            console.error('Prediction API Error:', err);
        }
    }

    // Render results on UI
    function renderResults(data) {
        const topDigit = data.predicted_digit;
        const confidence = data.confidence;
        const probs = data.probabilities;
        const matrix28 = data.matrix_28x28;

        // 1. Hero Card
        predictedDigitDisplay.textContent = topDigit;
        predictedDigitDisplay.classList.add('active');
        confidenceVal.textContent = `${confidence.toFixed(1)}%`;

        if (confidence >= 80) {
            confidenceBadge.className = 'badge badge-confident';
            confidenceBadge.textContent = 'High Precision';
            statusText.textContent = `Recognized as digit ${topDigit} (${data.model_used || selectedModel.toUpperCase()}).`;
        } else if (confidence >= 50) {
            confidenceBadge.className = 'badge badge-moderate';
            confidenceBadge.textContent = 'Moderate';
            statusText.textContent = `Likely digit ${topDigit}. Consider refining stroke or switching to CNN.`;
        } else {
            confidenceBadge.className = 'badge badge-waiting';
            confidenceBadge.textContent = 'Uncertain';
            statusText.textContent = `Ambiguous digit structure. Try drawing with standard handwriting.`;
        }

        // 2. Render 28x28 Model Vision Canvas
        if (matrix28) {
            const imgData = visionCtx.createImageData(28, 28);
            let ptr = 0;
            for (let r = 0; r < 28; r++) {
                for (let c = 0; c < 28; c++) {
                    const intensity = matrix28[r][c];
                    // Tint slightly cyan/electric blue for ink
                    imgData.data[ptr]     = Math.round(intensity * 0.22); // R
                    imgData.data[ptr + 1] = Math.round(intensity * 0.74); // G
                    imgData.data[ptr + 2] = intensity;                   // B
                    imgData.data[ptr + 3] = 255;                         // Alpha
                    ptr += 4;
                }
            }
            visionCtx.putImageData(imgData, 0, 0);
        }

        // 3. Update Confidence Bars (Digits 0-9)
        for (let i = 0; i < 10; i++) {
            const p = probs[i] || 0.0;
            const fill = document.getElementById(`prob-fill-${i}`);
            const val = document.getElementById(`prob-val-${i}`);
            const label = document.getElementById(`digit-label-${i}`);

            if (fill) {
                fill.style.width = `${p}%`;
                if (i === topDigit) {
                    fill.classList.add('top');
                } else {
                    fill.classList.remove('top');
                }
            }

            if (val) {
                val.textContent = `${p.toFixed(1)}%`;
                if (i === topDigit) {
                    val.classList.add('top');
                } else {
                    val.classList.remove('top');
                }
            }

            if (label) {
                if (i === topDigit) {
                    label.classList.add('top');
                } else {
                    label.classList.remove('top');
                }
            }
        }
    }

    // Quick Sample Presets Handler (Loads real MNIST sample)
    presetButtons.forEach(btn => {
        btn.addEventListener('click', async () => {
            const digit = btn.dataset.digit;
            try {
                const res = await fetch(`/preset/${digit}`);
                const data = await res.json();
                
                if (data.matrix_28x28) {
                    clearCanvas();
                    hasDrawn = true;
                    canvasOverlay.classList.add('hidden');

                    // Draw the 28x28 MNIST preset scaled onto the 280x280 canvas
                    const matrix = data.matrix_28x28;
                    const scale = 280 / 28;

                    for (let r = 0; r < 28; r++) {
                        for (let c = 0; c < 28; c++) {
                            const intensity = matrix[r][c];
                            if (intensity > 25) {
                                ctx.fillStyle = `rgb(${intensity}, ${intensity}, ${intensity})`;
                                ctx.fillRect(c * scale, r * scale, scale, scale);
                            }
                        }
                    }

                    // Trigger prediction
                    triggerRecognition();
                }
            } catch (err) {
                console.error('Error loading preset:', err);
            }
        });
    });
});
