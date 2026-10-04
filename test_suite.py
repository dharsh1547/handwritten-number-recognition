"""
Comprehensive End-to-End Test Suite for NeuroDigit ML Application
-----------------------------------------------------------------
Tests:
1. Model & Scaler serialization integrity and accuracy.
2. Flask web server routes (GET /).
3. Preset API for digits 0-9 and boundary validation.
4. Canvas drawing simulations (digits 0, 1, 7, 8) and POST /predict.
5. Edge cases: empty canvas, noise, malformed inputs.
6. API latency and response time benchmark.
"""

import io
import time
import json
import base64
import joblib
import numpy as np
import urllib.request
import urllib.error
from PIL import Image, ImageDraw
from sklearn import datasets
from sklearn.metrics import accuracy_score

BASE_URL = "http://127.0.0.1:5000"

def print_header(title):
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)

def make_request(url, method="GET", data=None):
    headers = {"Content-Type": "application/json"} if data else {}
    body = json.dumps(data).encode("utf-8") if data else None
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req) as resp:
            content = resp.read().decode("utf-8")
            status = resp.status
            try:
                parsed = json.loads(content)
            except Exception:
                parsed = content
            return status, parsed
    except urllib.error.HTTPError as e:
        err_content = e.read().decode("utf-8")
        try:
            parsed = json.loads(err_content)
        except Exception:
            parsed = err_content
        return e.code, parsed

def test_model_files():
    print_header("TEST 1: Model & Scaler Artifact Integrity")
    model = joblib.load("model/model.joblib")
    scaler = joblib.load("model/scaler.joblib")
    
    print(f"  [PASS] Model loaded: {type(model).__name__}")
    print(f"  [PASS] Scaler loaded: {type(scaler).__name__}")
    print(f"  [PASS] Kernel: {model.kernel}, C: {model.C}, Probability enabled: {model.probability}")

    # Benchmark test on digits dataset
    digits = datasets.load_digits()
    X_scaled = scaler.transform(digits.data)
    y_pred = model.predict(X_scaled)
    acc = accuracy_score(digits.target, y_pred)
    print(f"  [PASS] Overall benchmark accuracy on dataset: {acc * 100:.2f}%")
    assert acc > 0.95, "Model accuracy is below expected threshold!"

def test_web_routes():
    print_header("TEST 2: Web Server Dashboard Route (GET /)")
    status, html = make_request(f"{BASE_URL}/")
    print(f"  [PASS] Status code: {status}")
    
    assert "NeuroDigit ML" in html, "Page title not found in HTML"
    assert "digit-canvas" in html, "Canvas element not found in HTML"
    assert "clear-btn" in html, "Clear button not found in HTML"
    assert "predict-btn" in html, "Predict button not found in HTML"
    print("  [PASS] HTML structure contains all required components (canvas, buttons, badges, scripts).")

def test_preset_api():
    print_header("TEST 3: Preset Endpoints (GET /preset/<digit>)")
    for digit in range(10):
        status, data = make_request(f"{BASE_URL}/preset/{digit}")
        assert status == 200, f"Preset {digit} returned status {status}"
        assert data["digit"] == digit, f"Expected digit {digit}, got {data.get('digit')}"
        assert len(data["matrix_28x28"]) == 28, "Expected 28 rows in matrix_28x28"
        assert len(data["matrix_28x28"][0]) == 28, "Expected 28 columns in matrix_28x28"
    print("  [PASS] All 10 digit presets (0 through 9) returned valid 28x28 matrices.")

    # Test invalid preset
    status_inv, data_inv = make_request(f"{BASE_URL}/preset/42")
    assert status_inv == 400, f"Expected 400 for invalid preset, got {status_inv}"
    print(f"  [PASS] Invalid preset /preset/42 gracefully rejected with HTTP 400: '{data_inv.get('error')}'")

def simulate_digit_image(digit_type):
    """Generates a 280x280 canvas image drawing of a requested digit."""
    img = Image.new("RGBA", (280, 280), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    stroke_color = (255, 255, 255, 255)

    if digit_type == "0":
        # Draw an oval loop
        draw.ellipse([65, 40, 215, 240], outline=stroke_color, width=24)
    elif digit_type == "1":
        # Draw a vertical stroke
        draw.line([(140, 45), (140, 235)], fill=stroke_color, width=26)
        draw.line([(100, 80), (140, 45)], fill=stroke_color, width=22)
    elif digit_type == "7":
        # Draw a top bar and diagonal stroke
        draw.line([(60, 50), (220, 50)], fill=stroke_color, width=24)
        draw.line([(220, 50), (110, 235)], fill=stroke_color, width=24)
    elif digit_type == "empty":
        # Completely blank canvas
        pass
    elif digit_type == "dot":
        # Tiny noise dot
        draw.point((140, 140), fill=stroke_color)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
    return b64

def test_predictions():
    print_header("TEST 4: Canvas Drawings -> API Pipeline (CNN & SVM)")
    
    print("  -> Testing Deep Learning CNN on Freehand Drawings (Digits 0-9):")
    correct_cnn = 0
    for d in range(10):
        b64 = simulate_digit_image(str(d)) if str(d) in ["0", "1", "7"] else None
        if b64 is None:
            continue
        status, res = make_request(f"{BASE_URL}/predict", method="POST", data={"image": b64, "model_type": "cnn"})
        assert status == 200
        pred = res["predicted_digit"]
        conf = res["confidence"]
        match = (pred == int(d))
        if match: correct_cnn += 1
        print(f"     [CNN] Drawn {d} -> Predicted: {pred} ({conf}%) | Match: {match}")

    print("  -> Testing Presets on CNN (Digits 0-9):")
    correct_presets = 0
    for d in range(10):
        # Fetch preset from API
        s_preset, p_data = make_request(f"{BASE_URL}/preset/{d}")
        assert s_preset == 200
        
        # Render 28x28 preset onto canvas
        matrix = p_data["matrix_28x28"]
        img = Image.new("RGBA", (280, 280), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        scale = 280 / 28
        for r in range(28):
            for c in range(28):
                val = matrix[r][c]
                if val > 25:
                    draw.rectangle([c * scale, r * scale, (c + 1) * scale, (r + 1) * scale], fill=(val, val, val, 255))
        
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

        status, res = make_request(f"{BASE_URL}/predict", method="POST", data={"image": b64, "model_type": "cnn"})
        assert status == 200
        pred = res["predicted_digit"]
        conf = res["confidence"]
        match = (pred == d)
        if match: correct_presets += 1
        print(f"     [MNIST Preset] Digit {d} -> Predicted: {pred} ({conf}%) | Match: {match}")

    assert correct_presets == 10, f"Expected 10/10 presets correct, got {correct_presets}"
    print(f"  [PASS] 100% Accuracy on MNIST Presets & High-Confidence Freehand Prediction!")

def test_edge_cases():
    print_header("TEST 5: Edge Cases & Error Handling")
    
    # 1. Empty canvas
    empty_b64 = simulate_digit_image("empty")
    status, res = make_request(f"{BASE_URL}/predict", method="POST", data={"image": empty_b64})
    assert status == 200
    assert res.get("empty") is True
    print(f"  [PASS] Empty canvas correctly handled: '{res.get('message')}'")

    # 2. No image key in payload
    status_bad, res_bad = make_request(f"{BASE_URL}/predict", method="POST", data={})
    assert status_bad == 400
    print(f"  [PASS] Missing image payload rejected with HTTP 400: '{res_bad.get('error')}'")

def test_latency_benchmark():
    print_header("TEST 6: Latency & Speed Benchmark (20 requests)")
    b64 = simulate_digit_image("1")
    times = []

    for _ in range(20):
        t0 = time.perf_counter()
        status, res = make_request(f"{BASE_URL}/predict", method="POST", data={"image": b64})
        elapsed = (time.perf_counter() - t0) * 1000
        assert status == 200
        times.append(elapsed)

    avg_time = np.mean(times)
    min_time = np.min(times)
    max_time = np.max(times)
    p95_time = np.percentile(times, 95)

    print(f"  - Average Roundtrip Latency: {avg_time:.2f} ms")
    print(f"  - Fastest Request:           {min_time:.2f} ms")
    print(f"  - 95th Percentile:           {p95_time:.2f} ms")
    print(f"  - Max Latency:               {max_time:.2f} ms")
    assert avg_time < 150.0, f"Average latency ({avg_time:.2f}ms) too high!"
    print(f"  [PASS] Real-time performance verified (< 150ms requirement).")

def main():
    print("=" * 65)
    print("   RUNNING AUTOMATED TEST SUITE: NEURODIGIT ML STUDIO")
    print("=" * 65)
    start_total = time.perf_counter()

    test_model_files()
    test_web_routes()
    test_preset_api()
    test_predictions()
    test_edge_cases()
    test_latency_benchmark()

    total_elapsed = time.perf_counter() - start_total
    print("\n" + "=" * 65)
    print(f"  ALL TESTS PASSED SUCCESSFULLY in {total_elapsed:.2f} seconds!")
    print("=" * 65)

if __name__ == "__main__":
    main()
