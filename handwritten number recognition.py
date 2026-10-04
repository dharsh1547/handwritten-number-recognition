import os
import io
import base64
import joblib
import numpy as np
import scipy.ndimage
from PIL import Image
from flask import Flask, render_template, request, jsonify
import keras

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SVM_MODEL_PATH = os.path.join(BASE_DIR, "model", "model.joblib")
SVM_SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.joblib")
CNN_MODEL_PATH = os.path.join(BASE_DIR, "model", "mnist_cnn.keras")

# Global model caches
svm_model = None
svm_scaler = None
cnn_model = None
mnist_test_data = None

def load_resources():
    global svm_model, svm_scaler, cnn_model, mnist_test_data
    if svm_model is None and os.path.exists(SVM_MODEL_PATH):
        svm_model = joblib.load(SVM_MODEL_PATH)
        svm_scaler = joblib.load(SVM_SCALER_PATH)
    
    if cnn_model is None and os.path.exists(CNN_MODEL_PATH):
        cnn_model = keras.models.load_model(CNN_MODEL_PATH)

    if mnist_test_data is None:
        try:
            (_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
            mnist_test_data = (x_test, y_test)
        except Exception:
            pass

load_resources()

def segment_and_preprocess_digits(base64_str):
    """
    Finds and isolates multiple handwritten digits on the canvas.
    1. Extracts ink from RGBA/grayscale canvas.
    2. Uses connected-component labeling and morphological dilation
       to group strokes belonging to each digit.
    3. Sorts digits strictly from left to right.
    4. Applies LeCun Center-of-Mass normalization to each isolated digit.
    """
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]

    img_bytes = base64.b64decode(base64_str)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
    arr = np.array(img)

    r, g, b, a = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2], arr[:, :, 3]
    grayscale = (0.299 * r + 0.587 * g + 0.114 * b) * (a / 255.0)

    ink_mask = grayscale > 20
    if not np.any(ink_mask):
        return []

    # Morphological dilation: connect disconnected strokes within the same digit (e.g. '4', '5', crossbar of '7')
    # Use modest horizontal kernel so separate digits remain distinct
    struct = np.ones((5, 9), dtype=bool)
    dilated = scipy.ndimage.binary_dilation(ink_mask, structure=struct)

    labeled, num_features = scipy.ndimage.label(dilated)
    slices = scipy.ndimage.find_objects(labeled)

    raw_boxes = []
    for s in slices:
        ymin, ymax = s[0].start, s[0].stop
        xmin, xmax = s[1].start, s[1].stop
        height = ymax - ymin
        width = xmax - xmin
        # Filter out tiny noise specks
        if height > 12 and width > 6 and np.sum(ink_mask[ymin:ymax, xmin:xmax]) > 15:
            raw_boxes.append([xmin, ymin, xmax, ymax])

    if not raw_boxes:
        return []

    # Merge overlapping or vertically aligned boxes (e.g. dot or top stroke)
    merged_boxes = []
    raw_boxes.sort(key=lambda b: b[0]) # sort by xmin

    for box in raw_boxes:
        if not merged_boxes:
            merged_boxes.append(box)
        else:
            prev = merged_boxes[-1]
            # Check for horizontal overlap or very close proximity (< 10px) with vertical alignment
            overlap = min(prev[2], box[2]) - max(prev[0], box[0])
            gap = box[0] - prev[2]
            if overlap > 0 or gap < 6:
                # Merge into single box
                merged_boxes[-1] = [
                    min(prev[0], box[0]),
                    min(prev[1], box[1]),
                    max(prev[2], box[2]),
                    max(prev[3], box[3])
                ]
            else:
                merged_boxes.append(box)

    # Sort left to right
    merged_boxes.sort(key=lambda b: b[0])

    digit_items = []
    for xmin, ymin, xmax, ymax in merged_boxes:
        # Tight crop using actual ink inside the merged box
        sub_mask = ink_mask[ymin:ymax, xmin:xmax]
        if not np.any(sub_mask):
            continue
        
        sub_rows = np.any(sub_mask, axis=1)
        sub_cols = np.any(sub_mask, axis=0)
        tight_ymin = ymin + np.where(sub_rows)[0][0]
        tight_ymax = ymin + np.where(sub_rows)[0][-1] + 1
        tight_xmin = xmin + np.where(sub_cols)[0][0]
        tight_xmax = xmin + np.where(sub_cols)[0][-1] + 1

        crop = grayscale[tight_ymin:tight_ymax, tight_xmin:tight_xmax]
        h, w = crop.shape
        if h == 0 or w == 0:
            continue

        # Fit inside 20x20 box preserving aspect ratio
        if h > w:
            new_h = 20
            new_w = max(1, int(round((w * 20.0) / h)))
        else:
            new_w = 20
            new_h = max(1, int(round((h * 20.0) / w)))

        pil_crop = Image.fromarray(crop.astype(np.uint8)).resize((new_w, new_h), Image.Resampling.BICUBIC)
        crop_arr = np.array(pil_crop, dtype=np.float32)

        # Place inside 28x28 frame
        img28 = np.zeros((28, 28), dtype=np.float32)
        top = (28 - new_h) // 2
        left = (28 - new_w) // 2
        img28[top:top+new_h, left:left+new_w] = crop_arr

        # Center of mass translation
        cy, cx = scipy.ndimage.center_of_mass(img28)
        if not np.isnan(cy) and not np.isnan(cx):
            shift_y = int(round(14.0 - cy))
            shift_x = int(round(14.0 - cx))
            img28 = scipy.ndimage.shift(img28, (shift_y, shift_x), mode='constant', cval=0.0)

        img28 = np.clip(img28, 0, 255) / 255.0

        # Downsample to 8x8 for SVM
        pil_8x8 = Image.fromarray((img28 * 255.0).astype(np.uint8)).resize((8, 8), Image.Resampling.BICUBIC)
        vec_8x8 = (np.array(pil_8x8, dtype=np.float32) / 255.0) * 16.0

        digit_items.append({
            "box": [int(tight_xmin), int(tight_ymin), int(tight_xmax), int(tight_ymax)],
            "img28": img28,
            "vec8": vec_8x8
        })

    return digit_items

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    image_data = data.get("image")
    model_type = data.get("model_type", "cnn").lower()

    if not image_data:
        return jsonify({"error": "No image data provided"}), 400

    load_resources()

    try:
        digit_items = segment_and_preprocess_digits(image_data)
        if not digit_items:
            return jsonify({
                "empty": True,
                "message": "Canvas is clear. Draw one or multiple digits to recognize!"
            })

        recognized_digits = []
        full_number_chars = []
        confidences = []

        for idx, item in enumerate(digit_items):
            if model_type == "svm" and svm_model is not None and svm_scaler is not None:
                scaled = svm_scaler.transform(item["vec8"].reshape(1, -1))
                pred_digit = int(svm_model.predict(scaled)[0])
                probs = svm_model.predict_proba(scaled)[0].tolist()
                used_model = "Support Vector Machine (SVC)"
            else:
                input_tensor = item["img28"][None, ..., None]
                preds = cnn_model.predict(input_tensor, verbose=0)[0]
                pred_digit = int(np.argmax(preds))
                probs = preds.tolist()
                used_model = "Convolutional Neural Network (CNN)"

            conf = float(probs[pred_digit])
            confidences.append(conf)
            full_number_chars.append(str(pred_digit))

            recognized_digits.append({
                "index": idx + 1,
                "digit": pred_digit,
                "confidence": round(conf * 100, 1),
                "probabilities": [round(p * 100, 1) for p in probs],
                "bounding_box": item["box"],
                "matrix_28x28": (item["img28"] * 255).round(0).astype(int).tolist()
            })

        full_number = "".join(full_number_chars)
        avg_confidence = round(float(np.mean(confidences)) * 100, 1)

        return jsonify({
            "empty": False,
            "full_number": full_number,
            "total_digits": len(recognized_digits),
            "average_confidence": avg_confidence,
            "model_used": used_model,
            "digits": recognized_digits,
            # For backward compatibility with single digit inspector
            "predicted_digit": full_number,
            "confidence": avg_confidence,
            "matrix_28x28": recognized_digits[0]["matrix_28x28"],
            "probabilities": recognized_digits[0]["probabilities"]
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/preset/<path:sequence>")
def get_preset(sequence):
    """
    Returns real MNIST samples for single or multi-digit sequences (e.g. '42', '2026', '7').
    Stitches them side-by-side into a wide canvas image.
    """
    load_resources()
    if mnist_test_data is None:
        return jsonify({"error": "Dataset not loaded"}), 500

    x_test, y_test = mnist_test_data

    # Validate that sequence is all digits
    if not sequence.isdigit():
        return jsonify({"error": "Preset sequence must be numeric"}), 400

    digits = [int(ch) for ch in sequence]
    rendered_digits = []

    for d in digits:
        indices = np.where(y_test == d)[0]
        if len(indices) == 0:
            continue
        idx = int(np.random.choice(indices))
        rendered_digits.append(x_test[idx])

    if not rendered_digits:
        return jsonify({"error": "Could not fetch samples"}), 404

    # Stitch into a composite 280x(540) canvas array
    num_digits = len(rendered_digits)
    canvas_w = 540
    canvas_h = 260
    slot_w = canvas_w // max(1, num_digits)

    composite_img = Image.new("L", (canvas_w, canvas_h), 0)

    for i, digit_arr in enumerate(rendered_digits):
        pil_d = Image.fromarray(digit_arr)
        # Scale to fit nicely in slot (e.g. 140x140)
        target_size = min(150, int(slot_w * 0.75), int(canvas_h * 0.75))
        pil_resized = pil_d.resize((target_size, target_size), Image.Resampling.BICUBIC)
        
        offset_x = i * slot_w + (slot_w - target_size) // 2
        offset_y = (canvas_h - target_size) // 2
        composite_img.paste(pil_resized, (offset_x, offset_y))

    # Convert to base64
    buf = io.BytesIO()
    composite_img.save(buf, format="PNG")
    b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

    return jsonify({
        "sequence": sequence,
        "total_digits": num_digits,
        "image": b64
    })

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
