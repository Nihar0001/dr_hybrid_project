import os
import sys
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash, session
from werkzeug.utils import secure_filename

# --- make 'src' importable when app runs from app/ ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import config
from src.infer import infer_image

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = "dr-secret"  # set your own


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/scanner", methods=["GET", "POST"])
def scanner():
    context = {        
        "pred": None,
        "severity": None,
        "risk": None,
        "prediction_label": None,
        "confidence": None,}
    if request.method == "POST":
        if "file" not in request.files:
            flash("No file uploaded.")
            return redirect(url_for("scanner"))
        f = request.files["file"]
        if f.filename == "":
            flash("No selected file.")
            return redirect(url_for("scanner"))
        if not allowed_file(f.filename):
            flash("Please upload a PNG/JPG image.")
            return redirect(url_for("scanner"))
        
        filename = secure_filename(f.filename)
        save_path = os.path.join(config.UPLOADS_DIR, filename)
        f.save(save_path)

        try:
            pred, proba, heatmap_path, *_ = infer_image(save_path)

            import numpy as np
            
            proba = np.array(proba)
            
            pred_index = int(np.argmax(proba))
            confidence = float(proba[pred_index]) * 100
            
            #  SECOND BEST CLASS (VERY IMPORTANT)
            second_index = int(np.argsort(proba)[-2])
            second_conf = float(proba[second_index]) * 100
            
            #  DECISION CORRECTION LOGIC
            if abs(proba[pred_index] - proba[second_index]) < 0.05:
                # If close → pick higher severity (safer for medical use)
                # pred_index = max(pred_index, second_index)
                # mark as uncertain but keep original prediction
                confidence = float(proba[pred_index]) * 100 * 0.9

            # Standardized class names
            classes = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']
            pred_name = classes[pred_index]

            # Description (keep yours if needed)
            pred_description = config.CLASS_DESCRIPTIONS[pred_index]

            # predicted_class = int(pred)
            #  Clean label mapping
            if pred_index == 0:
                prediction_label = "No Diabetic Retinopathy"
                severity_level = "None"
                # severity_level = "Healthy"
                risk = "Low"

            elif pred_index == 1:
                prediction_label = "Mild DR"
                severity_level = "Class 1"
                risk = "Moderate"

            elif pred_index == 2:
                prediction_label = "Moderate DR"
                severity_level = "Class 2"
                risk = "Moderate"

            elif pred_index == 3:
                prediction_label = "Severe DR"
                severity_level = "Severe"
                risk = "High"

            elif pred_index == 4:
                prediction_label = "Proliferative DR"
                severity_level = "Critical"
                risk = "High"
                
            # 🧠 Confidence threshold safeguard
            if confidence < 50:
                prediction_label += " (Low Confidence)"

            # ✅ Create scan entry

            from datetime import datetime
            scan_entry = {
                "prediction": prediction_label,
                "pred_name": pred_name,
                "confidence": round(confidence, 1),
                "severity": severity_level,
                "risk": risk,
                "timestamp": datetime.now().strftime("%d %b %H:%M")
            }
            # ✅ Initialize history if not exists
            if "scan_history" not in session:
                session["scan_history"] = []
            # ✅ Add latest scan to top
            history = session.get("scan_history", [])
            history.insert(0, scan_entry)
            # ✅ Keep only last 5 scans (optional)
            session["scan_history"] = history[:10]
            # ✅ Store latest separately (for main cards)
            session["last_result"] = scan_entry
            session.modified = True

            context.update({
                "pred": pred_index,
                "severity": severity_level,
                "confidence": round(confidence, 1),
                "description": pred_description,
                "pred_name": pred_name,
                "prediction_label": prediction_label,
                "proba": proba.tolist(),
                "class_names": config.CLASS_NAMES,
                "overlay_url": url_for("outputs_file", filename=os.path.basename(heatmap_path)),
                "uploaded_url": url_for("uploads_file", filename=filename),
                "risk": risk,
                "scan_done": True,
            })
        except Exception as e:
            flash(f"Inference error: {e}")
            return redirect(url_for("scanner"))
            

    return render_template("scanner.html", **context)


@app.route("/")
def index():
    session.clear()  # clears history + last result
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    if "initialized" not in session:
        session["initialized"] = True
    history = session.get("scan_history", []) or []
    # result = session.get("last_result", {}) or {}   
    result = session.get("last_result") or {}

    # Chart files
    cm = "model_accuracy_bar_chart.png"
    f1 = "normalized_cm_votingclassifier.png"
    accuracy = "model_accuracy_bar_chart.png"
    cm = "normalized_cm_votingclassifier.png"
    radar = "model_radar_chart.png"

    accuracy_exists = os.path.exists(os.path.join(config.OUTPUTS_DIR, accuracy))
    cm_exists = os.path.exists(os.path.join(config.OUTPUTS_DIR, cm))
    f1_exists = os.path.exists(os.path.join(config.OUTPUTS_DIR, f1))
    radar_exists = os.path.exists(os.path.join(config.OUTPUTS_DIR, radar))

    healthy_count = sum(
        1 for h in history 
        if h.get("severity") in ["None", "Healthy"]
        or "No DR" in h.get("prediction", "")
    )

    high_risk_count = sum(
        1 for h in history 
        if h.get("risk") == "High"
    )

    moderate_count = sum(
    1 for h in history 
    if h.get("risk") == "Moderate"
    )

    total_scans = len(history)

    if total_scans == 0:
        healthy_percent = moderate_percent = high_percent = 0
    else:
        healthy_percent = round((healthy_count / total_scans) * 100, 1)
        moderate_percent = round((moderate_count / total_scans) * 100, 1)
        high_percent = round((high_risk_count / total_scans) * 100, 1)   

    return render_template(
        "dashboard.html",
        prediction=result.get("pred_name") or result.get("prediction"),
        confidence=result.get("confidence"),
        severity=result.get("severity"),
        risk=result.get("risk"),
        history=history,
        healthy_count=healthy_count,
        moderate_count=moderate_count,
        high_risk_count=high_risk_count,
        healthy_percent=healthy_percent,
        moderate_percent=moderate_percent,
        high_percent=high_percent,
        accuracy_url=url_for("outputs_file", filename=f1) if f1_exists else None,
        cm_url=url_for("outputs_file", filename=cm) if cm_exists else None,
        radar_url=url_for("outputs_file", filename=radar) if radar_exists else None,
        )

@app.route("/outputs/<path:filename>")
def outputs_file(filename):
    # Serve anything from outputs (images, txt)
    return send_from_directory(config.OUTPUTS_DIR, filename)

@app.route("/uploads/<path:filename>")
def uploads_file(filename):
    os.makedirs(config.UPLOADS_DIR, exist_ok=True)
    return send_from_directory(config.UPLOADS_DIR, filename)


if __name__ == "__main__":
    # For direct python app/app.py runs
    port = int(os.environ.get("PORT", 5001))
    app.run(debug=True, host="0.0.0.0", port=port)
