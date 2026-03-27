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
    context = {}
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

            # Calculate confidence as percentage of predicted class
            confidence = float(proba[pred]) * 100
            pred_class_name = config.CLASS_NAMES[int(pred)]
            pred_description = config.CLASS_DESCRIPTIONS[int(pred)]

            predicted_class = int(pred)
            if predicted_class == 0:
                prediction_label = "No Diabetic Retinopathy Detected"
                severity_level = "None"
            else:
                prediction_label = "Diabetic Retinopathy Detected"
                severity_level = f"Class {predicted_class}"
            if predicted_class == 0:
                    risk = "Low"
            elif predicted_class <= 2:
                    risk = "Moderate"
            else:
                    risk = "High"

            session["last_result"] = {
                "prediction": prediction_label,
                "confidence": round(confidence, 1),
                "severity": severity_level,
                "risk": risk,   # ✅ ADD THIS LINE
                "cm_url": url_for("outputs_file", filename="model_accuracy_bar_chart.png"),
                "f1_url": url_for("outputs_file", filename="normalized_cm_votingclassifier.png"),
                "radar_url": url_for("outputs_file", filename="model_radar_chart.png")
            }

            context.update({
                "pred": int(pred),
                "prediction_label": prediction_label,
                "severity": severity_level,
                "confidence": round(confidence, 1),
                "description": pred_description,
                "pred_name": pred_class_name,
                "proba": proba.tolist(),
                "class_names": config.CLASS_NAMES,
                "overlay_url": url_for("outputs_file", filename=os.path.basename(heatmap_path)),
                "uploaded_url": url_for("uploads_file", filename=filename),
                "risk": risk,
            })
        except Exception as e:
            flash(f"Inference error: {e}")
            return redirect(url_for("scanner"))

    return render_template("scanner.html", **context)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    result = session.get("last_result", None)

    # Chart files
    cm = "model_accuracy_bar_chart.png"
    f1 = "normalized_cm_votingclassifier.png"
    radar = "model_radar_chart.png"

    cm_exists = os.path.exists(os.path.join(config.OUTPUTS_DIR, cm))
    f1_exists = os.path.exists(os.path.join(config.OUTPUTS_DIR, f1))
    radar_exists = os.path.exists(os.path.join(config.OUTPUTS_DIR, radar))

    return render_template(
        "dashboard.html",

        # 🔥 Dynamic scan data
        prediction=result.get("prediction") if result else None,
        confidence=result.get("confidence") if result else None,
        severity=result.get("severity") if result else None,
        risk=result["risk"] if result else None,

        # 📊 Charts
        cm_url=url_for("outputs_file", filename=cm) if cm_exists else None,
        f1_url=url_for("outputs_file", filename=f1) if f1_exists else None,
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
