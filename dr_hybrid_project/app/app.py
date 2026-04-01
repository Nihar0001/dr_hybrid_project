import os
import sys
import json
from datetime import datetime
from io import BytesIO
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, send_file, flash, session
from werkzeug.utils import secure_filename
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# --- make 'src' importable when app runs from app/ ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import config
from src.infer import infer_image

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = "dr-secret"  # set your own
DATA_FILE = os.path.join(config.PROJECT_ROOT, "data", "patient_scans.json")


def load_data():
    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
    if not os.path.exists(DATA_FILE):
        data = {"patients": {}}
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return data
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        data = {"patients": {}}
    if "patients" not in data or not isinstance(data["patients"], dict):
        data["patients"] = {}
    return data


def save_data(data):
    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/scanner", methods=["GET", "POST"])
def scanner():
    context = {
        "pred": None,
        "severity": None,
        "risk": None,
        "prediction_label": None,
        "confidence": None,
        "decision": None,
        "recommendation": None,
    }
    if request.method == "POST":
        patient_name = request.form.get("patient_name", "").strip()
        if not patient_name:
            flash("Patient name is required.")
            return redirect(url_for("scanner"))

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
                severity_level = "Healthy"
                risk = "Low"
                decision = "Healthy Retina"
                recommendation = "Routine check-up recommended"
            else:
                prediction_label = "Diabetic Retinopathy Detected"
                severity_level = pred_class_name # e.g. 'Mild', 'Moderate', etc.
                if predicted_class == 1:
                    risk = "Mild"
                    decision = "Early DR Signs"
                    recommendation = "Monitor regularly and consult specialist"
                elif predicted_class == 2:
                    risk = "Moderate"
                    decision = "Attention Required"
                    recommendation = "Clinical evaluation advised soon"
                elif predicted_class == 3:
                    risk = "High"
                    decision = "Urgent Specialist Review"
                    recommendation = "Urgent ophthalmology consultation is recommended"
                else:
                    risk = "Critical"
                    decision = "Immediate Intervention Needed"
                    recommendation = "Immediate medical attention required"

            timestamp = datetime.utcnow().isoformat()

            # Persistent JSON storage grouped by patient
            data = load_data()
            patients = data.setdefault("patients", {})
            if patient_name not in patients:
                patients[patient_name] = []
            patients[patient_name].append({
                "date": timestamp,
                "result": pred_class_name,
                "severity": severity_level,
                "risk": risk,
                "decision": decision,
                "recommendation": recommendation,
                "confidence": round(confidence, 1),
                "image_path": filename,
                "overlay_path": os.path.basename(heatmap_path),
            })
            save_data(data)

            # ✅ Create scan entry
            scan_entry = {
                "patient_name": patient_name,
                "prediction": prediction_label,
                "confidence": round(confidence, 1),
                "severity": severity_level,
                "risk": risk,
                "decision": decision,
                "recommendation": recommendation,
                "timestamp": timestamp,
                "result": pred_class_name,
                "image_path": filename,
                "overlay_path": os.path.basename(heatmap_path),
            }
            # ✅ Initialize history if not exists
            if "scan_history" not in session:
                session["scan_history"] = []
            # ✅ Add latest scan to top
            history = session["scan_history"]
            history.insert(0, scan_entry)
            # ✅ Keep only last 5 scans (optional)
            session["scan_history"] = history[:5]
            # ✅ Store latest separately (for main cards)
            session["last_result"] = scan_entry

            context.update({
                "patient_name": patient_name,
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
                "decision": decision,
                "recommendation": recommendation,
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
    data = load_data()
    patients = data.get("patients", {})

    patient_summaries = []
    history = []
    for patient_name, scans in patients.items():
        if not isinstance(scans, list) or not scans:
            continue
        latest_scan = scans[-1]
        patient_summaries.append({
            "name": patient_name,
            "latest_result": latest_scan.get("result") or latest_scan.get("prediction") or "N/A",
            "latest_severity": latest_scan.get("severity") or "N/A",
            "latest_confidence": latest_scan.get("confidence"),
            "scan_count": len(scans),
        })
        for scan in scans:
            history.append({
                "patient_name": patient_name,
                "prediction": scan.get("result") or scan.get("prediction") or "N/A",
                "confidence": scan.get("confidence"),
                "severity": scan.get("severity"),
                "risk": scan.get("risk"),
                "timestamp": scan.get("date") or scan.get("timestamp") or "Recently",
            })

    history.sort(key=lambda s: s.get("timestamp") or "", reverse=True)
    patient_summaries.sort(key=lambda p: p["name"].lower())

    if history:
        latest = history[0]
        result = {
            "prediction": latest.get("prediction"),
            "confidence": latest.get("confidence"),
            "severity": latest.get("severity"),
            "risk": latest.get("risk"),
        }
    else:
        result = session.get("last_result", None)

    # Chart files
    accuracy = "model_accuracy_bar_chart.png"
    cm = "normalized_cm_votingclassifier.png"
    radar = "model_radar_chart.png"

    accuracy_exists = os.path.exists(os.path.join(config.OUTPUTS_DIR, accuracy))
    cm_exists = os.path.exists(os.path.join(config.OUTPUTS_DIR, cm))
    radar_exists = os.path.exists(os.path.join(config.OUTPUTS_DIR, radar))

    return render_template(
        "dashboard.html",

        # 🔥 Dynamic scan data
        prediction=result.get("prediction") if result else None,
        confidence=result.get("confidence") if result else None,
        severity=result.get("severity") if result else None,
        risk=result.get("risk") if result else None,

        history=history if history else session.get("scan_history", []),
        patient_summaries=patient_summaries,

        # 📊 Charts
        accuracy_url=url_for("outputs_file", filename=accuracy) if accuracy_exists else None,
        cm_url=url_for("outputs_file", filename=cm) if cm_exists else None,
        radar_url=url_for("outputs_file", filename=radar) if radar_exists else None,
    )


@app.route("/download_report/<path:patient_name>")
def download_report(patient_name):
    data = load_data()
    scans = data.get("patients", {}).get(patient_name, [])
    if not scans:
        flash("No report data found for this patient.")
        return redirect(url_for("dashboard"))

    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    _, height = A4

    y = height - 40
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(40, y, "RetinaScan AI - Patient Report")
    y -= 28

    pdf.setFont("Helvetica", 11)
    pdf.drawString(40, y, f"Patient Name: {patient_name}")
    y -= 18
    pdf.drawString(40, y, f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    y -= 28

    latest = scans[-1]
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(40, y, "Latest Scan")
    y -= 18
    pdf.setFont("Helvetica", 11)
    pdf.drawString(40, y, f"Date: {latest.get('date', 'N/A')}")
    y -= 16
    pdf.drawString(40, y, f"Result: {latest.get('result', 'N/A')}")
    y -= 16
    pdf.drawString(40, y, f"Severity: {latest.get('severity', 'N/A')}")
    y -= 16
    pdf.drawString(40, y, f"Confidence: {latest.get('confidence', 'N/A')}%")
    y -= 24

    image_name = latest.get("image_path")
    if image_name:
        image_file = os.path.join(config.UPLOADS_DIR, os.path.basename(image_name))
        if os.path.exists(image_file):
            try:
                pdf.drawImage(ImageReader(image_file), 40, y - 180, width=170, height=170, preserveAspectRatio=True, mask='auto')
                y -= 190
            except Exception:
                pass

    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(40, y, "Scan History")
    y -= 18
    pdf.setFont("Helvetica", 10)

    for idx, scan in enumerate(reversed(scans), start=1):
        line = (
            f"{idx}. {scan.get('date', 'N/A')} | Result: {scan.get('result', 'N/A')} | "
            f"Severity: {scan.get('severity', 'N/A')} | Confidence: {scan.get('confidence', 'N/A')}%"
        )
        pdf.drawString(40, y, line[:120])
        y -= 14
        if y < 60:
            pdf.showPage()
            y = height - 40
            pdf.setFont("Helvetica", 10)

    pdf.save()
    buffer.seek(0)
    safe_name = secure_filename(patient_name) or "patient"
    return send_file(
        buffer,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"{safe_name}_report.pdf",
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
