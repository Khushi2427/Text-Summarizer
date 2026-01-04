from flask import Flask, render_template, request
import os
from textSummarizer.pipeline.prediction import PredictionPipeline

app = Flask(__name__)

# 🔥 Load model ONCE (very important for Render)
pipeline = PredictionPipeline()

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/train", methods=["POST"])
def train():
    try:
        os.system("python main.py")
        return "✅ Training completed successfully!"
    except Exception as e:
        return f"❌ Error occurred: {e}"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        text = request.form["text"]
        summary = pipeline.predict(text)

        return render_template(
            "index.html",
            original_text=text,
            summary=summary
        )
    except Exception as e:
        return f"❌ Error occurred: {e}"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
