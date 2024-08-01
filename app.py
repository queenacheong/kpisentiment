from sentence_transformers import SentenceTransformer, util
from flask import Flask, request, jsonify, send_from_directory
import json

# Load KPIs and descriptions
with open("kpis.json") as f:
    kpi_data = json.load(f)

# Load sentence transformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

app = Flask(__name__, static_folder='public')

# Function to analyze text against KPIs
def analyze_kpis(text):
    text_embedding = model.encode(text, convert_to_tensor=True)
    results = []
    for kpi in kpi_data["KPIs"]:
        kpi_embedding = model.encode(kpi["description"], convert_to_tensor=True)
        relevance = util.pytorch_cos_sim(text_embedding, kpi_embedding).item()
        results.append({
            "kpi": kpi["name"],
            "relevance": relevance
        })
    return results

@app.route("/", methods=["POST"])
def analyze():
    try:
        data = request.json
        text = data["text"]
        kpi_results = analyze_kpis(text)
        return jsonify({"kpiResults": kpi_results})
    except Exception as e:
        print(e)
        return jsonify({"error": "Error performing analysis"}), 500

@app.route("/", methods=["GET"])
def index():
    return send_from_directory('.', 'index.html')

@app.route("/public/<path:path>")
def send_public(path):
    return send_from_directory('public', path)

if __name__ == "__main__":
    app.run(port=5500)
