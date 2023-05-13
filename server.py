from flask import Flask, request, jsonify
from flask_cors import CORS

from joblib import load
from model import predict, getUniqueValues

import warnings

warnings.filterwarnings(
    action="ignore",
    message=".*Unverified HTTPS.*",
)

model = load("model.joblib")

app = Flask(__name__)
CORS(app)


@app.route("/api/metadata", methods=["GET"])
def metadata():
    return jsonify(getUniqueValues())


@app.route("/api/predict", methods=["POST"])
def api():
    data = request.get_json()

    if "batting_team" not in data:
        return jsonify({"error": "batting_team is required"}), 400
    if "bowling_team" not in data:
        return jsonify({"error": "bowling_team is required"}), 400
    if "overs" not in data:
        return jsonify({"error": "overs is required"}), 400
    if "runs" not in data:
        return jsonify({"error": "runs is required"}), 400
    if "wickets" not in data:
        return jsonify({"error": "wickets is required"}), 400
    if "runs_in_prev_5" not in data:
        return jsonify({"error": "runs_in_prev_5 is required"}), 400
    """
    Sample JSON input:
    {
        "batting_team": "Kolkata Knight Riders",
        "bowling_team": "Delhi Daredevils",
        "overs": 9.2,
        "runs": 79,
        "wickets": 2,
        "runs_in_prev_5": 60,
        "wickets_in_prev_5": 1
    }
    """

    score = predict(
        model=model,
        batting_team=data["batting_team"],
        bowling_team=data["bowling_team"],
        overs=data["overs"],
        runs=data["runs"],
        wickets=data["wickets"],
        runs_in_prev_5=data["runs_in_prev_5"],
        wickets_in_prev_5=data["wickets_in_prev_5"],
    )

    return jsonify({"score": score}), 200


if __name__ == "__main__":
    app.run(debug=True)
