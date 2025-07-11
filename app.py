from flask import Flask, request
from flask_cors import CORS
from result import flask_response
from services import *

app = Flask(__name__)
CORS(app)


@app.route("/")
def home():
    return ""


@app.post("/news/check")
def check_news():
    query = request.get_json()
    return flask_response(checkNews(query))


@app.post("/news/explain")
def explain_news():
    query = request.get_json()
    return flask_response(explainNews(query))


@app.post("/news/scratch")
def scratch_news():
    query = request.get_json()
    return flask_response(scratchNews(query))


@app.post("/news/pic/check")
def multimodal_check_news():
    query = request.get_json()
    return flask_response(multimodalChecking(query))


@app.post("/news/pic/explain")
def multimodal_explain_news():
    query = request.get_json()
    return flask_response(multimodal_explanation(query))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

