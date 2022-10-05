from doc_similarity import normalize_image, extract
from scipy.spatial import distance
from flask import Flask, request
from dotenv import load_dotenv

import os

load_dotenv()

threshold = float(os.getenv("THRESHOLD"))
metric = os.getenv("METRIC")
debug = os.getenv("DEBUG")

app = Flask(__name__)
@app.route('/doc_similarity', methods=['POST'])
def doc_similarity():
    if request.method == 'POST':
        request_data = request.get_json()
        image_1 = extract(normalize_image(request_data['image_1']))
        image_2 = extract(normalize_image(request_data['image_2']))
        dc = distance.cdist([image_1], [image_2], metric)[0]
        if dc > threshold:
            return {"response":"TIDAK SAMA"}
        else:
            return {"response":"SAMA"}
    else:
        return

if __name__ == '__main__':
    app.run(debug=debug, port=8001)