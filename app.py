
import os
from flask import Flask, request
from flask_cors import cross_origin
from tf import evaluate_image
app = Flask(__name__)


@app.route('/result/<request_id>', methods=['POST'])
@cross_origin('http://143.248.36.226:3000', supports_credentials=True)
def receive_file(request_id):
    """
    Receive the image file user posted to use trained model.

    Uses tensorflow trained model here.
    """
    req_file = request.files['file']
    filename = req_file.filename
    req_file.save(os.path.join('./', filename))
    return str(evaluate_image(filename))

