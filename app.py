import os
from flask import Flask, request
from flask_cors import cross_origin
app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/result/<request_id>', methods=['POST'])
@cross_origin('http://localhost:3000', supports_credentials=True)
def receive_file(request_id):
    """
    Receive the image file user posted to use trained model.

    Uses tensorflow trained model here.
    """
    file = request.files['file']
    file.save(os.path.join('./', file.filename))
    return 'got it'
