from flask import Flask, jsonify, request, send_file
from werkzeug.utils import secure_filename
import os
from flask_cors import CORS
from models.model import cam, visualize_cam
from preprocessing.preprocessing_image import preprocess_image
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '')))
app = Flask(__name__)
cors = CORS(app)

UPLOAD_FOLDER = 'image/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['jpeg'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/upload", methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        resp = jsonify({
            "message": "No file part in the request",
            "status": 'failed'
        })
        resp.status_code = 400
        return resp

    file = request.files['file']
    print(file)
    
    if file and allowed_file(file.filename):
        extension = secure_filename(file.filename).rsplit('.', 1)[1].lower()
        custom_filename = f"xray.{extension}"
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], custom_filename))
        
        
        resp = jsonify({
            "message": "File successfully uploaded",
            "filename": custom_filename,
            "status": 'success'
        })
        resp.headers.add("Access-Control-Allow-Origin", "*")
        resp.status_code = 201
    
        return resp
    else:
        resp = jsonify({
            "message": 'File type is not allowed',
            "status": 'failed'
        })
        resp.status_code = 400
        return resp

@app.route('/predict', methods=['POST'])
def predict():
    original_img, preprocessed_img = preprocess_image()
    activationmap, output = cam(preprocessed_img)
    visualize_cam(original_img, activationmap, output)
    model_prediction = output.item()
    prediction = "Pneumonia" if model_prediction > 0.5 else "healthy"
    response = {
        'message': 'Prediction completed',
        'prediction': prediction,
        'model_prediction': model_prediction,  
    }
    return jsonify(response)
    
@app.route('/get-xray')
def get_xray():
    return send_file('image/xray_processed.jpeg', mimetype='image/jpeg')

if __name__ == '__main__':
    app.run()
    