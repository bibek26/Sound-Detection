import io
import os
import base64
import matplotlib
import numpy as np
from flask import Flask, request, jsonify
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from core import (create_bird_path, create_result, create_spectrogram,
                  get_bird_data, predict)

matplotlib.use('Agg')
THIS_DIR = os.path.dirname(os.path.realpath(__file__))
BIRD_DATA = os.path.join(THIS_DIR, 'data', 'bird_data.xlsx')
TEMPLATES = os.path.join(THIS_DIR, 'templates')
model = os.path.join(THIS_DIR, 'model', 'model.h5')
classes = np.array(['Acroc', 'Ember', 'Parus', 'Phyll', 'Sylvi'])
ALLOWED_EXTENSIONS = {'wav'}


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


# Method to check if the file uploaded is valid or not
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/upload', methods=['POST', 'GET'])
def upload_file():
    file = None
    file = request.files['file']
    
    if not file or file.filename == '':
        return jsonify({'error': 'No selected file!'}), 400

    if file and allowed_file(file.filename):
        image, fig = create_spectrogram(file)
        pred = predict(model, image)
        result = create_result(pred, classes)
        
        pngImage = io.BytesIO()
        FigureCanvas(fig).print_png(pngImage)
        spectrogram = "data:image/png;base64,"
        spectrogram += base64.b64encode(pngImage.getvalue()).decode('utf8')
        
        if result['probability'] > 74:
            bird_path = create_bird_path(result['bird'])
            probability = str(result['probability'])
            bird_type = result['bird']
            name, en_name, desc = get_bird_data(bird_type)

            return jsonify({
              'bird': bird_path,
              'probability': probability,
              'bird_type': bird_type,
              'name': name,
              'en_name': en_name,
              'desc': desc
            }), 200
        else:
          return jsonify({'error': 'Could not identify the bird!'}), 200
    else:
        return jsonify({'error': 'Wrong file format!'}), 400
  

@app.errorhandler(413)
def error413():
    return jsonify({'error': 'Too big file!'}), 413  


if __name__ == "__main__":
	app.run()