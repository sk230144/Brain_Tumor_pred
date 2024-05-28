import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import cv2
from keras.models import load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)
model = load_model('BrainTumor10EpochsCategorical.h5')
print('Model loaded. Check http://127.0.0.1:5000/')

def getResult(img):
    image = cv2.imread(img)
    img = Image.fromarray(image)
    img = img.resize((64, 64))
    img = np.array(img)
    input_img = img.astype('float32') / 255.0
    input_img = np.expand_dims(input_img, axis=0)
    predictions = model.predict(input_img)
    tumor_probability = predictions[0][1] * 100
    
    # Determine the binary classification based on a threshold
    threshold = 50  # Adjust the threshold as needed
    if tumor_probability >= threshold:
        binary_result = "Yes"
    else:
        binary_result = "No"
    
    return tumor_probability, binary_result

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        tumor_probability, binary_result = getResult(file_path)
        
        result = (
            f"The probability of tumor presence is: {tumor_probability:.2f}%  and  "
            
            f"Tumor presence: {binary_result}"
        )
        
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)








# @app.route('/predict', methods=['GET', 'POST'])
# def upload():
#     if request.method == 'POST':
#         f = request.files['file']
#         basepath = os.path.dirname(__file__)
#         file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
#         f.save(file_path)
#         tumor_probability, binary_result = getResult(file_path)
#         result = f"The probability of tumor presence is: {tumor_probability:.2f}%</br>Tumor presence: {binary_result}"
#         return result
#     return None