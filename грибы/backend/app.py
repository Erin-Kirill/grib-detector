from flask import Flask, request, render_template, jsonify
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model('model.h5')  # Укажи путь к своей модели
class_names = sorted(os.listdir('dataset/грибы/съедобные грибы')) + sorted(os.listdir('dataset/грибы/несъедобные грибы'))

def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Изменить на нужный размер
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Нет файла'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Файл не выбран'})

    filepath = os.path.join('static', 'uploads', file.filename)
    file.save(filepath)

    img = prepare_image(filepath)
    prediction = model.predict(img)
    predicted_class = class_names[np.argmax(prediction)]

    return jsonify({'class': predicted_class, 'filename': file.filename})

if __name__ == '__main__':
    os.makedirs('static/uploads', exist_ok=True)
    app.run(debug=True)
