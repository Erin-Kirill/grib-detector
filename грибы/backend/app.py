from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)

# Загружаем модель
model = load_model('mushroom_classifier_model_final.h5')

# Получаем классы из структуры train_data
class_names = sorted(os.listdir('train_data'))

# Предобработка изображения
def preprocess_image(img_path):
    img = load_img(img_path, target_size=(160, 160))
    img_array = img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0), img

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    top_predictions = []
    image_path = None

    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename != '':
            save_folder = os.path.join('static', 'uploads')
            os.makedirs(save_folder, exist_ok=True)
            img_path = os.path.join(save_folder, file.filename)
            file.save(img_path)

            img_array, _ = preprocess_image(img_path)
            pred = model.predict(img_array)[0]

            top_indices = pred.argsort()[-4:][::-1]
            top_predictions = [(class_names[i], round(pred[i] * 100, 2)) for i in top_indices]
            prediction = top_predictions[0][0].upper()
            image_path = '/' + img_path.replace('\\', '/')

    return render_template('index.html',
                           prediction=prediction,
                           top_predictions=top_predictions,
                           image_path=image_path)

if __name__ == '__main__':
    app.run(debug=True)
