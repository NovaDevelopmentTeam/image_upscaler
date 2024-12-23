import os
from flask import Flask, request, jsonify, send_file, render_template
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from PIL import Image
import io

# Sicherstellen, dass TensorFlow nur die CPU nutzt
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Initialisiere Flask
app = Flask(__name__)

# Erstelle ein einfaches CNN für Super-Resolution
from tensorflow.keras.layers import Input

def create_model(input_shape):
    inputs = Input(shape=input_shape)
    x = layers.Conv2D(64, (9, 9), activation='relu', padding='same')(inputs)
    x = layers.Conv2D(32, (1, 1), activation='relu', padding='same')(x)
    outputs = layers.Conv2D(3, (5, 5), activation='sigmoid', padding='same')(x)
    return models.Model(inputs, outputs)

# Lade das Modell
model = create_model((32, 32, 3))  # Beispiel für niedrigauflösendes Bild (32x32)
model.compile(optimizer='adam', loss='mse')

# Dummy-Daten generieren und Modell trainieren
# Hinweis: Dies ist nur ein Beispieltraining, in der Praxis sollte ein vortrainiertes Modell verwendet werden.
input_image = np.random.rand(1, 32, 32, 3)
target_image = np.random.rand(1, 32, 32, 3)  # Gleiche Dimension wie die Eingabe
model.fit(input_image, target_image, epochs=10, verbose=0)

@app.route('/')
def home():
    return render_template('index.html')  # Die HTML-Datei mit der Benutzeroberfläche

@app.route('/upscale', methods=['POST'])
def upscale_image():
    try:
        # Überprüfen, ob eine Bilddatei hochgeladen wurde
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        file = request.files['image']
        img = Image.open(file.stream).convert('RGB')  # Konvertieren in RGB

        # Bild vorverarbeiten: Größe anpassen und normalisieren
        img = img.resize((32, 32))  # Niedrige Auflösung simulieren
        img_array = np.array(img) / 255.0  # Normalisieren
        img_array = img_array.reshape((1, 32, 32, 3))

        # Upscaling durchführen
        upscaled_img = model.predict(img_array)

        # Vorhersage verarbeiten: Werte zurück auf [0, 255] skalieren
        upscaled_img = np.clip(upscaled_img[0], 0, 1) * 255  # Werte clippen und skalieren
        upscaled_img = upscaled_img.astype(np.uint8)

        # In ein PIL-Image konvertieren
        result_img = Image.fromarray(upscaled_img)

        # Das Ergebnis als Bytes zurückgeben
        img_byte_arr = io.BytesIO()
        result_img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        return send_file(img_byte_arr, mimetype='image/png')

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)), debug=True)
