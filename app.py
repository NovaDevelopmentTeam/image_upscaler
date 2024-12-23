import os
from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from PIL import Image
import io

# Initialisiere Flask
app = Flask(__name__)

# Erstelle ein einfaches CNN für Super-Resolution
def create_model(input_shape):
    model = models.Sequential()
    model.add(layers.Conv2D(64, (9, 9), activation='relu', padding='same', input_shape=input_shape))
    model.add(layers.Conv2D(32, (1, 1), activation='relu', padding='same'))
    model.add(layers.Conv2D(3, (5, 5), activation='sigmoid', padding='same'))  # 3 für RGB
    return model

# Lade das Modell
model = create_model((32, 32, 3))  # Beispiel für niedrigauflösendes Bild (32x32)
model.compile(optimizer='adam', loss='mse')

# Dummy-Daten laden (normalerweise würdest du echte Trainingsdaten verwenden)
# Hier lade ich ein zufälliges Bild als Beispiel
input_image = np.random.rand(1, 32, 32, 3)
target_image = np.random.rand(1, 128, 128, 3)
model.fit(input_image, target_image, epochs=10)

@app.route('/upscale', methods=['POST'])
def upscale_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    img = Image.open(file.stream)

    # Image vorverarbeiten: Konvertiere es zu einem NumPy Array
    img = img.resize((32, 32))  # Niedrige Auflösung simulieren
    img_array = np.array(img) / 255.0  # Normalisieren
    img_array = img_array.reshape((1, 32, 32, 3))

    # Upscaling durchführen
    upscaled_img = model.predict(img_array)

    # Vorhersage nach Bildgröße umwandeln
    upscaled_img = np.clip(upscaled_img[0], 0, 1) * 255  # Werte zurück auf [0, 255]
    upscaled_img = upscaled_img.astype(np.uint8)

    # Bild zurück in ein PIL-Image konvertieren
    result_img = Image.fromarray(upscaled_img)
    
    # Das Ergebnis als Bytes zurückgeben
    img_byte_arr = io.BytesIO()
    result_img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    return send_file(img_byte_arr, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
