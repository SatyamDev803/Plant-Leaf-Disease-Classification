from flask import Flask, render_template, request, redirect
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import io
import base64
import os

app = Flask(__name__)

MODEL_PATH = "models/2.keras"
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

class_names = ['Early Blight', 'Late Blight', 'Healthy']

def predict(image):
    try:
        # Preprocess the image
        img = image.resize((256, 256))  # Resize to match model input size
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)  # Add batch dimension

        # Predict the class
        predictions = model.predict(img)
        print(f"Predictions: {predictions}")  # print predictions
        predicted_class = class_names[np.argmax(predictions[0])]
        confidence = round(100 * (np.max(predictions[0])), 2)
        
        # Convert image to base64
        img_buffer = io.BytesIO()
        image.save(img_buffer, format="JPEG")
        img_str = base64.b64encode(img_buffer.getvalue()).decode()
        
        print(f"Base64 Image String: {img_str[:100]}...")  
        
        return predicted_class, confidence, img_str
    except Exception as e:
        print(f"Error in prediction: {e}")
        return "Error", 0, None

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            return redirect(request.url)
        
        if file:
            img = Image.open(file.stream)
            predicted_class, confidence, img_data = predict(img)
            
            return render_template('index.html', prediction=predicted_class, confidence=confidence, img_data=img_data)
    
    return render_template('index.html', prediction=None, confidence=None)


from flask_caching import Cache
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

@app.route('/clear_cache')
def clear_cache():
    cache.clear()
    return 'Cache cleared!'

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=True)
