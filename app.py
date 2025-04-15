from flask import Flask, render_template, request, jsonify, send_from_directory
from tensorflow.lite.python.interpreter import Interpreter
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 🔁 Use non-GUI backend

import matplotlib.pyplot as plt

import os
import uuid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load TFLite models
models = {
    'inceptionv3': Interpreter(model_path='models/inception.tflite'),
    'vgg19': Interpreter(model_path='models/VGG19.tflite'),
    'densenet': Interpreter(model_path='models/Densenet.tflite')
}
for model in models.values():
    model.allocate_tensors()

# Class labels
class_labels = ['Armadilles', 'Bear', 'Birds', 'Cow', 'Crocodile', 'Deer', 'Elephant',
                'Goat', 'Horse', 'Jaguar', 'Monkey', 'Rabbit', 'Skunk', 'Tiger', 'Wild Boar']

# Helper: Resize image based on model type
def preprocess_image(image_path, model_type):
    img = Image.open(image_path).convert('RGB')
    size = 299 if model_type == 'inceptionv3' else 224
    img = img.resize((size, size))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0).astype(np.float32)

# Helper: Predict with given model
def predict_with_model(model_key, image_path):
    interpreter = models[model_key]
    img_array = preprocess_image(image_path, model_key)
    input_index = interpreter.get_input_details()[0]['index']
    output_index = interpreter.get_output_details()[0]['index']
    interpreter.set_tensor(input_index, img_array)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_index)
    return class_labels[np.argmax(prediction)]

# Route: Index
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predict.html', methods=['GET'])
def predict_page():
    return render_template('predict.html')
# Route: Predict
@app.route('/predict', methods=['POST'])
def predict_route():
    file = request.files.get('image')
    model_choice = request.form.get('model')

    if not file or file.filename == '':
        return jsonify({'error': 'No image uploaded'}), 400
    if not model_choice:
        return jsonify({'error': 'No model selected'}), 400
    # Clear the uploads folder before saving the new file
    for f in os.listdir(app.config['UPLOAD_FOLDER']):
        os.remove(os.path.join(app.config['UPLOAD_FOLDER'], f))

# Save the uploaded image
    filename = f"{uuid.uuid4().hex}_{file.filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)


    predictions = {}

    # Perform prediction
    if model_choice == 'all':
        for key in models:
            predictions[key] = predict_with_model(key, filepath)
    else:
        predictions[model_choice] = predict_with_model(model_choice, filepath)

    # Draw the image with model labels using matplotlib
    img = Image.open(filepath)
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.axis('off')

    title_lines = ["Intrusion Detected !!"]
    for model_key, label in predictions.items():
        title_lines.append(f"{model_key.upper()}: {label}")
    plt.title("\n".join(title_lines), fontsize=12)

    # Save result image
    result_filename = f"result_{filename}.png"
    result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
    plt.savefig(result_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()

    return jsonify({
        'image_url': f'/uploads/{result_filename}',
        'prediction': predictions
    })

# Serve uploaded and result images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
