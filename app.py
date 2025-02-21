from flask import Flask, request, jsonify, render_template
import predict
import tensorflow as tf



from tensorflow.keras.optimizers import Adam
import keras

app = Flask(__name__)


model = tf.keras.models.load_model('./val-best-transfer.h5')
model1 = tf.keras.models.load_model('./efficientnetb4.h5', compile=False)
model1.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['binary_accuracy',keras.metrics.AUC(),keras.metrics.Precision(), keras.metrics.Recall()])

# Chỉ cho phép các loại tệp ảnh
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Hàm kiểm tra loại tệp hợp lệ
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('glaucoma_predict.html')

import base64
from io import BytesIO
from PIL import Image
@app.route('/predict', methods=['POST'])
def upload_predict():
    if 'image' not in request.files:
        return jsonify({'result': 'Không có tệp nào được tải lên.'})

    file = request.files['image']

    if file and allowed_file(file.filename):
        # Dự đoán và sinh heatmap
        predicted_label, probability, heatmap_img = predict.predict_glaucoma(file=file, model=model)
        predicted_label, probability1, heatmap_img1 = predict.predict_glaucoma(file=file, model=model1)

        # Chuyển đổi ảnh heatmap thành chuỗi base64
        buffered = BytesIO()
        heatmap_img.save(buffered, format="PNG")
        heatmap_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        heatmap_img1.save(buffered, format="PNG")
        heatmap1_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        # Kết quả trả về
        
        if predicted_label == "Tăng nhãn áp":
            result_text = f"Mobilenet V3 Large: Có khả năng: {predicted_label}, với tỉ lệ mắc bệnh là: {probability * 100:.2f}%"
            result_text1 = f"Efficientnet B4: Có khả năng: {predicted_label}, với tỉ lệ mắc bệnh là: {probability1 * 100:.2f}%"
            return jsonify({'result': result_text, 'heatmap': heatmap_base64, 'result1': result_text1, 'heatmap1': heatmap1_base64})
        else:
            result_text = f"Mobilenet V3 Large: Có khả năng: {predicted_label}"
            result_text1 = f"Efficientnet B4: Có khả năng: {predicted_label}"
            return jsonify({'result': result_text, 'result1': result_text1})
    else:
        return jsonify({'result': 'Tệp không hợp lệ hoặc không được hỗ trợ.'})

if __name__ == '__main__':
    app.run(debug=True)
    
    
    