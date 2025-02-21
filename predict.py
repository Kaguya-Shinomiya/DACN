import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


from tensorflow.keras.preprocessing.image import img_to_array
# from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
import numpy as np
from PIL import Image
import cv2

# # Hàm tiền xử lý ảnh kết hợp gamma correction và CLAHE
# def preprocess_image(image, gamma=1.2):
#     # Áp dụng gamma correction
#     invGamma = 1.0 / gamma
#     table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
#     image = image.astype(np.uint8)

#     if len(image.shape) == 3 and image.shape[2] == 3:  # Nếu là ảnh màu (RGB)
#         channels = cv2.split(image)
#         corrected_channels = [cv2.LUT(channel, table) for channel in channels]
#         corrected_image = cv2.merge(corrected_channels)
#     else:  # Nếu là ảnh grayscale
#         corrected_image = cv2.LUT(image, table)

#     # Áp dụng CLAHE
#     if len(corrected_image.shape) == 3 and corrected_image.shape[2] == 3:
#         lab = cv2.cvtColor(corrected_image, cv2.COLOR_RGB2LAB)
#         l, a, b = cv2.split(lab)
#         clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#         cl = clahe.apply(l)
#         limg = cv2.merge((cl, a, b))
#         enhanced_image = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
#     else:
#         clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#         enhanced_image = clahe.apply(corrected_image)

#     return enhanced_image


import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
# from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

def predict_glaucoma(file, model):
    # 1. Mở và chỉnh kích thước ảnh
    img = Image.open(file)
    img = img.resize((256, 256))
    img_array = np.array(img)

    # 2. Tiền xử lý ảnh
    # img_array = preprocess_image(img_array)
    img_array = img_to_array(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    # img_array = preprocess_input(img_array)

    # 3. Dự đoán
    predictions = model.predict(img_array)
    predicted_prob = predictions[0][0]
    predicted_label = "Tăng nhãn áp" if predicted_prob > 0.5 else "Không tăng nhãn áp"

    # 4. Tạo Grad-CAM
    last_conv_layer_name = 'conv2d'  # Thay bằng tên lớp convolution cuối cùng của MobileNetV3
    heatmap = generate_gradcam_heatmap(model, img_array, last_conv_layer_name)

    # 5. Vẽ hình vuông focus vào khu vực có khả năng bị bệnh
    img_with_square_focus = apply_focus_square_on_image(img, heatmap, predicted_prob)

    return predicted_label, predicted_prob, img_with_square_focus

def generate_gradcam_heatmap(model, img_array, last_conv_layer_name):
    grad_model = Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]  # Lấy lớp 0 cho Grad-CAM

    # Tính gradient của loss so với conv_outputs
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Nhân gradient với output của conv_outputs
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(pooled_grads * conv_outputs, axis=-1)

    # Chuẩn hóa heatmap
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1

    # Trả về NumPy array trực tiếp
    return heatmap

def apply_focus_square_on_image(img, heatmap, predicted_prob, alpha=0.4):
    # Resize heatmap để phù hợp với kích thước ảnh gốc
    heatmap = cv2.resize(heatmap, (img.width, img.height))

    # Tạo ngưỡng để xác định khu vực cần focus
    threshold = 0.2  # Ngưỡng giá trị để xác định các điểm cần chú ý
    heatmap_binary = heatmap > threshold

    # Tìm tất cả tọa độ của vùng có giá trị lớn hơn ngưỡng
    y_coords, x_coords = np.where(heatmap_binary)

    # Nếu tìm được vùng có giá trị cao
    if len(x_coords) > 0 and len(y_coords) > 0:
        # Xác định tọa độ của hình chữ nhật bao quanh vùng sáng nhất
        top_left_x = np.min(x_coords)
        top_left_y = np.min(y_coords)
        bottom_right_x = np.max(x_coords)
        bottom_right_y = np.max(y_coords)

        # Vẽ hình chữ nhật lên ảnh gốc
        img_array = np.array(img)
        cv2.rectangle(img_array, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 255, 0), 2)

        return Image.fromarray(img_array)

    # Trả về ảnh gốc nếu không có vùng nào đủ sáng
    return img
