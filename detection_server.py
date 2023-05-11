import torch
import cv2
from flask import Flask, request, jsonify
from io import BytesIO
import base64
from PIL import Image

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = 0.35  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold
model.agnostic = False  # NMS class-agnostic
model.multi_label = False  # NMS multiple labels per box
model.max_det = 1000  # maximum number of detections per image

app = Flask(__name__)

@app.route('/detect', methods=['POST'])
def detect_objects():
    if 'file' not in request.files:
        return jsonify({'error': 'No file in the request'})

    # Read the uploaded image
    file = request.files['file']
    image_bytes = file.read()
    image = Image.open(BytesIO(image_bytes))
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Perform object detection
    results = model(image)

    # Parse results
    predictions = results.pred[0]
    objects = []
    for prediction in predictions:
        box = prediction[:4].tolist()
        class_name = str(prediction[5])
        confidence = float(prediction[4])
        cropped_image = image[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
        cropped_image = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
        buffered = BytesIO()
        cropped_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        objects.append({'class': class_name, 'confidence': confidence, 'image': img_str})

    return jsonify({'objects': objects})

if __name__ == '__main__':
    app.run(port=8501)
