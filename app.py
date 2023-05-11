import streamlit as st
import torch
import cv2

st.title("YOLOv5 Object Detection")
st.write("Upload an image and perform object detection using YOLOv5.")


# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
  
# set model parameters
model.conf = 0.35  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold
model.agnostic = False  # NMS class-agnostic
model.multi_label = False  # NMS multiple labels per box
model.max_det = 1000  # maximum number of detections per image




def run_inference(image):
    # perform inference
    results = model(image)

    # parse results
    predictions = results.pred[0]
    boxes = predictions[:, :4] # x1, y1, x2, y2
    scores = predictions[:, 4]
    categories = predictions[:, 5]

    # show detection bounding boxes on image
    results.show()

    # save results into "results/" folder
    results.save(save_dir='results/')




uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Read the image file
    image = cv2.imdecode(np.fromstring(uploaded_image.read(), np.uint8), 1)

    # Perform inference
    run_inference(image)



if __name__ == '__main__':
    app.run()
