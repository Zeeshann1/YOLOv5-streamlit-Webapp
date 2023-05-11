import streamlit as st
import torch
from PIL import Image
from torchvision.transforms import ToTensor
from yolov5.utils.plots import plot_one_box
from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets import letterbox

st.set_page_config(page_title="YOLOv5 Object Detection", page_icon=":guardsman:", layout="wide")

@st.cache(allow_output_mutation=True)
def load_model():
    model = attempt_load("yolov5s.pt", map_location=torch.device('cpu'))
    return model

def detect(image):
    model = load_model()
    img = Image.open(image)
    img = img.convert("RGB")
    img = letterbox(img, new_shape=640)[0]
    img = ToTensor()(img)
    img = img.unsqueeze(0)
    pred = model(img)[0]
    for det in pred:
        plot_one_box(det[:4], det[4], color=(0, 255, 0), line_thickness=2)

def main():
    st.title("YOLOv5 Object Detection")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        if st.button("Detect Objects"):
            detect(uploaded_file)
            st.image(image, caption="Annotated Image", use_column_width=True)

if __name__ == "__main__":
    main()
