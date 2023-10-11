from PIL import Image
import streamlit as st
from ultralytics import YOLO

MODEL_DIR = './runs/detect/train/weights/best.pt'


def main():
    # load a model
    model = YOLO(MODEL_DIR)

    #st.sidebar.header("**Fabric Defect Detection**")

    #for animal in sorted(os.listdir('./data/raw')):
     #   st.sidebar.markdown(f"- *{animal.capitalize()}*")

    st.title("Real-time Fabric Defect Detection")
    st.write("The aim of this project is to develop an efficient computer vision model capable of real-time Fabric Defect detection.")

    # Load image or video
    uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

    if uploaded_file:
        if uploaded_file.type.startswith('image'):
            inference_images(uploaded_file, model)
        
       # if uploaded_file.type.startswith('video'):
        #    inference_video(uploaded_file)


def inference_images(uploaded_file, model):
    image = Image.open(uploaded_file)
     # predict the image
    predict = model.predict(image)

    # plot boxes
    boxes = predict[0].boxes
    plotted = predict[0].plot()[:, :, ::-1]

    if len(boxes) == 0:
        st.markdown("**No Detection**")

    # open the image.
    st.image(plotted, caption="Detected Image", width=600)
    #logging.info("Detected Image")


if __name__=='__main__':
    main()