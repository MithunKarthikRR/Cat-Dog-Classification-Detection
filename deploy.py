import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import pickle

st.write("""
          # Cat & Dog Classification
          """
          )
st.sidebar.header('User Input Parameters')

upload_file = st.sidebar.file_uploader("Upload an Image", type="jpg")
Generate_pred=st.sidebar.button("Predict")

basemodel = tf.keras.models.load_model('models/basemodel.h5')
vgg16 = tf.keras.models.load_model('models/vgg.h5')
yolov8 = pickle.load(open('models/yolov8n.pkl', 'rb'))

def model_pred(image_data, model):
    size = (128,128)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.asarray(image)
    reshape=img[np.newaxis,...]
    pred = model.predict(reshape)
    print(pred)
    if pred[0][0]<0.5:
        return "Cat"
    else:
        return "Dog"
    
def model_detect(image_data, model):
    image_data.save('test-images/predicted_image.jpg')  # Save the image with a supported extension
    model.predict('test-images/predicted_image.jpg', save=True)  # Perform object detection

    
if Generate_pred:
    if upload_file is not None:
        image=Image.open(upload_file)
        with st.expander('Uploaded Image', expanded = True):
            st.image(image, use_column_width=True)

        base = model_pred(image, basemodel)
        vgg = model_pred(image, vgg16)
        yolo = model_detect(image, yolov8)

        labels = ['Cat', 'Dog']
        st.title("Prediction of Base Model is {}".format(base))
        st.title("Prediction of VGG16 is {}".format(vgg))
        st.write('\n')

        st.title("Image Detection using YOLOv8 pre trained model")
        st.write('\n')
        # Display the detected image
        file_location = "runs/detect/predict/predicted_image.jpg"
        image_from_location = Image.open(file_location)
        st.image(image_from_location, use_column_width=True)

    else:
        st.write("Please upload an image")



## To run the app
# 1. streamlit run deploy.py
# 2. python -m streamlit run deploy.py