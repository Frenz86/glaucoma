import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import requests
import os
from io import BytesIO

def import_and_predict(image_data, model):
    image = ImageOps.fit(image_data, (100,100),Image.ANTIALIAS)
    image = image.convert('RGB')
    image = np.asarray(image)
    st.image(image, channels='RGB')
    image = (image.astype(np.float32) / 255.0)
    img_reshape = image[np.newaxis,...]
    prediction = model.predict(img_reshape)
    return prediction

import wget
def download_model():
    path1 = './my_model2.h5'
    if not os.path.exists(path1):
        url = 'https://frenzy86.s3.eu-west-2.amazonaws.com/python/models/my_model2.h5'
        filename = wget.download(url)
    else:
        print("Model is here.")

##### MAIN ####
def main():
    ################ css background #########################
    # html_temp = """
    # <iframe src='https://flo.uri.sh/visualisation/2943892/embed' title='Interactive or visual content' frameborder='0' scrolling='no' 
    # style='width:100%;height:600px;'></iframe>
    # """
    # st.markdown(html_temp, unsafe_allow_html=True)
    
    page_bg_img = '''
    <style>
    body {
    background-image: url("https://i.pinimg.com/originals/85/6f/31/856f31d9f475501c7552c97dbe727319.jpg");
    background-size: cover;
    }
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)  
    
    ################ load logo from web #########################
    image = Image.open('the-biggest.jpg')
    st.title("AI APP to predict glaucoma through fundus image of eye")
    st.image(image, caption='',use_column_width=True)
    download_model()
    model = tf.keras.models.load_model('my_model2.h5')
    file = st.file_uploader("Please upload an image(jpg) file", type=["jpg"])
    if file is None:
        st.text("You haven't uploaded a jpg image file")
    else:
        imageI = Image.open(file)
        prediction = import_and_predict(imageI, model)
        pred = prediction[0][0]
        if(pred > 0.5):
            st.write("""
                     ## **Prediction:** You eye is Healthy. Great!!
                     """
                     )
        else:
            st.write("""
                     ## **Prediction:** You have an high probability to be affected by Glaucoma. Please consult an ophthalmologist as soon as possible.
                     """
                     )
if __name__ == '__main__':
    main()