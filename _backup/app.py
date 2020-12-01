import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import requests
from io import BytesIO

st.set_option('deprecation.showfileUploaderEncoding', False)

def import_and_predict(image_data, model):
    image = ImageOps.fit(image_data, (100,100),Image.ANTIALIAS)
    image = image.convert('RGB')
    image = np.asarray(image)
    st.image(image, channels='RGB')
    image = (image.astype(np.float32) / 255.0)
    img_reshape = image[np.newaxis,...]
    prediction = model.predict(img_reshape)
    return prediction

def main():
    ################ load logo from web #########################
    image = Image.open('the-biggest.jpg')
    st.title("AI APP to predict glaucoma through fundus image of eye")
    st.image(image, caption='',use_column_width=True)
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