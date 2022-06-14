import streamlit as st
import tensorflow as tf
import numpy as np
from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt

st.set_option('deprecation.showfileUploaderEncoding', False)

@st.cache(allow_output_mutation=True)
def load_model():
	model = tf.keras.models.load_model('keras.h5')
	return model


def predict_class(image, model):

	image = tf.cast(image, tf.float32)
	image = tf.image.resize(image, [180, 180])

	image = np.expand_dims(image, axis = 0)

	prediction = model.predict(image)

	return prediction

def format_image(image):
    image= resize(image, (256, 256,1), mode = 'constant', preserve_range = True)
    image_shape = np.shape(image)
    image_formatted = np.zeros((1,image_shape[0],image_shape[1],image_shape[2]))
    image_formatted[0,:,:,:] = image
    return image_formatted
    # 
    # print(image_shape)

def predict_image(image, model):
    pred = model.predict(image)
    pred_argmax = np.argmax(pred,axis=3)
    output =pred_argmax[0,:,:]
    return output

# print(np.shape(sample_image_formatted))



model = load_model()
st.title('Cell Segmentation Tool')

file = st.file_uploader("Upload an image of cells", type=["jpg", "png"])


if file is None:
	st.text('Waiting for upload....')

else:
    slot = st.empty()
    slot.text('Running inference....')

    #uploaded_image = Image.open(file).convert('L')
    #print(np.shape(uploaded_image))
    #uploaded_image = uploaded_image[..., np.newaxis]
    uploaded_image = imread(file,as_gray=True)
    #st.image(uploaded_image, caption="Input Image", width = 400)
    fig, ax = plt.subplots()
    ax.title.set_text('Uploaded Image')
    ax.imshow(uploaded_image, cmap='gray')
    st.pyplot(fig)


    formatted_image = format_image(uploaded_image)
    print(np.shape(formatted_image))
    output_image = predict_image(formatted_image, model)
    fig2, ax2 = plt.subplots()
    ax2.title.set_text('Output Image')
    ax2.imshow(output_image, cmap='jet')
    # plt.figure(figsize=(8, 8))
    # plt.subplot(221)
    # plt.title('Testing Image')
    # plt.imshow(uploaded_image, cmap='gray')
    # plt.subplot(222)
    # plt.title('Prediction on test image')
    # plt.imshow(output_image)
    # plt.savefig('x',dpi=400)
    #st.image('x.png',caption="Segtmented Image")

    #st.image(output_image, caption="Segmented Image")
    st.pyplot(fig2)
    slot.text('Done')
    #st.success(output)
