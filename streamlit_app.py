import streamlit as st
import tensorflow as tf
import numpy as np
from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

st.set_option('deprecation.showfileUploaderEncoding', False)

@st.cache(allow_output_mutation=True)
def load_model(model_name='LinkNet_Resnet34'):
    model_name = 'models/' + str(model_name) + '.h5'
    model = tf.keras.models.load_model(model_name)
    return model

def format_image(image):
    image = resize(image, (512, 512,3), mode = 'constant', preserve_range = True)
    image = image/255
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


# Add a selectbox to the sidebar:
select_model = st.sidebar.selectbox(
    'Which model would you like to try?',
    ('LinkNet_Resnet34', 'U-Net_Resnet34', 'U-Net_vgg19', 'U-Net_Mobilenet',
    'U-Net_Inceptionv3', 'U-Net_Densenet121')
)

select_image = st.sidebar.selectbox(
    'Choose an Image.',
    (None, 'train_6.png', 'train_7.png', 'train_8.png', 'train_9.png')
)
if (select_image is not None):
    select_image = 'images/' + select_image 

model = load_model(select_model)
st.title('Cell Segmentation Tool')

# File Uploader toolbar for uploading images
file = st.file_uploader("Upload an image of cells. Don't have an image? Choose a sample from the sidebar.", type=["jpg", "png"])
file = select_image

if file is None:
	st.text('Waiting for upload....')

else:
    slot = st.empty()
    slot.text('Running inference....')

    # Read uploaded image or selected imge from file object
    uploaded_image = imread(file)

    # Plot uploaded/selected image
    fig, ax = plt.subplots()
    ax.title.set_text('Uploaded Image')
    ax.imshow(uploaded_image, cmap='gray')
    st.pyplot(fig)

    # Format input image for model inferencing
    formatted_image = format_image(uploaded_image)

    # Obtain output image from model
    output_image = predict_image(formatted_image, model)

    # Plot output image
    fig2, ax2 = plt.subplots()
    ax2.title.set_text('Output Image')
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes('bottom', size='5%', pad=0.05)
    im2 = ax2.imshow(output_image, cmap='jet',vmin=0, vmax=7)
    fig2.colorbar(im2, cax = cax2, orientation='horizontal')
    st.pyplot(fig2)

    slot.text('Completed Inference!')
