import streamlit as st
import tensorflow as tf
import numpy as np
from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
import segmentation_models as sm
from mpl_toolkits.axes_grid1 import make_axes_locatable

st.set_option('deprecation.showfileUploaderEncoding', False)


if 'upload_changed' not in st.session_state:
	st.session_state.upload_changed = False

if 'image_changed' not in st.session_state:
	st.session_state.image_changed = False

if 'model_changed' not in st.session_state:
	st.session_state.model_changed = False

if 'uploaded_image' not in st.session_state:
	st.session_state.uploaded_image = None

@st.cache(allow_output_mutation=True)
def load_model(model_name='U-Net_Densenet121'):
    if (model_name == 'U-Net_Vgg16'):
        model_name = 'models/' + str(model_name) + '.h5'
        model = tf.keras.models.load_model(model_name, custom_objects={'binary_crossentropy_plus_jaccard_loss': 
        sm.losses.bce_jaccard_loss,'iou_score':sm.metrics.IOUScore})
    else:
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

def predict_image(image, model):

    # This is for the U-Net_Vgg19 model with 69 layers
    if (len(model.layers) == 69):
        pred = model.predict(image)
        pred = np.rint(pred)
        output =pred[0,:,:,0]
    else:
        pred = model.predict(image)
        pred_argmax = np.argmax(pred,axis=3)
        output =pred_argmax[0,:,:]

    return output

def plot_input(uploaded_image):
    # Plot uploaded/selected image
    fig, ax = plt.subplots()
    ax.title.set_text('Uploaded Image')
    ax.imshow(uploaded_image, cmap='gray')
    st.pyplot(fig)

def plot_output(uploaded_image, model):
    # Format input image for model inferencing
    formatted_image = format_image(uploaded_image)

    # Obtain output image from model
    output_image = predict_image(formatted_image, model)

    # Plot output image
    fig2, ax2 = plt.subplots()
    ax2.title.set_text('Output Image')
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes('bottom', size='5%', pad=0.35)

        # This is for the U-Net_Vgg19 model with 69 layers
    if (len(model.layers) == 69):
        im2 = ax2.imshow(output_image, cmap='gray')
    else:
        im2 = ax2.imshow(output_image, cmap='jet',vmin=0, vmax=7)

    fig2.colorbar(im2, cax = cax2, orientation='horizontal')
    st.pyplot(fig2)

# Callback for when a different image is chosen from dropdown
def on_change_image():
    st.session_state.image_changed = True

# Callback for when a different image is chosen from file uploader
def on_change_uploader():
    st.session_state.upload_changed = True

# Callback for when a different model is chosen from dropdown
def on_change_model():
    st.session_state.model_changed = True

# Add a selectbox to the sidebar:
select_model = st.sidebar.selectbox(
    label = 'Which model would you like to try?',
    options = ('U-Net_Densenet121', 'Linknet_Inceptionv3', 'U-Net_Vgg16'),
    on_change = on_change_model
)

select_image = st.sidebar.selectbox(
    label = 'Choose an Image.',
    options = (None, 'train_6.png', 'train_7.png', 'train_8.png', 'train_9.png'),
    on_change=on_change_image
)
if (select_image is not None):
    select_image = 'images/' + select_image 

model = load_model(select_model)

st.title('Cell Segmentation Tool')


st.write("This is a simple cell segmentation tool that has been built using [tensorflow](https://www.tensorflow.org/) and [streamlit](https://streamlit.io/). The dataset used to train our models is the Colorectal Nuclear Segmntation and Phenotypes (CoNSeP) dataset introduced by [Graham et al.](https://www.sciencedirect.com/science/article/abs/pii/S1361841519301045)")


# File Uploader toolbar for uploading images
file = st.file_uploader("Upload an image of cells. Don't have an image? Choose a sample from the sidebar.", type=["jpg", "png"],
on_change=on_change_uploader)

# Image selected from dropdown
sample_file = select_image

if file is None and sample_file is None:
	st.text('Waiting for Image....')

else:
    slot = st.empty()
    slot.text('Running inference....')

    # Read uploaded image or selected imge from file object
    if (st.session_state.upload_changed and file is not None):
        print("UPLOAD CHANGED\n")
        st.session_state.uploaded_image = imread(file)
        uploaded_image = st.session_state.uploaded_image
        plot_input(uploaded_image)
        plot_output(uploaded_image, model)
        st.session_state.upload_changed = False

    elif (st.session_state.image_changed and sample_file is not None):
        print("IMAGE CHANGED\n")
        st.session_state.uploaded_image = imread(sample_file)
        uploaded_image = st.session_state.uploaded_image
        plot_input(uploaded_image)    
        plot_output(uploaded_image, model)
        st.session_state.image_changed = False

    else:
        print("MODEL CHANGED\n")
        uploaded_image = st.session_state.uploaded_image
        plot_input(uploaded_image)    
        plot_output(uploaded_image, model)
        st.session_state.model_changed = False



    slot.text('Completed Inference!')
