# MAKE A PREDICTION FOR A NEW IMAGES
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model

# LOAD AND PREPARE THE IAMGES
def load_image(filename):
    # LOAD THE IMAGE
    img = load_img(filename, grayscale=True, target_size=(28, 28))
    # CONVERT TO ARRAY
    img = img_to_array(img)
    # RESHAPE INTO A SINGLE SAMPLE WITH 1 CHANNEL
    img = img.reshape(1, 28, 28, 1)
    # PREPARE PIXEL DATA
    img = img.astype('float32')
    img = img / 255.0
    return img

# LOAD AN IMAGE AND PREDICT THE CLASS
def run_example():
    # LOAD THE IMAGE
    img = load_image('F:/EclipseWS/network/fashion_mnist_clothing_classification_CNN/test_image/sample_image.png')
    # LOAD MODEL
    model = load_model('F:/EclipseWS/network/fashion_mnist_clothing_classification_CNN/final_model.h5')
    # PREDICT THE CLASS
    result = model.predict_classes(img)
    print('Predicted Values is : ', result[0])
    
# ENTRY POINT, RUN THE EXAMPLE
run_example()
