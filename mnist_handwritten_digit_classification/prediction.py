# MAKE A PREDICTION FOR A NEW IMAGE
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model

# LOAD AND PREPARE THE IMAGE
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
 
# load an image and predict the class
def run_example():
    # LOAD THE IMAGE
    img = load_image('F:/EclipseWS/network/mnist_handwritten_digit_classification/test_image/another_test_image.png')
    #img = load_image('F:/EclipseWS/network/mnist_handwritten_digit_classification/test_image/test_image.png')
    # LOAD MODEL
    model = load_model('F:/EclipseWS/network/mnist_handwritten_digit_classification/final_model.h5')
    # PREDICT THE CLASS
    digit = model.predict_classes(img)
    print('THE INPUT IMAGE DIGIT IS : ', digit[0])
 
# ENTRY POINT, RUN THE EXAMPLE
run_example()