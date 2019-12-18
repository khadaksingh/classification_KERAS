""" PLOT DOG PHOTOS FROM THE DOGS VS CATS DATA-SET """
from matplotlib import pyplot
from matplotlib.image import imread
from shutil import copyfile
from keras.applications.vgg16 import VGG16


# DEFINE LOCATION OF DATA-SET
folder = 'F:/EclipseWS/network/cat_dog_classification_problem/data/dogs-vs-cats/train/'
# PLOT FIRST FEW IMAGES
for i in range(9):
    # DEFINE SUB-PLOT
    pyplot.subplot(330 + 1 + i)
    # DEFINE FILE-NAME
    filename = folder + 'dog.' + str(i) + '.jpg'
    # LOAD IMAGE PIXEL
    image = imread(filename)
    # PLOT RAW PIXEL DATA
    pyplot.imshow(image)
# SHOW THE FIGURE
pyplot.show()


""" PLOT CAT PHOTOS FROM THE DOGS VS CAT DATA-SET """

# DEFINE LOCATION OF DATA-SET
folder = 'F:/EclipseWS/network/cat_dog_classification_problem/data/dogs-vs-cats/train/'
# PLOT FIRST FEW IMAGES
for i in range(9):
    # DEFINE SUB-PLOT
    pyplot.subplot(330 + 1 + i)
    # DEFINE FILE-NAME
    filename = folder + 'cat.' + str(i) + '.jpg'
    # LOAD IMAGE PIXEL
    image = imread(filename)
    # PLOT RAW PIXEL DATA
    pyplot.imshow(image)
# SHOW THE FIGURE
pyplot.show()


""" LOAD DOGS VS CATS DATA-SET, RE-SHAPE AND SAVE TO A NEW FILE """
from os import listdir
from numpy import asarray
from numpy import save
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array


# DEFINE LOCATION OF DATA-SET
folder = 'F:/EclipseWS/network/cat_dog_classification_problem/data/dogs-vs-cats/train/'
photos, labels = list(), list()
# ENUMERATE FILES IN THE DIRECTORY
for file in listdir(folder):
    # DETERMINE CLASS
    output = 0.0
    if file.startswith('cat'):
        output = 1.0
    # LOAD IMAGE
    photo = load_img(folder + file, target_size=(200, 200))
    # CONVERT TO NUMPY ARRAY
    photo = img_to_array(photo)
    # STORE
    photos.append(photo)
    labels.append(labels)
    
# CONVERT TO NUMPY ARRAY
photos = asarray(photos)
labels = asarray(labels)
print(photos.shape, labels.shape)

# SAVE THE RE-SHAPED PHOTOS
save('dogs_vs_cats_photos.npy')
save('dogs_vs_cats_labels.npy')

# LOAD DATA AND CONFIRM THE SHAPE
from numpy import load
photos = load('dogs_vs_cats_photos.npy')
labels = load('dogs_vs_cats_labels.npy')
print(photos.shape, labels.shape)


from os import makedirs
from os import listdir
from shutil import copyfile
from random import seed
from random import random

# CREATE DIRECTORIES
dataset_home = 'dataset_dogs_vs_cats/'
subdirs = ['train/', 'test/']
for subdir in subdirs:
    # CREATE LABEL SUB-DIRECTORIES
    labeldirs = ['dogs/', 'cats/']
    for labeldir in labeldirs:
        newdir = dataset_home + subdir + labeldir
        makedirs(newdir, exist_ok=True)
        
# SEEN RANDOM NUMBER GENERATOR
seed(1)
# DEFINE RATIO OF PICTURES TO USE FOR VALIDATION
val_ratio = 0.25
# COPY TRAINING DATA-SET IMAGES INTO SUB-DIRECTORIES
src_directory = 'train/'
for file in _listdir(src_directory):
    src = src_directory + '/' + file
    dst_dir = 'train/'
    if random() < val_ratio:
        dst_dir = 'test/'
        if file.startswith('cat'):
            dst = dataset_home + dst_dir + 'cats/' + file
            copyfile(src, dst)
        elif file.startswith('dog'):
            dst = dataset_home + dst_dir + 'dogs/' + file
            copyfile(src, dst)
  
  
import sys
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

 # DEVELOP A BASELINE CNN
def define_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
    model.add(MaxPool2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(1, activation='sigmoid'))
    
    # COMPILE MODEL
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model
          
    
# TRANSFER LEARNING DEMO:::      
def define_trained_model():
    # LOAD MODEL
    model = VGG16(include_top=False, input_shape=(224, 224, 3))
    # MARK LOADED LAYERS AS NOT TRAINABLE
    for layer in model.layers:
        layer.trainable = False
    # ADD NEW CLASSIFIER LAYERS
    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
    output = Dense(1, activation='sigmoid')(class1)
    # DEFINE NEW MODEL
    model = Model(inputs=model.inputs, outputs=output)
    # COMPILE NEW MODEL
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrices=['accuracy'])
    return model


# PLOT DIAGNOSTIC LEARNING CURVES
def summarize_diagnostics(history):
    # PLOT LOSS
    pyplot.subplot(211)
    pyplot.title('Cross Entropy Loss')
    pyplot.plot(history.history['loss'], color='blue', label='train')
    pyplot.plot(history.history['val_loss'], color='orange', label='test')
    
    # PLOT ACCURACY
    pyplot.subplot(212)
    pyplot.title('Classificaton Accuracy')
    pyplot.plot(history.history['acc'], color='blue', label='train')
    pyplot.plot(history.history['val_acc'], color='orange', label='test')
    
    # SAVE PLOT TO FILE
    filename = sys.argv[0].split('/'[-1])
    pyplot.savefig(filename + '_plot.png')
    

# RUN THE TEST HARNESS FOR EVALUATING A MODEL
def run_test_harness():
    # DEFINE MODEL
    model = define_model()
    
    # CREATE DATA GENERATOR
    train_datagen = ImageDataGenerator(rescale=1.0 / 255.0,
                                 width_shift_range=0.1,
                                 height_shift_range=0.1,
                                 horizontal_flip=True)
    
    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    
    # PREPARE ITERATORS
    train_it = train_datagen.flow_from_directory('dataset_dogs_vs_cats/train/', class_mode='binary', batch_size=64, target_size=(200, 200))
    test_it = test_datagen.flow_from_directory('dataset_dogs_vs_cats/test/', class_mode='binary', batch_size=64, target_size=(200, 200))
    
    # FIT MODEL
    history = model.fit_generator(train_it, steps_per_epoch=len(train_it), validation_data=test_it, validation_steps=len(test_it), epochs=20, verbose=1)
    
    # SAVE MODEL
    model.save('F:/EclipseWS/network/cat_dog_classification_problem/final_model.h5')
    
    # EVALUATE MODEL
    _, acc = model.evaluate_generator(test_it, steps=len(test_it), verbose=1)
    print('> %.3f' % (acc * 100.0))
    
    # LEARNING CURVES
    summarize_diagnostics(history)
    

# ENTRY POINT, RUN THE TEST HARNESS
run_test_harness()

""" MAKE PREDICTION PART-- 
    THE MODEL ASSUMES THAT NEW IMAGES ARE COLOR AND THEY HAVE BEEN SEGMENTED SO THAT ONE IMAGE CONTAINS AT LEAST ONE DOG OR CAT
    """

from keras.models import load_model

# LOAD AND PREPARE THE IMAGES
def load_image(filename):
    # LOAD THE IMAGE
    img = load_img(filename, target_size=(224, 224))
    # CONVERT TO ARRAY
    img = img_to_array(img)
    # RE-SHAPE INTO A SINGLE SAMPLE WITH 3 CHANNELS
    img = img.reshape(1, 224, 224, 3)
    # CENTER PIXEL DATA
    img = img.astype('float32')
    img = img - [123.68, 116.779, 103.939]
    return img

# LOAD AN IMAGE AND PREDICT THE CLASS
def run_example():
    # LOAD THE IMAGE
    img = load_image('F:/EclipseWS/network/cat_dog_classification_problem/sample_image.jpg')
    # LOAD MODEL
    model = load_model('F:/EclipseWS/network/cat_dog_classification_problem/final_model.h5')
    # PREDICT THE CLASS
    result = model.predict(img)
    print(result[0])

# ENTRY POINT, RUN THE EXAMPLE
run_example()
