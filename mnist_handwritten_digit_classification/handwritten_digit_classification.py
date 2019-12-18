from numpy import mean
from numpy import std
from matplotlib import pyplot
from sklearn.model_selection import KFold
import keras
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.layers import BatchNormalization
from holoviews.examples.gallery.apps.bokeh.game_of_life import img
from sympy.ntheory.tests.test_bbp_pi import dig

# LOAD TRAIN AND TEST DATASETS
def load_dataset():
    # LOAD DATASET
    (trainX, trainY), (testX, testY) = mnist.load_data()
    # RESHAPE DATASET TO HAVE A SINGLE CHANNEL
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))
    # ONE-HOT ENCODE TARGET VALUES
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    return trainX, trainY, testX, testY


# SCALE PIXELS
def prep_pixels(train, test):
    # CONVERT FROM INTEGER TO FLOAT
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    # NORMALIZE TO RANGE 0-1
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    # RETURN NORMALIZED IMAGES
    return train_norm, test_norm


# DEFINE CNN MODEL
def define_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Dense(10, activation='softmax'))
    # COMPILE MODEL
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# EVALUATE A MODEL USING K-FOLD CROSS-VALIDATION
def evaluate_model(model, dataX, dataY, n_folds=5):
    scores, histories = list(), list()
    # PREPARE CORSS VALIDATION
    kfold = KFold(n_folds, shuffle=True, random_state=1)
    # ENUMERATE SPLITS
    for train_ix, test_ix in kfold.split(dataX):
        # SELECT ROWS FOR TRAIN AND TEST
        trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
        # FIT MODEL
        history = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY), verbose=1)
        #SAVE MODEL
        model.save('F:/EclipseWS/network/mnist_handwritten_digit_classification/final_model.h5')
        # EVALUATE MODEL
        _, acc = model.evaluate(testX, testY, verbose=1)
        print('> %.3f' % (acc * 100.0))
        # STORE SCORES
        scores.append(acc)
        histories.append(history)
    return scores, histories


# PLOT DIAGNOSTIC LEARNING CURVES
def summarize_diagnostics(histories):
    for i in range(len(histories)):
        # plot loss
        pyplot.subplot(211)
        pyplot.title('Cross Entropy Loss')
        pyplot.plot(histories[i].history['loss'], color='blue', label='train')
        pyplot.plot(histories[i].history['val_loss'], color='orange', label='test')
        # plot accuracy
        pyplot.subplot(212)
        pyplot.title('Classification Accuracy')
        pyplot.plot(histories[i].history['acc'], color='blue', label='train')
        pyplot.plot(histories[i].history['val_acc'], color='orange', label='test')
    pyplot.show()
    
    
# SUMMARIZE MODEL PERFORMANCE
def summarize_performances(scores):
    # PRINT SUMMARY
    print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores)*100, std(scores)*100, len(scores)))
    # box and whisker plots of results
    pyplot.boxplot(scores)
    pyplot.show()  
    
    
# RUN THE TEST HARNESS FOR EVALUATING A MODEL
def run_test_harness():
    # LOAD DATASET
    trainX, trainY, testX, testY = load_dataset()
    # PREPARE PIXEL DATA
    trainX, testX = prep_pixels(trainX, testX)
    # DEFINE MODEL
    model = define_model()
    # EVALUATE MODEL
    scores, histories = evaluate_model(model, trainX, trainY)
    # LEARNING CURVES
    summarize_diagnostics(histories)
    # SUMMARIZE ESTIMATED PERFORMANCE
    summarize_performances(scores)

# ENTRY POINT, RUN THE TEST HARNESS
run_test_harness()
