"""
    Fashion-MNIST, It is a dataset comprised of 60,000 small square 28*28 pixel grayscale images of items of 10 type of clothing.
    The mapping of all 0-9 integers to class labels.
    0: Tshirt/top
    1: Trouser
    2: Pullover
    3: Dress
    4: Coat
    5: Sandal
    6: Shirt
    7: Sneaker
    8: Bag
    9: Ankle boot
"""


from numpy import mean
from numpy import std
from matplotlib import pyplot

from sklearn.model_selection import KFold

from keras.datasets import fashion_mnist
from keras.utils import to_categorical

from keras.models import Sequential

from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
 
from keras.optimizers import SGD

# LOAD DATASET
(trainX, trainy), (testX, testy) = fashion_mnist.load_data()
# SUMMARIZE LOADED DATASET
print('Train: X=%s, y=%s' % (trainX.shape, trainy.shape))
print('Test: X=%s, y=%s' % (testX.shape, testy.shape))
# PLOT FIRST FEW IMAGES
for i in range(9):
    # define subplot
    pyplot.subplot(330 + 1 + i)
    # plot raw pixel data
    pyplot.imshow(trainX[i], cmap=pyplot.get_cmap('gray'))
# show the figure
pyplot.show()

# LOAD TRAIN AND TEST DATASET
def load_dataset():
    # LOAD DATASET
    (trainX, trainY), (testX, testY) = fashion_mnist.load_data()
    # RESHAPE DATASET TO HAVE A SINGLE CHANNEL
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))
    # ONE HOT ENCODE TARGE VALUE
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    return trainX, trainY, testX, testY
 
# SCALE PIXELS
def prep_pixels(train, test):
    # CONVERT FROM INTEGER TO FLOATS
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
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    # COMPILE MODEL
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model
 
# EVALUATE A MODEL USING K-FOLD CROSS VALIDATION
def evaluate_model(dataX, dataY, n_folds=5):
    scores, histories = list(), list()
    # PREPARE CROSS VALIDATION
    kfold = KFold(n_folds, shuffle=True, random_state=1)
    # ENUMERATE SPLITS
    for train_ix, test_ix in kfold.split(dataX):
        # DEFINE MODEL
        model = define_model()
        # SELECT ROWS FOR TRAIN AND TEST
        trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
        # FIT MODEL
        history = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY), verbose=1)
        #SAVE MODEL
        model.save('F:/EclipseWS/network/fashion_mnist_clothing_classification_CNN/final_model.h5')
        # EVALUATE MODEL
        _, acc = model.evaluate(testX, testY, verbose=1)
        print('> %.3f' % (acc * 100.0))
        # APPEND SCORES
        scores.append(acc)
        histories.append(history)
    return scores, histories
 
# PLOT DIAGNOSTIC LEARNING CURVES
def summarize_diagnostics(histories):
    for i in range(len(histories)):
        # PLOT LOSS
        pyplot.subplot(211)
        pyplot.title('Cross Entropy Loss')
        pyplot.plot(histories[i].history['loss'], color='blue', label='train')
        pyplot.plot(histories[i].history['val_loss'], color='orange', label='test')
        # PLOT ACCURACY
        pyplot.subplot(212)
        pyplot.title('Classification Accuracy')
        pyplot.plot(histories[i].history['acc'], color='blue', label='train')
        pyplot.plot(histories[i].history['val_acc'], color='orange', label='test')
    pyplot.show()
 
# SUMMARIZE MODEL PERFORMANCE
def summarize_performance(scores):
    # PRINT SUMMARY
    print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores)*100, std(scores)*100, len(scores)))
    # BOX AND WHISKER PLOTS OF RESULTS
    pyplot.boxplot(scores)
    pyplot.show()
 
# RUN THE TEST HARNESS FOR EVALUATING A MODEL
def run_test_harness():
    # LOAD DATASET
    trainX, trainY, testX, testY = load_dataset()
    # PREPARE PIXEL DATA
    trainX, testX = prep_pixels(trainX, testX)
    # EVALUATE MODEL
    scores, histories = evaluate_model(trainX, trainY)
    # LEARNING CURVES
    summarize_diagnostics(histories)
    # SUMMARIZE ESTIMATED PERFORMANCE
    summarize_performance(scores)
 
# ENTRY POINT, RUN TEST HARNESS
run_test_harness()
