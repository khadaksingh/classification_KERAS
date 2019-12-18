""" WE CAN LOAD AND SUMMARIZE VGG-16 MODEL """
# LOAD VGG MODEL
from keras.applications.vgg16 import VGG16
from matplotlib import pyplot
from keras.preprocessing.image import img_to_array

# LOAD THE MODEL
model = VGG16()

# SUMMARIZE THE MODEL
model.summary()

""" WE CAN ACCESS ALL OF THE LAYERS OF THE MODEL VIA THE model.layers PROPERTY 
    EACH LAYER HAS A layer.name PROPERTY, WHERE THE CONVOLUTIONAL LAYERS HAVE A
    NAMING CONVOLUTIONAL LIKE block#_conv# WHRE # IS AN INTEGER i.e. WE CAN CHECK
    THE NAME OF EACH LAYER AND SKIP ANY THAT DON'T CONTAIN THE STRING 'conv' 
    """

# SUMMARIZE FILTER SHAPES
for layer in model.layers:
    # CHECK FOR CONVOLUTIONAL LAYER
    if 'conv' not in layer.name:
        continue
    
    """ EACH CONVOLUTIONAL LAYERS HAS TWO SETS OF WEIGHTS
        ONE IS block_of_filter and the
        OTHER IS THE block_of_bias VALUES
        THESE ARE ACCESSIBLE VIA THE layer.get_weights() FUNCTION 
        WE CAN RETRIEVE THESE WEIGHTS AND THEN SUMMARIZE THEIR SHAPE
    """
    # GET FILTER WEIGHTS
    filters, biases = layer.get_weights()
    print(layer.name, '-', filters.shape)
    
    

# RETRIEVE WEIGHTS FROM THE SECOND HIDDEN LAYER
filters, biases = model.layers[1].get_weights()

# NORMALIZE FILTER VALUES TO 0-1 SO WE CAN VISUALIZE THEM
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)


# PLOT FIRST FEW FILTERS
n_filters, ix = 6, 1
for i in range(n_filters):
    # GET THE FILTER
    f = filters[:, :, :, i]
    # PLOT EACH CHANNEL SEPRATELY
    for j in range(3):
        # SPECIFY SUB-PLOT AND TURNS OF AXIS
        ax = pyplot.subplot(n_filters, 3, ix)
        ax.set_xticks([])
        ax.set_yticks([])
        # PLOT FILTER CHANNEL IN GRAYSCALE
        pyplot.imshow(f[:, :, j], cmap='gray')
        ix += 1
# SHOW THE FIGURE
pyplot.show()

print('-----------------------------------------------------------------------')

""" VISUALIZE FEATURE MAP """
# SUMMARIZE FEATURE MAP SIZE FOR EACH CONV LAYER
from keras.applications.vgg16 import VGG16
from matplotlib import pyplot
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import load_img
from keras.models import Model
from numpy import expand_dims

# LOAD THE MODEL
model = VGG16()

# SUMMARIZE FEATURE MAP SHAPES
for i in range(len(model.layers)):
    layer = model.layers[i]
    # CHECK FOR CONVOLUTIONAL LAYER
    if 'conv' not in layer.name:
        continue
    # SUMMARIZE OUTPUT SHAPE
    print(i, layer.name, layer.output.shape)


# RE-DEFINE MODEL TO OUTPUT RIGHT AFTER THE FIRST HIDDEN LAYER
model = Model(inputs=model.inputs, outputs=model.layers[1].output)

# LOAD THE IMAGE WITH THE REQUIRED SHAPE
img = load_img('F:/EclipseWS/network/visualize_filters_and_feature_map_CNN/bird.jpg', target_size=(224, 224))

# CONVERT THE IMAGE TO AN ARRAY
# EXPAND DIMENSION SO THAT IT REPRESENTS A SINGLE 'SAMPLE'
img = expand_dims(img, axis=0)

# PREPARE THE IMAGE (e.g. SCALE PIXEL VALUES FOR THE VGG)
img = preprocess_input(img)

# GET FEATURE MAP FOR THE FIRST HIDDEN LAYER
feature_maps = model.predict(img)

""" WE KNOW THE RESULT WILL BE A FEATURE MAP WITH 224*224*64. WE CAN PLOT ALL 64 TWO-DIMENSIONAL IMAGES AS AN 8*8 SQUARE OF IMAGE"""
# PLOT ALL 64 MAPS IN AN 8*8 SQUARES
square = 8
ix = 1
for _ in range(square):
    # SPECIFY SUB-PLOT AND TURN OF AXIS
    ax = pyplot.subplot(square, square, ix)
    ax.set_xticks([])
    ax.set_yticks([])
    # PLOT FILTER CHANNEL IN GRAY-SCALE
    pyplot.imshow(feature_maps[0, :, :, ix-1], cmap='gray')
    ix += 1
# SHOW THE FIGURE
pyplot.show()

print('-------------------------------------------------------------------')

# LOAD THE MODEL
model = VGG16()
# RE-DEFINE MODEL TO OUTPUT RIGHT AFTER THE FIRST HIDDEN LAYER
ixs = [2, 5, 9, 13, 17]
outputs = [model.layers[i].output for i in ixs]
model = Model(inputs=model.inputs, outputs=outputs)
# LOAD THE IMAGE WITH THE REQUIRED SHAPE
img = load_img('F:/EclipseWS/network/visualize_filters_and_feature_map_CNN/bird.jpg', target_size=(224, 224))
# CONVERT THE IMAGE TO AN ARRAY
img = img_to_array(img)
# EXPAND DIMENSIONS SO THAT IT REPRESENT A SINGLE 'SAMPLE'
img = expand_dims(img, axis=0)
# PREPARE THE IMAGE (e.g. SCALE PIXEL VALUES FOR THE VGG)
img = preprocess_input(img)
# GET FEATURE MAP FOR FIRST HIDDEN LAYER
feature_maps = model.predict(img)
# PLOT THE OUTPUT FOR EACH BLOCK
square = 8
for fmap in feature_maps:
    # PLOT ALL 64 MAPS IN AN 8*8 SQUARES
    ix = 1
    for _ in range(square):
        for _ in range(square):
            # SPECIFY SUBPLOT AND TURN OF AXIS
            ax = pyplot.subplot(square, square, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # PLOT FILTER CHANNEL IN GRAY-SCALE
            pyplot.imshow(fmap[0, :, :, ix-1], cmap='gray')
            ix += 1
    # SHOW THE FIGURE
    pyplot.show()
    
