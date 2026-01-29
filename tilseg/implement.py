'''
2. IMPLEMENT 3-CLASS CLASSIFICATION
[descriptions]
'''
import time
import os
import numpy as np
import gc
import glob

from keras.layers import Dropout, Flatten, Dense
from keras.applications import VGG19
from keras.models import Sequential
from keras import utils
import tensorflow as tf
import PIL
from PIL import ImageDraw, Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    gpu_details = tf.config.experimental.get_device_details(gpus[0])

    if gpu_details.get('memory_size', 0) >= 9560 * 1024 * 1024:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=9560)]
        )
        print(f"GPU configured with 9560 MB memory limit.")
    else:
        print("Insufficient GPU memory. Using CPU instead.")
else:
    print("No GPU available. Using CPU.")

logical_gpus = tf.config.list_logical_devices('GPU')
print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")

dx = 50
dy = 50

def paths(mainpath, inpath):
    global mainPath
    global INPATH
    mainPath = mainpath
    INPATH = inpath
    return None

def constructVGGModel():
    res_conv = VGG19(weights='imagenet', include_top=False, input_shape=(48, 48, 3))

    # get the directory where the current script is located (i.e. tilseg)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # go up one level (i.e. TILseg repository)
    parent_dir = os.path.dirname(script_dir)
    
    # Construct the model path relative to grandparent directory
    modelPath = os.path.join(
        parent_dir, 
        "models", 
        "3CC_independent_val.h5" # TODO (to use another model, please change the string to the name of the desired model)
    )

    res_conv = VGG19(weights='imagenet', include_top=False, input_shape=(48, 48, 3))

    if modelPath:
        # Freeze the layers except the last 12 layers
        # for layer in res_conv.layers[:-12]:
        #     layer.trainable = False
        # Create the model
        global model1
        model1 = Sequential()
        # Add the res convolutional base model
        model1.add(res_conv)
        # Add new layers
        model1.add(Flatten())
        model1.add(Dense(1024, activation='relu'))
        model1.add(Dropout(0.2))
        model1.add(Dense(3, activation='softmax'))
        model1.load_weights(modelPath)
    else:
        print("No 3CC model is found.")

def predictImage(test_image):
    test_image = utils.img_to_array(test_image)
    test_image = test_image/255
    test_image = np.expand_dims(test_image, axis = 0)
    result = model1.predict(test_image, verbose=0)
    return result.argmax(1)

'''
The inputs are:
* DataColoredTileClassification = the original images
* ColoredMasks = the masks with 3 colors for the original images.
The colors correspond to stroma, epithelium and others.
Gets the most common element from a list.
Used for cleaning the colored mask image for epithelium/stroma/others
'''
def mostCommon(lst):
    return max(set(lst), key=lst.count)
'''
Receives a matrix of predictions and returns a
matrix in which the singleton classes
are replaced by the common classes around them.
'''
def cleanMatrixClasses(matrix):
    m = matrix
    X, Y = matrix.shape

    classesAround = lambda x, y : [matrix[x2][y2] for x2 in range(x-1, x+2)
                                for y2 in range(y-1, y+2)
                                if (-1 < x < X and
                                    -1 < y < Y and
                                    (x != x2 or y != y2) and
                                    (0 <= x2 < X) and
                                    (0 <= y2 < Y))]
    for j in range(Y):
        for i in range(X):
            if len(classesAround(i, j)) == 3: #if it is in the corner
                #if there no similar tile
                if classesAround(i, j).count(matrix[i][j]) == 0:
                    m[i][j] = mostCommon(classesAround(i, j))
            else: #there are 5 or 8 neighbors
                #if there one single tile or 0, change it to the majority
                if classesAround(i, j).count(matrix[i][j]) < 3:
                    m[i][j] = mostCommon(classesAround(i, j))
        return m

def drawColoredRectangles(draw, x, y, w, h, predictedOutput):
    if predictedOutput == -1:
        draw.rectangle([(x,y),x + dx, y + dy], fill = 'black' )
    elif predictedOutput == 1:
        draw.rectangle([(x,y),x + dx, y + dy], fill = 'darkgreen' )
    elif predictedOutput == 0:
        draw.rectangle([(x,y),x + dx, y + dy], fill = 'blue' )
    elif predictedOutput == 2:
        draw.rectangle([(x,y),x + dx, y + dy], fill = 'red' )
    return draw

#a tile is good if at least percentPixel% are below 230,
#that is they are not only white
#uses PIL as input
def goodTile(im, percentPixels = 90):
    good = True
    pix = np.array(im, dtype=float)
    mask = (pix>230).all(axis=-1)           # boolean mask where True = nearly white pixels (i.e. background)
    pix[mask] = 1                           # markes background pixels = True

    w, h = mask.shape

    if (np.sum(mask) / (w * h)) > (percentPixels / 100): # if the number of background pixels > %Pixels --> the patch is marked as bad
        good = False
    return good

'''
Get the original image, make a mask that identifies light pixels (>230),
add this mask to the colored mask image and return the obtained image.
'''
def combineOriginalAndMaskToEliminateWhite(imOrig, imMask):
    pix = np.array(imOrig, dtype=float)
    pixMask = np.array(imMask, dtype=float)
    mask = (pix>230).all(axis=-1)               # boolean mask where True = nearly white pixels (i.e. background)
    pixMask[mask] = 1                           # marks background pixels = True
    return Image.fromarray(np.uint8(pixMask))

'''
Receives a fileName and makes a matrix of classes for it.
'''

def makeClassMatrixForOneFile(fileName, batch_size=64):
    
    startImagePred = time.time()
    
    im = PIL.Image.open(fileName)
    w, h = im.size
    timesInWidth = int(w / dx) + (w % dx > 0)
    timesInHeight = int(h / dy) + (h % dy > 0)
    noOfTilesTotal = timesInWidth * timesInHeight
    
    print(f"{timesInWidth} x {timesInHeight} tiles → {noOfTilesTotal} predictions for {fileName}")
   
    matrixOfClasses = np.zeros((timesInWidth, timesInHeight), dtype=int) # hold the class prediction for each tile
    
    # holding image arrays for batch predictions
    batch_images = []
    batch_coords = [] # stores index of each kernel
    
    # iterate over each 50x50 kernel
    for x1, x in enumerate(range(0, w - dx + 1, dx)):
        for y1, y in enumerate(range(0, h - dy + 1, dy)):
            croppedImage = im.crop((x, y, x + dx, y + dy)).resize((48, 48))
            if goodTile(croppedImage): # checks if mostly white/empty
                img_array = utils.img_to_array(croppedImage) / 255.0
                batch_images.append(img_array)
                batch_coords.append((x1, y1))
            else:
                matrixOfClasses[x1][y1] = -1  # mark as bad tile
            
            # predict once batch is full
            if len(batch_images) >= batch_size:
                predictions = model1.predict(np.array(batch_images), verbose=0)
                predicted_classes = predictions.argmax(axis=1) # get predictions for each tile
                for (i, j), pred in zip(batch_coords, predicted_classes):
                    matrixOfClasses[i][j] = pred
                batch_images.clear()
                batch_coords.clear()
                gc.collect() # garbage collection
    
    # process any remaining kernels in the batch
    if batch_images:
        predictions = model1.predict(np.array(batch_images), verbose=0)
        predicted_classes = predictions.argmax(axis=1)
        for (i, j), pred in zip(batch_coords, predicted_classes):
            matrixOfClasses[i][j] = pred
        batch_images.clear()
        batch_coords.clear()
        gc.collect()
        
    endImagePred = time.time()
    return matrixOfClasses, endImagePred - startImagePred

# NOT BATCHED
# def makeClassMatrixForOneFile(fileName):
#     startImagePred = time.time()
#     im = PIL.Image.open(fileName)
#     w, h = im.size
#     timesInWidth = int(w / dx) + (w % dx > 0)
#     timesInHight = int(h / dy) + (h % dy > 0)
#     noOfTilesTotal = timesInWidth * timesInHight
#     print(f'{timesInWidth:.2f} times in width and {timesInHight:.2f} in height, '
#     f'that is {noOfTilesTotal:.2f} applications for the main part.')
#     print(f'This will take approx {(noOfTilesTotal * 0.035):.2f} seconds, '
#     f'that is {((noOfTilesTotal * 0.035) / 60):.2f} minutes.')

#     matrixOfClasses = np.zeros((timesInWidth, timesInHight))
#     x = 0
#     y = 0

#     x1 = 0
#     y1 = 0
    
#     while x <= w - dx:
#         while y <= h - dy:
#             croppedImage = im.crop((x, y, x + dx, y + dy))
#             croppedImage = croppedImage.resize((48,48))
#             # predictedOutput = predictImage(croppedImage)
#             if goodTile(croppedImage):
#                 predictedOutput = predictImage(croppedImage)
#             else:
#                 predictedOutput = -1 #will be colored as black
#             matrixOfClasses[x1][y1] = predictedOutput
#             y = y + dy
#             y1 += 1
#         x = x + dx
#         x1 += 1
#         y = 0
#         y1 = 0

#     endImagePred = time.time()

#     matrixOfClasses = matrixOfClasses.astype(int)

#     return matrixOfClasses, endImagePred - startImagePred
'''
Next method receives the fileName and the matrix
to make the colored mask.
'''
def makeColoredMaskFromMatrix(fileName, m):
    startImagePred = time.time()
    im = PIL.Image.open(fileName)
    x = 0
    y = 0
    w, h = im.size
    maskImage = PIL.Image.new("RGB", (w, h), (0, 0, 0))
    draw = ImageDraw.Draw(maskImage)
    x = 0
    y = 0
    x1 = 0
    y1 = 0
    while x <= w - dx:
        while y <= h - dy:
            drawColoredRectangles(draw, x, y, x + dx, y + dy, m[x1][y1])
            y = y + dy
            y1 += 1
        x = x + dx
        x1 += 1
        y = 0
        y1 = 0

    del draw
    endImagePred = time.time()
    return maskImage, endImagePred - startImagePred

'''
Gets the initial H&E files for which the K Means in
applied and the clustered images are saved.
It also checks if the clustered images are
already produced to avoid overwork.
'''

def makeColoredMasks(inputFolder, batch_size=3):
    files = [file for file in os.listdir(inputFolder) if file.endswith('tif')]
    numOfImages = len(files)
    print('There are ', numOfImages,'to process in the input folder: ', inputFolder)

    # Split files into batches
    batches = [files[i:i+batch_size] for i in range(0, numOfImages, batch_size)]

    for batch in batches:
        for file in batch:
            mask_filename = 'Classified_' + file[:-4] + '.tif'

            # Org x mask output
            out_3cc = os.path.join(mainPath, '3class')
            os.makedirs(out_3cc, exist_ok=True)
            class_path = os.path.join(out_3cc, mask_filename)

            # Mask output:
            out_mask = os.path.join(mainPath, 'raw_3class')
            os.makedirs(out_mask, exist_ok=True)
            mask_path = os.path.join(out_mask, mask_filename)

            if os.path.isfile(mask_path):
                print('File', file, 'already has a colored mask.')
            else:
                print('Making the colored mask for', file,'starting at', time.ctime())
                start = time.time()
                matrix, _ = makeClassMatrixForOneFile(os.path.join(inputFolder, file))
                maskImage, _ = makeColoredMaskFromMatrix(os.path.join(inputFolder, file), cleanMatrixClasses(matrix))
                imOrig = PIL.Image.open(os.path.join(inputFolder, file))
                obtainedMaskImage = combineOriginalAndMaskToEliminateWhite(imOrig, maskImage)
                obtainedMaskImage.save(class_path)
                maskImage.save(mask_path)
                endTime = time.time()
                print('Producing the mask for ', file,'took ', (endTime - start),'seconds.')

        # Free up memory after processing the batch
        gc.collect()

def implement(mainPath):
    _ = constructVGGModel()
    INPATH = os.path.join(mainPath, "patches")
    # Globalize variables
    paths(mainPath, INPATH)
    # Process extracted patches (.tif)
    _ = makeColoredMasks(INPATH)

    return None
