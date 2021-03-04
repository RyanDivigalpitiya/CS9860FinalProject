#Ryan Divigalpitiya | CS9860 Final Project | Western University
## PACKAGE IMPORTS ##
from scipy.interpolate import griddata
from matplotlib import pyplot as plt
from matplotlib import colors
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow.python.client import device_lib
import tensorflow.keras
from tensorflow.keras import backend
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
# from tensorflow.python.framework import ops
# ops.reset_default_graph()
print(device_lib.list_local_devices())
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
# tf.debugging.set_log_device_placement(True)

## GLOBAL CONSTANTS ########################################################################################
max_pixel_value = 115 #found max value of images, placed here for reference

## FUNCTION DEFINITIONS ####################################################################################
## For future re-factor: these function definitions should be placed in different file and imported ########
############################################################################################################
def showThermalImage(image, filterNoise=False, filterThreshold=None):
    ## showThermalImage() displays an instance of a thermal image using the below defined color maps ##
    red_values = np.linspace(0,255,27)
    cmap = colors.ListedColormap([
                                    [red_values[0]/255,  0/255 , 0/255],
                                    [red_values[1]/255,  0/255 , 0/255],
                                    [red_values[2]/255,  0/255 , 0/255],
                                    [red_values[3]/255,  0/255 , 0/255],
                                    [red_values[4]/255,  0/255 , 0/255],
                                    [red_values[5]/255,  0/255 , 0/255],
                                    [red_values[6]/255,  0/255 , 0/255],
                                    [red_values[7]/255,  0/255 , 0/255],
                                    [red_values[8]/255,  0/255 , 0/255],
                                    [red_values[9]/255,  0/255 , 0/255],
                                    [red_values[10]/255,  0/255 , 0/255],
                                    [red_values[11]/255,  0/255 , 0/255],
                                    [red_values[12]/255,  0/255 , 0/255],
                                    [red_values[13]/255,  0/255 , 0/255],
                                    [red_values[14]/255,  0/255 , 0/255],
                                    [red_values[15]/255,  0/255 , 0/255],
                                    [red_values[16]/255,  0/255 , 0/255],
                                    [red_values[17]/255,  0/255 , 0/255],
                                    [red_values[18]/255,  0/255 , 0/255],
                                    [red_values[19]/255,  0/255 , 0/255],
                                    [red_values[20]/255,  0/255 , 0/255],
                                    [red_values[21]/255,  0/255 , 0/255],
                                    [red_values[22]/255,  0/255 , 0/255],
                                    [red_values[23]/255,  0/255 , 0/255],
                                    [red_values[24]/255,  0/255 , 0/255],
                                    [red_values[25]/255,  0/255 , 0/255],
                                    [red_values[26]/255,  0/255 , 0/255],
                                ])

    colorBoundaries = list(np.linspace(40,max_pixel_value,26))
    colorBoundaries.insert(0,0)

    if filterNoise:
        filteredImage = highPassFilter(image,filterThreshold)
        norm = colors.BoundaryNorm(colorBoundaries, cmap.N)
        plt.pcolor(filteredImage, cmap=cmap, norm=norm)
        plt.show()
    else:
        norm = colors.BoundaryNorm(colorBoundaries, cmap.N)
        plt.pcolor(image, cmap=cmap, norm=norm)
        plt.show()

def calcScaledImageDimensions(scalingFactor, ogImage):
    ## when given an image (ogImage) and a scaling factor (scalingFactor),
    # this function will calculate the dimensions of the new scaled image
    # (assumes ogImage is a square matrix)
    # this function is a helper function that is used in imageResolutionScaler() ##
    return ogImage.shape[0]*(scalingFactor+1)-scalingFactor

def imageResolutionScaler(image,scalingFactor,interpolationMethod):
    ## imageResolutionScaler() is my custom-made resolution upscaler function.
    # genDataCoordsInScaledImage() below is a helper function that that determines the coordinates of
    # where the orignal pixels belong in the new upscaled image
    def genDataCoordsInScaledImage(originalImage,scalingFactor): #returns data_x_y_coords_in_ScaledImage
        data_x_y_coords_in_ScaledImage = np.zeros((originalImage.shape[0]*originalImage.shape[1],2))
        dataPointCounter = 0
        for rowNum in range(0,calcScaledImageDimensions(scalingFactor,originalImage),scalingFactor+1):
            # if rowNum % 2 == 0:
            for colNum in range(0,calcScaledImageDimensions(scalingFactor,originalImage),scalingFactor+1):
                # if colNum % 2 == 0:
                data_x_y_coords_in_ScaledImage[dataPointCounter][0] = rowNum
                data_x_y_coords_in_ScaledImage[dataPointCounter][1] = colNum
                dataPointCounter += 1
        return data_x_y_coords_in_ScaledImage

    def genScaledImageCoordGrids(originalImage,scalingFactor):
        return np.mgrid[0:calcScaledImageDimensions(scalingFactor, originalImage), 0:calcScaledImageDimensions(scalingFactor, originalImage)]

    # Use this print statement to view intermediate results
    # print("Interp Num:",scalingFactor,"  Interp Image Size:",calcScaledImageDimensions(scalingFactor,image),"x",calcScaledImageDimensions(scalingFactor,image))
    data_x_y_coords_in_ScaledImage = genDataCoordsInScaledImage (originalImage = image, scalingFactor = scalingFactor)
    grid_x, grid_y                 = genScaledImageCoordGrids   (originalImage = image, scalingFactor = scalingFactor)
    ogImageValues = image.ravel()
    # giddata() is the function from scipy.interpolate that powers imageResolutionScaler()
    scaledImage = griddata(data_x_y_coords_in_ScaledImage, ogImageValues, (grid_x, grid_y), method=interpolationMethod)
    return scaledImage

def highPassFilter(image,filterThreshold):
    # sets pixels that are below the filterThreshold to zero.
    # recommended parameters: filterThreshold = 80
    imageToFilter = np.copy(image)
    for rowIx,pixelRow in enumerate(imageToFilter):
        for colIx,pixelValue in enumerate(pixelRow):
            if pixelValue < filterThreshold:
                imageToFilter[rowIx,colIx] = 0 # Apply a high-pass filter: set pixel values that are below filterThreshold to 0 to help filter noise
    return imageToFilter

def normalizeImage(image):
    # my own normalizing transformer
    normalizedImage = np.zeros(image.shape)
    max = image.max()
    normalizedImage = image/max
    return normalizedImage

def testSetAccuracy(yhat,yActual):
    # just a simple function to test basic accuracy when rapidly prototyping code
    if len(yhat) == len(yActual):
        comp = yhat==yActual.ravel()
        if comp.all():
            print("100% Accuracy!")
        else:
            correctCounter = 0
            for index in range(len(yhat)):
                if yhat[index]==yActual[index]:
                    correctCounter+=1
            print("Accuracy: ",(correctCounter/len(yActual))*100,"%",sep='')
    else:
        print("Error: Arguments are not the same dimensions!")

## IMPORT/EXPORT DATA FUNCTIONS ####################################################################################
def exportImageData(X, x_csvFileName, y=None, y_csvFileName=None):
    # Can use this method to export pre-processed image data if pre-processing takes too long
    # so it can be loaded during next session rather than re-pre-processing from scratch
    #Computation Time for exporting 2,000,000 images: 2min, 6sec MacBook Pro. Same as desktop PC
    #create numpy array size (8*numberOfImages,8) if X contains 8x8 images
    exportX = np.zeros((X.shape[1]*X.shape[0]+X.shape[0],X.shape[2]))
    #dump image data into continous 8x8 matrix, each image seperated by row of -1
    rowCounter = 0
    for image in X:
        for pixelRow in image:
            exportX[rowCounter] = pixelRow
            rowCounter += 1
        exportX[rowCounter] = np.repeat(-1,image.shape[1])
        rowCounter += 1
    Xdf = pd.DataFrame(data=exportX)
    #### X READY FOR EXPORT TO CSV FILE ####
    Xdf.to_csv(x_csvFileName,index=False)
    ######################################
    if y != None:
        ydf = pd.DataFrame(data=y)
        #### y READY FOR EXPORT TO CSV FILE ####
        ydf.to_csv(y_csvFileName,index=False)
    ######################################
    # For optional use:
    # #Test X import:
    # compare = X_imp == X
    # compare.all()
    # #Test y import:
    # compare = y_imp == y
    # compare.all()

def importImageData(x_csvFileName,y_csvFileName=None):
    # Used for importing thermal image data stored in CSV files
    #Computation Time for importing 2,000,000 images: 15sec, MacBook Pro. Same as desktop PC
    #Import X:
    importedXarray = pd.read_csv(x_csvFileName).to_numpy()
    #Count how many images are in the imported dataset:
    numberOfImages = 0
    imageHeightDetermined = False
    imageHeight = 0
    for row in importedXarray:
        if row[0] == -1:
            imageHeightDetermined = True
            numberOfImages += 1
        if not imageHeightDetermined:
            imageHeight += 1

    X = np.zeros((numberOfImages,imageHeight,importedXarray.shape[1]))
    for i in range(X.shape[0]):
        startSlice = (imageHeight*i)+i
        endSlice   = startSlice+imageHeight
        X[i] = importedXarray[startSlice:endSlice]
    #Import y:
    if y_csvFileName != None:
        y = pd.read_csv(y_csvFileName).to_numpy()
        return X,y
    return X
    # For optional use:
    # #Test X import:
    # compare = X_imp == X
    # compare.all()
    # #Test y import:
    # compare = y_imp == y
    # compare.all()

def visualizeXyData(imageIndex=0):
    # simple function to display both the thermal image (X) and the corresponding label (y)
    showThermalImage(X[imageIndex])
    print("Human") if y[imageIndex] == 1 else print("Air Vent")

## PRE-PROCESSING PIPELINES ############################################################################################################
## Should re-factor this code to make use of SKLearn pipelines and transformers to make it more readable by other people in the industry
########################################################################################################################################
def imageResolutionScalerPipline(images,scalingFactor,interpolationMethod):
    # utilizes the above functions involved in generating upscaled images and returns the set of upscaled images
    # images are returned as a numpy array of new dimensions corresponding to the new upscaled dimensions
    scaledImages = np.zeros((images.shape[0],calcScaledImageDimensions(scalingFactor, images[0]),calcScaledImageDimensions(scalingFactor, images[0])))
    for index,image in enumerate(images):
        scaledImages[index] = imageResolutionScaler(image,scalingFactor,interpolationMethod)
    return scaledImages

def highPassFilterPipeline(images,filterThreshold):
    # utilizes the highPassFilter() function above and applies the HPF to a set of input images
    # pipe returns set of filtered images as numpy array with same dimensions as 'images'
    filteredImages = np.zeros(images.shape)
    for index,image in enumerate(images):
        filteredImages[index] = highPassFilter(image,filterThreshold)
    return filteredImages

def normalizerPipeline(images):
    # returns numpy array of same dimensions of 'images' where images are normalized using the normalizeImage() function
    normalizedImages = np.zeros(images.shape)
    for index,image in enumerate(images):
        normalizedImages[index] = normalizeImage(image)
    return normalizedImages

def runAllPreProcessingPipline(X,scalingFactor,interpolationMethod):
    # runs all pre-processing pipes in the below order:
    higherResImages = imageResolutionScalerPipline(images = X,scalingFactor = 3,interpolationMethod = 'linear')
    filteredImages  = highPassFilterPipeline(higherResImages,80)
    X_pre_processed = normalizerPipeline(filteredImages)
    return X_pre_processed

## SCRIPT ##
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# IMPORT CSV DATA + PRE-PROCESS IMAGES
x_csvFileName = "thermalSensorImages.csv"
y_csvFileName = "thermalSensorLabels.csv"

# #IMPORT IMAGE DATASETS
X,y = importImageData(x_csvFileName, y_csvFileName)
X.shape
y.shape

# #Visualize first X image + first label:
visualizeXyData(imageIndex=0)

# PRE-PROCESS IMAGES
# Pipline order for pr-processing
# Pipe 1: Res Scaler
# Pipe 2: Filter
# Pipe 3: Normalizer

X_pre_processed = runAllPreProcessingPipline(X,scalingFactor=3,interpolationMethod='linear')
#Computation time: 26 seconds for 18,000 images, MacBook Pro i7

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## MODELS ##

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# CNN LeNet-5 Model

#Computation Times
#Macbook Pro:  3000 images      :  4.5s
#Macbook Pro:  15,000 images    : 29.8s

##############################
# DATASET SIZE: 18,000 images
##############################

sizeOfTrainSet = 1000
sizeOfTestSet  = 200
Xtrain = X_pre_processed[0:sizeOfTrainSet]
ytrain = y[0:sizeOfTrainSet]
Xtest  = X_pre_processed[sizeOfTrainSet:sizeOfTrainSet+sizeOfTestSet]
ytest  = y[sizeOfTrainSet:sizeOfTrainSet+sizeOfTestSet]

Xtrain.shape
ytrain.shape
Xtest.shape
ytest.shape

Xtrain_unrolled = Xtrain.reshape(Xtrain.shape[0], Xtrain.shape[1], Xtrain.shape[2], 1).astype('float32')
Xtest_unrolled = Xtest.reshape(Xtest.shape[0], Xtest.shape[1], Xtest.shape[2], 1).astype('float32')
input_shape=(Xtest.shape[1], Xtest.shape[2], 1)
print("input_shape:",input_shape)

ytrain_encoded = tensorflow.keras.utils.to_categorical(ytrain)
ytest_encoded  = tensorflow.keras.utils.to_categorical(ytest)
print("# of Classes:",ytest_encoded.shape[1])

def leNet_5(input_shape):
    print("Input Layer Dimensions:",input_shape)
    #Implementing LeNet-5 CNN:
    model = tensorflow.keras.Sequential()
    model.add(layers.Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.AveragePooling2D())
    model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
    model.add(layers.AveragePooling2D(pool_size=(2,2),padding='same'))
    model.add(layers.Flatten())
    model.add(layers.Dense(units=120, activation='relu'))
    model.add(layers.Dense(units=84, activation='relu'))
    model.add(layers.Dense(units=2, activation = 'softmax'))

    #Compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

leNet_5_SF3 = leNet_5(input_shape)

leNet_5_SF3.fit(Xtrain_unrolled, ytrain_encoded, validation_data=(Xtest_unrolled,ytest_encoded),epochs=10,batch_size=100,verbose=2)

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Test .predict() on new un-seen data (images between 3000 - 3500, ie. 500 unseen images)
X_testSet = X_pre_processed [sizeOfTrainSet+sizeOfTestSet : sizeOfTrainSet+sizeOfTestSet+sizeOfTestSet]
y_testSet = y               [sizeOfTrainSet+sizeOfTestSet : sizeOfTrainSet+sizeOfTestSet+sizeOfTestSet]

X_testSet_pre_processed = runAllPreProcessingPipline(X_testSet,scalingFactor=3,interpolationMethod='linear')

Xtest_preProcessed_unrolled = X_testSet_pre_processed.reshape(X_testSet_pre_processed.shape[0], X_testSet_pre_processed.shape[1], X_testSet_pre_processed.shape[2], 1).astype('float32')

yhat_raw = leNet_5_SF3.predict(Xtest_preProcessed_unrolled)

def convertYHATtoArray(yhat_raw):
    yhat = np.zeros(yhat_raw.shape[0])
    for index,yhat_raw_cat_predictions in enumerate(yhat_raw):
        if yhat_raw_cat_predictions[1] >= yhat_raw_cat_predictions[0]:
            #ID'd a human:
            yhat[index] = 1
        else:
            yhat[index] = 0
    return yhat

testSetAccuracy(convertYHATtoArray(yhat_raw),y_testSet)

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#UNPROCESSED IMAGE EXPERIMENTS:
# Train on 1000 samples, validate on 200 samples, epoch = 10
# Epoch 1/10
# 1000/1000 - 0s - loss: 1.7278 - accuracy: 0.4950 - val_loss: 0.7863 - val_accuracy: 0.5000
# Epoch 2/10
# 1000/1000 - 0s - loss: 0.7844 - accuracy: 0.5140 - val_loss: 0.6933 - val_accuracy: 0.5350
# Epoch 3/10
# 1000/1000 - 0s - loss: 0.6713 - accuracy: 0.5990 - val_loss: 0.6828 - val_accuracy: 0.6050
# Epoch 4/10
# 1000/1000 - 0s - loss: 0.6462 - accuracy: 0.6190 - val_loss: 0.6579 - val_accuracy: 0.5600
# Epoch 5/10
# 1000/1000 - 0s - loss: 0.6205 - accuracy: 0.6300 - val_loss: 0.6474 - val_accuracy: 0.5500
# Epoch 6/10
# 1000/1000 - 0s - loss: 0.6011 - accuracy: 0.6280 - val_loss: 0.6190 - val_accuracy: 0.6350
# Epoch 7/10
# 1000/1000 - 0s - loss: 0.5709 - accuracy: 0.7150 - val_loss: 0.5822 - val_accuracy: 0.6450
# Epoch 8/10
# 1000/1000 - 0s - loss: 0.5342 - accuracy: 0.7400 - val_loss: 0.5516 - val_accuracy: 0.7550
# Epoch 9/10
# 1000/1000 - 0s - loss: 0.5087 - accuracy: 0.7890 - val_loss: 0.5220 - val_accuracy: 0.7650
# Epoch 10/10
# 1000/1000 - 0s - loss: 0.4965 - accuracy: 0.7440 - val_loss: 0.5345 - val_accuracy: 0.6800

#PRE-PROCESSED IMAGE EXPERIMENTS:
# Train on 1000 samples, validate on 200 samples. epoch = 10
# Epoch 1/10
# 1000/1000 - 0s - loss: 0.5365 - accuracy: 0.6700 - val_loss: 0.3240 - val_accuracy: 1.0000
# Epoch 2/10
# 1000/1000 - 0s - loss: 0.1445 - accuracy: 1.0000 - val_loss: 0.0243 - val_accuracy: 1.0000
# Epoch 3/10
# 1000/1000 - 0s - loss: 0.0059 - accuracy: 1.0000 - val_loss: 0.0011 - val_accuracy: 1.0000
# Epoch 4/10
# 1000/1000 - 0s - loss: 5.1237e-04 - accuracy: 1.0000 - val_loss: 2.2952e-04 - val_accuracy: 1.0000
# Epoch 5/10
# 1000/1000 - 0s - loss: 1.7337e-04 - accuracy: 1.0000 - val_loss: 1.6253e-04 - val_accuracy: 1.0000
# Epoch 6/10
# 1000/1000 - 0s - loss: 1.0130e-04 - accuracy: 1.0000 - val_loss: 6.8378e-05 - val_accuracy: 1.0000
# Epoch 7/10
# 1000/1000 - 0s - loss: 6.8886e-05 - accuracy: 1.0000 - val_loss: 6.5127e-05 - val_accuracy: 1.0000
# Epoch 8/10
# 1000/1000 - 0s - loss: 5.5768e-05 - accuracy: 1.0000 - val_loss: 6.2917e-05 - val_accuracy: 1.0000
# Epoch 9/10
# 1000/1000 - 0s - loss: 5.0803e-05 - accuracy: 1.0000 - val_loss: 6.3479e-05 - val_accuracy: 1.0000
# Epoch 10/10
# 1000/1000 - 0s - loss: 4.7964e-05 - accuracy: 1.0000 - val_loss: 4.8577e-05 - val_accuracy: 1.0000

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------













#UNUSED CODE:

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# TESTING OTHER MODEL TYPES (SKLEARN MODELS + DNN)
# preparing data for SKLearn models first:

sizeOfTrainSet = 1000
sizeOfTestSet  = 200
Xtrain3 = X[0:sizeOfTrainSet]
ytrain3 = y[0:sizeOfTrainSet]
Xtest3  = X[sizeOfTrainSet:sizeOfTrainSet+sizeOfTestSet]
ytest3  = y[sizeOfTrainSet:sizeOfTrainSet+sizeOfTestSet]

#Convert Xtrain3 into XtrainLR dataset that is compatible for LR training (rows = 1000, columns = 29*29)
# XtrainLR.shape
# Xtrain3[0][14][14]
# Xtrain3[0].ravel()[(29*(15-1)+15)-1]

XtrainLR = np.zeros((Xtrain3.shape[0],Xtrain3.shape[1]*Xtrain3.shape[2]))
for index in range(XtrainLR.shape[0]):
    XtrainLR[index] = Xtrain3[0].ravel()
XtestLR = np.zeros((Xtest3.shape[0],Xtest3.shape[1]*Xtest3.shape[2]))
for index in range(XtestLR.shape[0]):
    XtestLR[index] = Xtest3[0].ravel()

ytrainLR = ytrain3.reshape(-1,1).ravel()
ytestLR = ytest3.reshape(-1,1).ravel()
#SANITY CHECK:
XtrainLR.shape
ytrainLR.shape
# Xtrain3[0][14][14] == XtrainLR[0][(29*(15-1)+15)-1]

#Data ready. Define perfomance computation function:
def compute_performance_Array(yhat, y):
    correctCounter = 0
    for index in range(y.shape[0]):
        if y[index] == yhat[index]:
            correctCounter += 1
    acc = (correctCounter / y.shape[0])
    return (round(acc*100,2))

#SKLearn Models to train/evaluate:
#LOGREG:
from sklearn.linear_model import LogisticRegression
lm = LogisticRegression()
lm.fit(XtrainLR,ytrainLR)
yhat = lm.predict(XtrainLR)
compute_performance_Array(yhat,ytrainLR)

#RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state = 1, n_estimators=10)
rf.fit(XtrainLR,ytrainLR)
compute_performance_Array(rf.predict(XtrainLR),ytestLR)
#XGBOOST
from xgboost import XGBClassifier
xgb = XGBClassifier(random_state = 1)
xgb.fit(XtrainLR,ytrainLR)
compute_performance_Array(xgb.predict(XtrainLR),ytrainLR)

# Testing my SKLearn models on MNIST as a sanity check:
from sklearn.datasets import load_digits
digits = load_digits()
# digits.data[0]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=0)
lm = LogisticRegression()
lm.fit(x_train,y_train)
compute_performance_Array(lm.predict(x_train),y_train)

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# traditional Deep Neural Network (Non-convolutional):
def dNN(input_shape):
    model = Sequential()
    #Input Layer:
    model.add(Input(shape=(input_shape[0]**2,)))
    #Hidden Layer: (5 hidden layers)
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    #Output Layer:
    model.add(Dense(1, activation='sigmoid'))
    #Compile network:
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

dNN = shallow_NeuralNetwork((8,8,1))

sizeOfTrainSet = 1000
sizeOfTestSet  = 200
Xtrain_sNN = X[0:sizeOfTrainSet]
ytrain_sNN = y[0:sizeOfTrainSet]
Xtest_sNN  = X[sizeOfTrainSet:sizeOfTrainSet+sizeOfTestSet]
ytest_sNN  = y[sizeOfTrainSet:sizeOfTrainSet+sizeOfTestSet]

#X unrolling:
Xtrain_sNN_unrolled = np.zeros((Xtrain_sNN.shape[0],Xtrain_sNN.shape[1]*Xtrain_sNN.shape[2]))
for index in range(Xtrain_sNN_unrolled.shape[0]):
    Xtrain_sNN_unrolled[index] = Xtrain_sNN[0].ravel()
Xtest_sNN_unrolled = np.zeros((Xtest_sNN.shape[0],Xtest_sNN.shape[1]*Xtest_sNN.shape[2]))
for index in range(Xtest_sNN_unrolled.shape[0]):
    Xtest_sNN_unrolled[index] = Xtest_sNN[0].ravel()

#y reshaping:
ytrain_sNN_reshaped = ytrain_sNN.reshape(-1,1).ravel()
ytest_sNN_reshaped = ytest_sNN.reshape(-1,1).ravel()
#CHECK:
Xtrain_sNN_unrolled.shape
ytrain_sNN_reshaped.shape


dNN.fit(Xtrain_sNN_unrolled, ytrain_sNN_reshaped, validation_data=(Xtest_sNN_unrolled,ytest_sNN_reshaped),epochs=10,batch_size=100,verbose=2)

dNN.evaluate(Xtrain_sNN_unrolled,ytrain_sNN_reshaped)


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------





































#End-of-file
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
