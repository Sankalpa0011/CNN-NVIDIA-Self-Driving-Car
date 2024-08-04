import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.utils import shuffle
import matplotlib.image as mpimg
from imgaug import augmenters as iaa
import cv2
import random

from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense
from keras.optimizers import Adam



# get only image name 
def getName(filePath):
    return filePath.split("\\")[-1]




def importDataInfo(path):
    coloumns =  ["Center", "Left", "Right", "Steering", "Throttle", "Break", "Speed"]
    data = pd.read_csv(os.path.join(path, "driving_log.csv"), names=coloumns)
    #print(data.head())

    #print(data["Center"][0])
    #print(getName(data["Center"][0]))

    # apply getnName function to Center column
    data["Center"] = data["Center"].apply(getName)
    #print(data.head())
    print("Total Images Imported Only For Center: ", data.shape[0])
    return data




def balanceData(data, display=True):
    nBins = 31 # this has to be odd number because want to be 0 at the middle of the numbers
    samplesPerBin = 2500
    hist, bins = np.histogram(data["Steering"], nBins)
    #print(bins)
    #print(hist)

    if display:
        center = (bins[:-1] + bins[1:])*0.5
        #print(center)
        plt.bar(center, hist, width=0.06)
        plt.plot((-1, 1), (samplesPerBin, samplesPerBin))
        plt.show()

    removeIndexList = []
    for j in range(nBins):
        binDataList = []
        for i in range(len(data["Steering"])):
            if data["Steering"][i] >= bins[j] and data["Steering"][i] <= bins[j+1]:
                binDataList.append(i)
        binDataList = shuffle(binDataList)
        binDataList = binDataList[samplesPerBin:]
        removeIndexList.extend(binDataList)
    print("Removed Images: ", len(removeIndexList))
    data.drop(data.index[removeIndexList], inplace=True)
    print("Remaining Images: ", len(data))

    if display:
        hist, _ = np.histogram(data["Steering"], nBins)
        plt.bar(center, hist, width=0.06)
        plt.plot((-1, 1), (samplesPerBin, samplesPerBin))
        plt.show()

    return data




def loadData(path, data):
    imagesPath = []
    steerings = []

    for i in range (len(data)):
        indexedData = data.iloc[i]
        #print(indexedData)
        imagesPath.append(os.path.join(path, "IMG", indexedData[0]))
        #print(os.path.join(path, "IMG", indexedData[0]))
        steerings.append(float(indexedData[3]))
    imagesPath = np.asarray(imagesPath)
    steerings = np.asarray(steerings)
    return imagesPath, steerings




def augmentImages(imgPath, steering):
    img = mpimg.imread(imgPath)

    # Augment happen in randomly
    ## PAN
    if np.random.rand() < 0.5:
        pan = iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)})
        img = pan.augment_image(img)

    ## ZOOM
    if np.random.rand() < 0.5:
        zoom = iaa.Affine(scale=(1, 1.2))
        img = zoom.augment_image(img)

    ## BRIGHTNESS
    if np.random.rand() < 0.5:
        brightness = iaa.Multiply((0.4, 1.2))
        img = brightness.augment_image(img)

    #FLIP
    if np.random.rand() < 0.5:
        img = cv2.flip(img, 1)
        steering = -steering


    return img, steering

#imgRe , st = augmentImages("test.jpg", 0)
#plt.imshow(imgRe)
#plt.show()




def preProcessing(img):
    # Crop the image 
    img = img[60:135, :, :]
    # Convert color to YUV
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    # Blur Image
    img = cv2.GaussianBlur(img, (3, 3), 0)
    # Resize the image (NVIDIA used size)
    img = cv2.resize(img, (200, 66))
    # Normalization
    img = img / 255.0

    return img

#imgRe = preProcessing(mpimg.imread("test.jpg"))
#plt.imshow(imgRe)
#plt.show()




def batchGen(imagesPath,  steeringList, batchSize, trainFlag):
    while True:
        imgBatch = []
        steeringBatch = []

        for i in range(batchSize):
            index = random.randint(0, len(imagesPath)-1)
            if trainFlag:
                img, steering = augmentImages(imagesPath[index], steeringList[index])
            else:
                img = mpimg.imread(imagesPath[index])
                steering = steeringList[index]

            img = preProcessing(img)
            imgBatch.append(img)
            steeringBatch.append(steering)
        yield (np.asarray(imgBatch), np.asarray(steeringBatch))





def createModel():
    model = Sequential()

    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation="elu", input_shape=(66, 200, 3)))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation="elu"))
    model.add(Conv2D(48, (3, 3), strides=(2, 2), activation="elu"))
    model.add(Conv2D(64, (3, 3), activation="elu"))
    model.add(Conv2D(64, (3, 3), activation="elu"))

    model.add(Flatten())  

    model.add(Dense(100, activation="elu"))
    model.add(Dense(50, activation="elu"))
    model.add(Dense(10, activation="elu"))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.0001), loss="mse")

    return model
