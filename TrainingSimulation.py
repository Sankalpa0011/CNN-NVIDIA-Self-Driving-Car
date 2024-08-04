from utils import *
from sklearn.model_selection import train_test_split

#### Step 1 - Import data
path = "data"
data = importDataInfo(path)


#### Step 2 - Visualization and Distribution of Data (Balance Data)
data = balanceData(data, display=False)


#### Step 3 - Image and Steering Data Preprocessing
imagesPath, steerings = loadData(path, data)
#print(imagesPath[0])
#print(steering[0])


#### Step 4 - Split The Data Tarin Validation
x_train, x_test, y_train, y_test = train_test_split(imagesPath, steerings, test_size=0.2, random_state=42)
#print(x_train.shape, x_test.shape)
#print(y_train.shape, y_test.shape)


#### Step 5 - Data Augmentation


#### Step 6 - Preprocessing of the image


#### Step 7 - Batch Generator To Send Images To Train The Model


#### Step 8 - Create NVIDIA Model
model = createModel()
model.summary()


### Step 9 - Train The Model
history = model.fit(batchGen(x_train, y_train,100, 1), steps_per_epoch=300, epochs=10,
                    validation_data= batchGen(x_test, y_test, 100, 0), validation_steps=200)


#### Step 10 - Saving And Plotting The Model
model.save("self_drive_model.h5")
print("Model Saved")
model.save("self_drive_model.keras")
print("Model Saved")


plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.legend(["Training", "validation"])
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()