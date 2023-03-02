## Face Recognition using CNN

### Dataset

this dataset was used from [kaggle](https://www.kaggle.com/datasets/vasukipatel/face-recognition-dataset) with some
modification


Dataset structure

```
+-- face-recognition-dataset
|   +-- Faces
|   +-- Original Images
|   +-- Dataset.csv(labes for images)
+-- test(images taken from internet)
```
Also I used ImageDataGenerator from tensorflow to ease the process of feeding data to CNN

### CNN architecture
```
model = Sequential()
model.add(Conv2D(32, kernel_size = (3, 3), activation='relu', input_shape=(224,224,3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(96, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
#model.add(Dropout(0.3))
model.add(Dense(len(classes),activation='softmax'))
```
#### achieved 98.91% accuracy using this (trained on Nvidia RTX 3070 laptop gpu)

```
Output exceeds the size limit. Open the full output data in a text editor
Epoch 1/30
81/81 [==============================] - 49s 551ms/step - loss: 3.4975 - accuracy: 0.0867
Epoch 2/30
81/81 [==============================] - 46s 570ms/step - loss: 2.7112 - accuracy: 0.2287
Epoch 3/30
81/81 [==============================] - 47s 574ms/step - loss: 2.2892 - accuracy: 0.3240
Epoch 4/30
81/81 [==============================] - 46s 567ms/step - loss: 1.8529 - accuracy: 0.4500
Epoch 5/30
81/81 [==============================] - 46s 564ms/step - loss: 1.5813 - accuracy: 0.5238
Epoch 6/30
81/81 [==============================] - 46s 562ms/step - loss: 1.3103 - accuracy: 0.6019
Epoch 7/30
81/81 [==============================] - 46s 565ms/step - loss: 1.0332 - accuracy: 0.6784
Epoch 8/30
81/81 [==============================] - 45s 551ms/step - loss: 0.7741 - accuracy: 0.7763
Epoch 9/30
81/81 [==============================] - 45s 553ms/step - loss: 0.6064 - accuracy: 0.8345
Epoch 10/30
81/81 [==============================] - 45s 551ms/step - loss: 0.4238 - accuracy: 0.8825
Epoch 11/30
81/81 [==============================] - 45s 552ms/step - loss: 0.3339 - accuracy: 0.9173
Epoch 12/30
81/81 [==============================] - 45s 548ms/step - loss: 0.2982 - accuracy: 0.9254
Epoch 13/30
...
Epoch 29/30
81/81 [==============================] - 46s 563ms/step - loss: 0.0683 - accuracy: 0.9820
Epoch 30/30
81/81 [==============================] - 45s 556ms/step - loss: 0.0421 - accuracy: 0.9891

```

more details in jupyter notebook
