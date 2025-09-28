import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

#dataset path
fruit_dir = r"D:\coderun\AIML\datasets\fruits"
flower_dir = r"D:\coderun\AIML\datasets\flowers"

img_size = (150,150)
x=[] # x -> import images in float32 -> a,b,c
y=[] #y -> importing labels -> 0,1

# loading image
for filename in os.listdir(fruit_dir):
    path = os.path.join(fruit_dir,filename)
    img = cv2.imread(path)
    if img is not None:
        img = cv2.resize(img,img_size)
        x.append(img)
        y.append(0)

for filename in os.listdir(flower_dir):
    path = os.path.join(flower_dir,filename)
    img = cv2.imread(path)
    if img is not None:
        img = cv2.resize(img,img_size)
        x.append(img)
        y.append(1)

#convert to numpy arrays
x = np.array(x,dtype="float32") # error
y = np.array(y)

print("Dataset loaded", x.shape,y.shape)

#model training
#a)  building the model
model = Sequential([
        Conv2D(32,(3,3), activation = "relu", input_shape=(150,150,3)),
        MaxPooling2D(2,2),

        Conv2D(64,(3,3)),
        MaxPooling2D(2,2),

        Conv2D(128,(3,3)),
        MaxPooling2D(2,2),

        Flatten(),
        Dense(128, activation = "relu"),
        Dropout(0.5),
        Dense(1,activation="sigmoid") # binary outputs
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',  #error
              metrics=['accuracy'])

# 100%
history = model.fit(x,y,epochs=15,batch_size=32,shuffle=True)

#saving the model
model.save(r"D:\coderun\AIML\model\classification_model.h5")