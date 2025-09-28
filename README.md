# Image_Recognization
The Fruit vs Flower Image Classification project uses TensorFlow/Keras and CNNs to classify images into fruits or flowers. Images are preprocessed, labeled, and trained on a deep learning model with convolution, pooling, and dense layers. The trained model is saved for future predictions.

# Fruit vs Flower Image Classification

This project is a **binary image classification model** built using **TensorFlow / Keras**.  
It classifies images into **fruits (label = 0)** or **flowers (label = 1)**.

---

## 📂 Dataset
The dataset should be organized into two separate folders:

```text
datasets/
│── fruits/   # contains fruit images
│── flowers/  # contains flower images
```

Update the dataset paths in the code:

```python
fruit_dir = r"D:\coderun\AIML\datasets\fruits"
flower_dir = r"D:\coderun\AIML\datasets\flowers"
```

---

## ⚙️ Requirements
Install the required libraries before running:

```bash
pip install tensorflow opencv-python numpy
```

---

## 🧾 Model Architecture
The CNN architecture used:

- **Conv2D + MaxPooling2D** layers for feature extraction  
- **Flatten + Dense (ReLU)** for learning patterns  
- **Dropout (0.5)** for regularization  
- **Dense (Sigmoid)** output layer for binary classification  

Example model snippet:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(150,150,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3)),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3)),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])
```

---

## 🚀 Training
Run the script to:

1. Load and preprocess images (resized to **150×150**).  
2. Assign labels (**0 for fruits, 1 for flowers**).  
3. Train the CNN model for **15 epochs**.

Training command in script:

```python
history = model.fit(x, y, epochs=15, batch_size=32, shuffle=True)
```

---

## 💾 Saving & Loading the Model
The trained model is saved as:

```text
D:\coderun\AIML\model\classification_model.h5
```

Load it later with:

```python
from tensorflow.keras.models import load_model
model = load_model(r"D:\coderun\AIML\model\classification_model.h5")
```

---

## 📊 Output
During training, you will see:

- Dataset shape (number of images, resized dimensions)  
- Training accuracy & loss per epoch

Example output:

```text
Dataset loaded (500, 150, 150, 3) (500,)
Epoch 1/15 ...
Epoch 15/15 ...
```

---

## ✅ Future Improvements
- Add validation/test split (train/val/test)  
- Use `ImageDataGenerator` or `tf.data` for augmentation & pipeline  
- Support multi-class classification (more categories)  
- Visualize training curves (accuracy / loss)  
- Save model checkpoints and use callbacks (EarlyStopping, ReduceLROnPlateau)

---

## 👨‍💻 Author
Developed as part of an **Image Classification** practice project in Python.
