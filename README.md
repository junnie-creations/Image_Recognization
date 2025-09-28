# Image_Recognization
The Fruit vs Flower Image Classification project uses TensorFlow/Keras and CNNs to classify images into fruits or flowers. Images are preprocessed, labeled, and trained on a deep learning model with convolution, pooling, and dense layers. The trained model is saved for future predictions.

# 🍒 Fruit-Flower Image Classification CNN 🌸
---
This project implements a Convolutional Neural Network (CNN) using TensorFlow/Keras and OpenCV (cv2) to perform binary classification on image data (Fruits vs. Flowers). It is an excellent, self-contained project for building a profile in AI/ML and deep learning.

# 📖 Project Report
---
1. Introduction
This project focuses on building and training a simple yet effective CNN for image classification. The goal is to accurately distinguish between two classes of images: "Fruit" and "Flower". The script handles all necessary steps, from recursively loading images from disk and preprocessing them, to defining, compiling, and fitting the deep learning model.

# 2. Project Goals
---
✅ Implement efficient data loading and preprocessing using os and cv2.

✅ Define a robust CNN architecture suitable for small image datasets.

✅ Train the classification model using labeled data.

✅ Serialize and save the trained model (.h5 format) for future inference.

✅ Demonstrate fundamental deep learning practices for computer vision.

3. Technology Stack
```
Core Language: Python 3.x

Deep Learning Framework: TensorFlow / Keras

Image Processing: OpenCV (cv2)

Numerical Computing: NumPy

OS Interaction: os

```
---

# 4. \text{Features & Functionality}
🖼️ Image Preprocessing
Data Loading: Recursively loads all images from the specified fruit and flower directories.
Resizing: Standardizes all input images to a fixed size of (150,150) pixels.
Normalization: Converts image data to NumPy arrays of type float32 (though explicit normalization to [0,1] is recommended after this step, which you can add as a refinement).

---

#🏗️ CNN Architecture
Layers: Uses three Conv2D layers followed by MaxPooling2D layers to progressively extract features.
Regularization: Includes a Dropout layer (rate 0.5) before the output layer to mitigate overfitting.
Output: A final Dense layer with a sigmoid activation function for binary classification.

⚙️ Model Training
Optimizer: Adam optimizer with a learning rate of 0.001.
Loss Function: Binary Crossentropy is used, appropriate for a two-class problem.
Training: The model is trained for 15 epochs with a batch_size of 32.

5. Code Overview
🧠 CNN Architecture Definition
The core of the model uses a standard Conv-Pool pattern:

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
    Dense(1,activation="sigmoid") # Binary classification output
])

💾 Key File Operations
The script loads data and saves the final model to specific paths:

# Data Loading Paths (MUST BE UPDATED BY USER)
fruit_dir = r"D:\coderun\AIML\datasets\fruits"
flower_dir = r"D:\coderun\AIML\datasets\flowers"

# Model Saving Path (MUST BE UPDATED BY USER)
model.save(r"D:\coderun\AIML\model\classification_model.h5")

🛠️ Getting Started
1. Clone Repository
git clone <repository-url>
cd fruit-flower-classifier

2. Install Dependencies
This project requires TensorFlow/Keras, OpenCV, and NumPy.

pip install tensorflow keras opencv-python numpy

3. Setup Data Directories
Create two directories containing your images, for example:

datasets/
├── fruits/
│   └── apple.jpg
│   └── banana.jpg
├── flowers/
│   └── rose.jpg
│   └── tulip.jpg

CRITICAL: You must update the fruit_dir and flower_dir variables in the Python script to match the absolute paths on your system.

4. Run Training Script
Execute the Python script to start the training process. The model will be saved once training is complete.

python your_script_name.py

📂 Project Structure
fruit-flower-classifier/
│── your_script_name.py     # Main CNN definition and training script
│── datasets/                # Image data folder (must be created)
│   ├── fruits/              # Contains fruit images (Label 0)
│   └── flowers/             # Contains flower images (Label 1)
│── model/                   # Output directory for saved model
│   └── classification_model.h5
│── README.md                # Project documentation

🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

Fork the Project

Create your Feature Branch (git checkout -b feature/AmazingFeature)

Commit your Changes (git commit -m 'Add some AmazingFeature')

Push to the Branch (git push origin feature/AmazingFeature)

Open a Pull Request

📜 License
This project is licensed under the MIT License.
