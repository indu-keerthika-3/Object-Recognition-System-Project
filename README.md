📌 Object Recognition System

📖 Overview

This project implements an object recognition system using TensorFlow and OpenCV. It includes dataset handling, image preprocessing, and a pre-trained MobileNetV2 model for object classification.

🚀 Features

Downloads a sample dataset (cats and dogs) from Unsplash.

Supports dataset download from Google Drive using gdown.

Utilizes a pre-trained MobileNetV2 model for object recognition.

Displays predictions along with the processed image.

🔧 Installation

Before running the notebook, install the required dependencies:

pip install tensorflow opencv-python matplotlib numpy gdown requests

🛠️ Usage

Run the dataset setup cell to download images from Unsplash.

(Optional) Download a dataset from Google Drive.

Run the object recognition model on a sample image.

📂 Running the Notebook

Open Merged_Object_Recognition.ipynb in Jupyter Notebook or Google Colab and execute the cells sequentially.

🎯 Example Output

The model will predict the top 3 objects in an image along with their confidence scores:

1: Tabby Cat (0.92)
2: Tiger Cat (0.05)
3: Egyptian Cat (0.02)

📜 License

This project is intended for educational purposes only.
