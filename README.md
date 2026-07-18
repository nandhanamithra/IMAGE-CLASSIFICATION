# IMAGE-CLASSIFICATION
Image classification using MobileNet and Flask API by Mithra Nandhana  

## Problem Statement
Develop a Cat and Dog Image Classification Web Application using Deep Learning (MobileNet) and deploy it using a Flask API.

## Answer
Below,
1. Implementation of the image classification model is done using Python (TensorFlow/Keras). The code is saved in `IMAGE-CLASSIFICATION/` along with the training script `train_model.py` and the saved model `cat_dog_classify.keras`.

The code and the output along with the prediction are given below.
## *Source Code*
![Code](output/program.png)
## *output*
![Predict](output/output.png)
## *Output*
![Output](output/output.png)

2. A Flask web app was also built so the model can be used by uploading images through a browser instead of the command line. The code is saved in the `web/` folder, containing `app.py`, `templates/index.html`. A demo recording of the web app is available at `web/Screen Recording 2026-07-18.mp4`.
## *Web App*
[Watch the demo recording](web/Screen%20Recording%202026-07-18.mp4)

## Final Answer
Given an input image, the model predicts whether it is a Cat 🐱 or a Dog 🐶 along with the confidence percentage.
# What I Learned
By this assignment and class, I learned:
1. Convolutional Neural Networks (CNN) and Transfer Learning
2. Using a pretrained MobileNetV2 model with frozen base layers
3. Image preprocessing — resizing and rescaling for MobileNetV2 input
4. Building a Flask API to accept image uploads and return predictions
5. Building a drag-and-drop web UI to interact with the model

:D
