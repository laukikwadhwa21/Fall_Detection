🤸‍♂️ Elderly Fall Detection using TinyML on Arduino
This project implements an end-to-end machine learning system for detecting human falls in real-time. A 1D Convolutional Neural Network (CNN) is trained on accelerometer data and deployed to an Arduino Nano 33 BLE Sense using the TensorFlow Lite for Microcontrollers framework.

✨ Features
Real-Time Inference: Classifies accelerometer data on-device to provide immediate fall detection.

High Accuracy: The model is trained to be highly sensitive to falls, achieving 99% recall on test data.

Efficient On-Device Performance: Uses an INT8 quantized TensorFlow Lite model, making it small, fast, and suitable for resource-constrained microcontrollers.

End-to-End Workflow: Covers the complete process from data preprocessing and model training to hardware deployment and real-time application.

⚙️ Workflow
The project is broken down into two main stages:

Model Training & Conversion (Python):

The MobiFall dataset, containing accelerometer readings for various activities of daily living (ADLs) and simulated falls, is processed.

The raw data is segmented into windows to create time-series samples.

A 1D CNN model is built and trained in TensorFlow/Keras to classify these samples.

The final trained model is converted to a TensorFlow Lite format and quantized to 8-bit integers to drastically reduce its size.

The quantized model is exported as a C header file (model_data.h).

Deployment (Arduino/C++):

An Arduino sketch is written to run on the Nano 33 BLE Sense.

It continuously reads data from the onboard accelerometer.

It applies the exact same preprocessing (scaling and quantization) as the training script.

It uses a sliding window to feed data to the TFLite Micro interpreter.

On detecting a fall, it triggers an alert (e.g., lights up the onboard LED and prints to the Serial Monitor).

🛠️ Technology Stack
Machine Learning: Python, TensorFlow, Keras, Scikit-learn, Pandas

Embedded System: C++, Arduino Framework, TensorFlow Lite for Microcontrollers

Hardware: Arduino Nano 33 BLE Sense Rev 2

📂 Repository Structure
.
├── Code/
│   ├── Model_Training.ipynb    
│   └── Fall_Detection_Device.ino    
└── README.md 

## Contact
For any questions or feedback, please contact:
- Laukik Wadhwa: laukikwadhwa21@gmail.com
