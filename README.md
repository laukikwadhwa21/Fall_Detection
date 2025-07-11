# 🤸‍♂️ Elderly Fall Detection using TinyML on Arduino

This project implements an end-to-end machine learning system for detecting human falls in real-time. A 1D Convolutional Neural Network (CNN) is trained on accelerometer data and deployed to an Arduino Nano 33 BLE Sense using the TensorFlow Lite for Microcontrollers framework.

-----

## ✨ Features

  * **Real-Time Inference**: Classifies accelerometer data on-device to provide immediate fall detection.
  * **High Accuracy**: The model is trained to be highly sensitive to falls, achieving **99% recall** on test data.
  * **Efficient On-Device Performance**: Uses an **INT8 quantized** TensorFlow Lite model, making it small, fast, and suitable for resource-constrained microcontrollers.
  * **End-to-End Workflow**: Covers the complete process from data preprocessing and model training to hardware deployment and real-time application.

-----

## ⚙️ Workflow

The project is broken down into two main stages:

#### 1\. Model Training & Conversion (Python)

The `MobiFall` dataset, containing accelerometer readings for various activities of daily living (ADLs) and simulated falls, is processed. A 1D CNN model is built and trained in TensorFlow/Keras to classify these samples. The final trained model is converted to a TensorFlow Lite format, quantized to 8-bit integers, and exported as a C header file (`model_data.h`).

#### 2\. Deployment (Arduino/C++)

An Arduino sketch is written to run on the Nano 33 BLE Sense. It continuously reads data from the onboard accelerometer, applies the same preprocessing as the training script, and uses a sliding window to feed data to the TFLite Micro interpreter. On detecting a fall, it triggers an alert.

-----

## 🛠️ Technology Stack

  * **Machine Learning**: Python, TensorFlow, Keras, Scikit-learn, Pandas
  * **Embedded System**: C++, Arduino Framework, TensorFlow Lite for Microcontrollers
  * **Hardware**: Arduino Nano 33 BLE Sense Rev 2

-----

## 📂 Repository Structure

```
.
├── Code/
│   ├── Model_Training.ipynb
│   └── Fall_Detection_Device.ino
└── README.md
```

-----

## 🚀 Setup and Usage

### Part 1: Training the Model and Generating the Header File

1.  **Open the Notebook**: Open the `Code/Model_Training.ipynb` file in Google Colab or a local Jupyter environment.
2.  **Run the Cells**: Execute the cells sequentially to train the model and generate the `model_data.h` file.
3.  **Download the Model**: Download the generated `model_data.h` file.

### Part 2: Deploying to the Arduino

1.  **Setup Arduino IDE**: Install the Arduino IDE and the **Arduino Mbed OS Nano Boards** package.
2.  **Install Libraries**: From the Library Manager, install the `Arduino_TensorFlowLite` library.
3.  **Prepare the Sketch**:
      * Open the `Code/Fall_Detection_Device.ino` sketch file.
      * Place the `model_data.h` file you downloaded into the same folder as the `.ino` file.
4.  **Upload**: Connect your Arduino Nano 33 BLE Sense, select the correct board and port, and click **Upload**.
5.  **Monitor**: Open the **Serial Monitor** (set to 9600 baud) to see the real-time classification results.

-----

## Contact

For any questions or feedback, please contact:

  * **Laukik Wadhwa**: laukikwadhwa21@gmail.com
