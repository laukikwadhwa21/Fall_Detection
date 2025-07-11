#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>

// CORRECT library for the Arduino Nano 33 BLE Sense Rev2
#include <Arduino_BMI270_BMM150.h>

// The header file containing our TFLite model data
#include "fall_detection_model.h"

// --- Globals for TensorFlow Lite ---
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Arena size may need adjustment based on model complexity. 8KB is a good start.
constexpr int kTensorArenaSize = 8 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

// Variables to hold the quantization parameters of the model.
float    input_scale = 0.0f;
int8_t   input_zero_point = 0;
float    output_scale = 0.0f;
int8_t   output_zero_point = 0;
} // namespace

// --- Application Constants ---
const int NUM_FEATURES = 3;  // ax, ay, az
const int TIME_STEPS = 150; // Must match the training script
const int SAMPLES_TO_DISCARD = 75; // Sliding window step, must match training
const int SAMPLES_TO_KEEP = TIME_STEPS - SAMPLES_TO_DISCARD;
const float G_TO_MS2 = 9.81; // Gravitational constant for unit conversion

// FIX: Control the sampling rate to ~50Hz (1000ms / 20ms = 50Hz)
const int SAMPLING_DELAY_MS = 20;
unsigned long last_sample_time = 0;

int sample_count = 0; // Counter for samples in the current window

// --- SCALER VALUES (FROM YOUR PYTHON SCRIPT) ---
float SCALER_MEAN[] = {-0.22145857198715083, 7.362072807828835, 1.299067169081437};
float SCALER_SCALE[] = {4.034813242379331, 5.193313431588992, 4.143631195725836};


// ======================================================================================
//                             SETUP
// ======================================================================================
void setup() {
  Serial.begin(9600);
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, LOW);

  // --- Initialize the TFLite Error Reporter ---
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;
  error_reporter->Report("TFLite setup starting.");

  // --- Initialize the IMU (using the correct BMI270 library) ---
  if (!IMU.begin()) {
    error_reporter->Report("Failed to initialize IMU!");
    // Blink LED to indicate fatal error
    while (1) {
      digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN));
      delay(500);
    }
  }
  error_reporter->Report("IMU initialized successfully (BMI270).");

  // --- Load the TFLite Model ---
  model = tflite::GetModel(fall_detection_model_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("Model schema version mismatch!");
    return;
  }

  // --- Create the Op Resolver ---
  static tflite::AllOpsResolver resolver;

  // --- Instantiate the Interpreter ---
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  // --- Allocate Tensors ---
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed.");
    return;
  }
  error_reporter->Report("Tensor arena allocated.");

  // --- Get Pointers to Input and Output TFLiteTensors ---
  input = interpreter->input(0);
  output = interpreter->output(0);

  // --- Get Quantization Parameters ---
  input_scale = input->params.scale;
  input_zero_point = input->params.zero_point;
  output_scale = output->params.scale;
  output_zero_point = output->params.zero_point;

  error_reporter->Report("Setup complete. Starting data collection...");
}


// ======================================================================================
//                             LOOP
// ======================================================================================
void loop() {
  // Enforce a consistent sampling rate
  if (millis() - last_sample_time < SAMPLING_DELAY_MS) {
    return;
  }
  last_sample_time = millis();

  float aX, aY, aZ;

  if (IMU.accelerationAvailable()) {
    IMU.readAcceleration(aX, aY, aZ);

    float ax_ms2 = aX * G_TO_MS2;
    // FIX: Invert the Y-axis to match the training data's orientation.
    float ay_ms2 = -aY * G_TO_MS2;
    float az_ms2 = aZ * G_TO_MS2;

    float scaled_ax = (ax_ms2 - SCALER_MEAN[0]) / SCALER_SCALE[0];
    float scaled_ay = (ay_ms2 - SCALER_MEAN[1]) / SCALER_SCALE[1];
    float scaled_az = (az_ms2 - SCALER_MEAN[2]) / SCALER_SCALE[2];

    int8_t quant_ax = (int8_t)(scaled_ax / input_scale + input_zero_point);
    int8_t quant_ay = (int8_t)(scaled_ay / input_scale + input_zero_point);
    int8_t quant_az = (int8_t)(scaled_az / input_scale + input_zero_point);

    // Fill the input buffer
    if (sample_count < TIME_STEPS) {
      input->data.int8[sample_count * NUM_FEATURES + 0] = quant_ax;
      input->data.int8[sample_count * NUM_FEATURES + 1] = quant_ay;
      input->data.int8[sample_count * NUM_FEATURES + 2] = quant_az;
      sample_count++;
    }

    // When buffer is full, run inference
    if (sample_count == TIME_STEPS) {
      if (interpreter->Invoke() != kTfLiteOk) {
        error_reporter->Report("Invoke failed!");
        return;
      }
      
      int8_t raw_adl_score = output->data.int8[0];
      int8_t raw_fall_score = output->data.int8[1];
      float adl_score_float = (float)(raw_adl_score - output_zero_point) * output_scale;
      float fall_score_float = (float)(raw_fall_score - output_zero_point) * output_scale;
      
      Serial.println("--- NEW INFERENCE ---");
      Serial.print("Processed -> X: ");
      Serial.print(ax_ms2, 2);
      Serial.print(" Y: ");
      Serial.print(ay_ms2, 2); // This should now be positive when still
      Serial.print(" Z: ");
      Serial.println(az_ms2, 2);
      
      Serial.print("ADL Score: ");
      Serial.print(adl_score_float, 4);
      Serial.print(" | Fall Score: ");
      Serial.println(fall_score_float, 4);

      if (isnan(adl_score_float) || isnan(fall_score_float)) {
          error_reporter->Report("Error: Score is NaN. Check calculations.");
      } else {
          if (fall_score_float > 0.75) {
              digitalWrite(LED_BUILTIN, HIGH);
              error_reporter->Report("****** FALL DETECTED! ******");
          } else {
              digitalWrite(LED_BUILTIN, LOW);
          }
      }
      
      Serial.println(); // Add a blank line for readability

      memmove(input->data.int8, &input->data.int8[SAMPLES_TO_DISCARD * NUM_FEATURES], SAMPLES_TO_KEEP * NUM_FEATURES);
      sample_count = SAMPLES_TO_KEEP;
    }
  }
}