# End-to-End Self-Driving Car Using a CNN (NVIDIA Architecture)

A behavioural-cloning project in which a Convolutional Neural Network predicts a vehicle's steering angle directly from front-camera images. The network is based on the architecture proposed in NVIDIA's paper *End to End Learning for Self-Driving Cars*. It is trained on driving data recorded in the Udacity Self-Driving Car Simulator and then deployed back into the simulator in Autonomous Mode to drive the track without human input.

**Tech stack:** Python, TensorFlow/Keras, OpenCV, imgaug, scikit-learn, Socket.IO, Flask

---

## Table of Contents

- [Overview](#overview)
- [Methodology](#methodology)
- [Model Architecture](#model-architecture)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Training Configuration](#training-configuration)
- [Troubleshooting](#troubleshooting)
- [Limitations and Future Work](#limitations-and-future-work)
- [References](#references)

---

## Overview

The system learns a direct mapping from camera image to steering angle (image in, steering angle out), without any hand-written lane detection or rule-based control.

1. Drive the car manually in the simulator while the center camera image and steering angle are recorded.
2. Train a CNN to regress the steering angle from the image.
3. Run the trained model as a server. The simulator streams camera frames to it over Socket.IO, and the model returns steering and throttle commands in real time.

---

## Methodology

### Training pipeline (`TrainingSimulation.py`)

| Step | Description |
|------|-------------|
| 1. Data import | Reads `driving_log.csv` and extracts the center-camera image names. |
| 2. Data balancing | Recorded steering data is heavily biased towards zero (straight driving). The steering range is divided into 31 bins and each bin is capped at 2,500 samples to prevent the model from favouring straight-line prediction. |
| 3. Data loading | Builds arrays of image paths and corresponding steering values. |
| 4. Train/validation split | 80% training and 20% validation (`random_state=42`). |
| 5. Augmentation | Random pan, zoom, brightness change and horizontal flip, applied to training data only. |
| 6. Preprocessing | Crop, colour-space conversion, blur, resize and normalization. |
| 7. Batch generation | An infinite generator yields random batches: augmented for training, unaltered for validation. |
| 8. Model creation | NVIDIA-style CNN (see Model Architecture). |
| 9. Training | Adam optimizer with mean squared error loss. |
| 10. Saving and evaluation | Saves the model in `.h5` and `.keras` formats and plots training and validation loss. |

### Data augmentation (`utils.py`)

Each augmentation is applied independently with a probability of 0.5:

- **Pan:** translation of up to 10% along both axes
- **Zoom:** scaling between 1.0x and 1.2x
- **Brightness:** multiplication factor between 0.4 and 1.2
- **Flip:** horizontal mirroring, with the steering angle negated

### Image preprocessing

The same preprocessing is applied during training and inference:

1. Crop rows 60 to 135 to remove the sky and the vehicle bonnet
2. Convert from RGB to YUV colour space
3. Apply a 3x3 Gaussian blur
4. Resize to 200 x 66 pixels (the input size used by NVIDIA)
5. Normalize pixel values to the range [0, 1]

---

## Model Architecture

Input shape: 66 x 200 x 3 (YUV image)

| Layer | Type | Filters / Units | Kernel | Stride | Activation |
|-------|------|-----------------|--------|--------|------------|
| 1 | Conv2D | 24 | 5x5 | 2x2 | ELU |
| 2 | Conv2D | 36 | 5x5 | 2x2 | ELU |
| 3 | Conv2D | 48 | 3x3 | 2x2 | ELU |
| 4 | Conv2D | 64 | 3x3 | 1x1 | ELU |
| 5 | Conv2D | 64 | 3x3 | 1x1 | ELU |
| 6 | Flatten | - | - | - | - |
| 7 | Dense | 100 | - | - | ELU |
| 8 | Dense | 50 | - | - | ELU |
| 9 | Dense | 10 | - | - | ELU |
| 10 | Dense (output) | 1 | - | - | Linear |

- **Optimizer:** Adam (learning rate 0.0001)
- **Loss function:** Mean Squared Error (regression on steering angle)

---

## Project Structure

```
CNN-NVIDIA-Self-Driving-Car/
|-- TrainingSimulation.py        # Data loading, model training and loss plot
|-- TestSimulation.py            # Socket.IO server that drives the car using the trained model
|-- utils.py                     # Balancing, augmentation, preprocessing, batch generator, model definition
|-- requirements.txt             # Python dependencies
|-- end-to-end-dl-using-px.pdf   # NVIDIA reference paper
|-- test.jpg                     # Sample image for testing augmentation and preprocessing
|-- udacity self driving car simulator download.txt   # Simulator download link
`-- .gitignore
```

The `data/` directory (recorded driving data) and the trained model files are not included in the repository. They must be generated as described below.

---

## Getting Started

### Prerequisites

- Python 3.x
- Udacity Self-Driving Car Simulator (download link provided in `udacity self driving car simulator download.txt`)

### Installation

```bash
# Clone the repository
git clone https://github.com/Sankalpa0011/CNN-NVIDIA-Self-Driving-Car.git
cd CNN-NVIDIA-Self-Driving-Car

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### 1. Collect training data

1. Open the simulator and select Training Mode.
2. Click Record, choose a folder named `data` inside the project directory, and drive several laps. Staying near the lane center and driving in both directions improves generalization.
3. After recording, the folder should have the following structure:

```
data/
|-- driving_log.csv
`-- IMG/
    |-- center_....jpg
    |-- left_....jpg
    `-- right_....jpg
```

### 2. Train the model

```bash
python TrainingSimulation.py
```

This balances the dataset, trains the network, saves `self_drive_model.h5` and `self_drive_model.keras`, and displays the loss curves.

### 3. Run in Autonomous Mode

```bash
python TestSimulation.py
```

Then start the simulator and select Autonomous Mode. The server listens on port 4567. For each telemetry frame it:

1. Decodes the camera image sent by the simulator
2. Applies the same preprocessing used in training
3. Predicts the steering angle using the trained model
4. Computes throttle as `1.0 - speed / maxSpeed` (with `maxSpeed = 10`), which reduces throttle as the car approaches the speed limit
5. Sends the steering angle and throttle back to the simulator

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Batch size | 100 |
| Epochs | 10 |
| Steps per epoch | 300 |
| Validation steps | 200 |
| Learning rate | 0.0001 |
| Train / validation split | 80% / 20% |
| Samples per steering bin | 2500 |
| Number of steering bins | 31 |

These parameters can be modified in `TrainingSimulation.py` and `utils.py`.

---

## Troubleshooting

- **`driving_log.csv` not found:** Ensure the recorded data is inside a `data/` folder next to the scripts.
- **Incorrect image paths during training:** `getName()` in `utils.py` splits file paths on `\\`, which matches logs recorded on Windows. For data recorded on Linux or macOS, change it to split on `/`.
- **Socket.IO or eventlet errors:** `requirements.txt` pins `python-socketio==4.2.1`. Newer versions are not compatible with the `socketio.Middleware` call used in `TestSimulation.py`.
- **Model file not found:** Run `TrainingSimulation.py` first so that `self_drive_model.h5` is generated.
- **Vehicle leaves the track:** Record additional data (particularly recovery manoeuvres), train for more epochs, or reduce `maxSpeed`.

---

## Limitations and Future Work

- The model uses only the center camera. The left and right camera images recorded by the simulator are not yet used for steering correction.
- Only steering is predicted by the network. Throttle is controlled by a simple speed-based rule.
- Evaluation is currently qualitative (autonomous driving in the simulator) alongside validation loss. Quantitative driving metrics, such as time on track or number of interventions, could be added.
- Possible extensions include testing on additional tracks, adding dropout or other regularization, and comparing against other architectures.

---

## References

- Bojarski, M. et al., *End to End Learning for Self-Driving Cars*, NVIDIA, 2016. A copy is included in this repository as `end-to-end-dl-using-px.pdf`.
- [Udacity Self-Driving Car Simulator](https://github.com/udacity/self-driving-car-sim)

---

## Author

Sankalpa - [github.com/Sankalpa0011](https://github.com/Sankalpa0011)
