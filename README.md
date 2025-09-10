# BNNForge

## Use Cases

### 1. Time-of-Flight (ToF) Sensor Classification
Located in `b-ToF.ipynb`, this implementation processes 8x8 ToF sensor data for binary classification. The model is designed to efficiently process spatial data from ToF sensors while maintaining high accuracy with minimal computational overhead.

### 4. Electronics Parts Classification
Located in `b-EPD.ipynb`, this notebook performs multi-class image classification on 30x30 grayscale images from the Electronics-Parts-Dataset. The data are fetched within the notebook and cleaned before training binarized CNNs.

### 3. Asphalt Pavement Condition Classification
Located in `b-AR.ipynb`, this notebook uses the AsphaltRegularity dataset to classify road surface conditions (e.g., regular vs. deteriorated) from accelerometer-derived 1D signals. It employs binarized 1D CNN layers for efficient inference.

### 2. Z24 Bridge Vibration Analysis
Located in `b-z24.ipynb`, this notebook analyzes vibration data from the Z24 Bridge dataset for structural health monitoring. It processes time-series data to detect and classify different structural conditions.

### 5. Speech Commands (V2) – 12-class
Located in `b-SC.ipynb`, this notebook builds a 12-class keyword spotter on Google Speech Commands v0.02. It extracts MFCC features and trains a compact binarized 1D CNN for efficient on-device speech recognition.

## Setup Instructions

### Development Environment Setup

1. Create and activate a new conda environment with Python 3.10:
```bash
conda create -n bnnforge python=3.10
conda activate bnnforge
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Run cell by cell in the provided Jupyter notebooks to train and evaluate models.

### Deployment on Raspberry Pi (ARM64)

#### Option 1: Building from Source

```bash
# Clone the Larq Compute Engine repository
git clone https://github.com/larq/compute-engine.git
cd compute-engine

# Initialize submodules
git submodule update --init

# Install dependencies
sudo apt-get install cmake build-essential

# Build the project
cmake -S . -B build
cmake --build build
```

#### Option 2: Using Release Version

```bash
# Download pre-built binary
wget https://github.com/larq/compute-engine/releases/download/v0.16.0/lce_benchmark_model_aarch64 -o lce_benchmark_model
```

### Benchmarking Models

To benchmark the deployed models on Raspberry Pi:

If you built from source, the binary will be in the `build` directory (`cd /home/username/compute-engine/build`). If you downloaded the release version, it should be in your current directory.

```bash
chmod +x lce_benchmark_model
./lce_benchmark_model --graph=/path/to/your/model.tflite --num_threads=1 --num_runs=100 --warmup_runs=5 --report_peak_memory_footprint=false
```

Benchmark parameters:
- `--num_threads`: Number of threads to use (1 for single-core performance)
- `--num_runs`: Number of inference runs (100 for reliable metrics)
- `--warmup_runs`: Number of initial runs to warm up the system (5 is typical)
- `--report_peak_memory_footprint`: Set to true to enable memory footprint reporting (approximate)

## Project Structure

```
.
├── b-AR.ipynb          # Asphalt Pavement condition classification (1D signals)
├── b-EPD.ipynb         # Electronics Parts image classification (30x30 grayscale)
├── b-SC.ipynb          # Speech Commands v2, 12-class keyword spotting
├── b-ToF.ipynb          # Time-of-Flight binary classification
├── b-z24.ipynb          # Z24 Bridge 15-class classification
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

## Dependencies

Key dependencies include:
- TensorFlow 2.14.1
- Larq
- Larq Compute Engine
- Keras-tuner
- Other standard ML libraries (see requirements.txt)

## Citing

Soon available.