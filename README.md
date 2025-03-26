# Binary Neural Networks for Embedded Systems

## Use Cases

### 1. Time-of-Flight (ToF) Sensor Classification
Located in `b-ToF.ipynb`, this implementation processes 8x8 ToF sensor data for binary classification. The model is designed to efficiently process spatial data from ToF sensors while maintaining high accuracy with minimal computational overhead.

### 2. Z24 Bridge Vibration Analysis
Located in `b-z24.ipynb`, this model analyzes vibration data from the Z24 Bridge dataset for structural health monitoring. It processes time-series data to detect and classify different structural conditions.

## Setup Instructions

### Development Environment Setup

1. Create and activate a new conda environment with Python 3.10:
```bash
conda create -n bnn python=3.10
conda activate bnn
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

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

If you use this work in an academic context, please consider citing our paper:

```
@article{your_paper_reference,
  title={Title of Your Paper},
  author={Your Name},
  journal={Journal Name},
  year={Year of Publication}
}
```
