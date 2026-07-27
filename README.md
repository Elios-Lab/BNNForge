# BNNForge

BNNForge is the companion implementation for experiments on binarized neural
networks (BNNs) for resource-constrained and embedded systems. The repository
contains five end-to-end Jupyter workflows covering image, spatial-sensor,
audio, and time-series classification.

Each notebook follows the same experimental pipeline:

1. acquire and preprocess the dataset;
2. create training, validation, and test splits;
3. tune a compact BNN with Keras Tuner;
4. train and evaluate it with accuracy, precision, recall, F1 score, and a
   confusion matrix;
5. export the trained model to TensorFlow Lite with Larq Compute Engine (LCE);
6. generate SHAP explanations; and
7. optionally convert the Keras model to C with CBin-NN.

## Experiments

| Notebook | Task | Input | Classes | Dataset |
| --- | --- | --- | ---: | --- |
| `b-AR.ipynb` | Asphalt pavement condition classification | 1D accelerometer-derived signal | 2 | [AsphaltRegularity](https://www.timeseriesclassification.com/description.php?Dataset=AsphaltRegularity) |
| `b-EPD.ipynb` | Electronic-parts classification | 30 x 30 grayscale image | Multiple | [Electronics-Parts-Dataset](https://github.com/praneelchand10/Electronics-Parts-Dataset) |
| `b-SC.ipynb` | Keyword spotting | MFCC representation of one-second audio | 12 | [Speech Commands v0.02](http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz) |
| `b-ToF.ipynb` | Time-of-Flight sensor classification | 8 x 8 distance map | 2 | Downloaded by the notebook |
| `b-z24.ipynb` | Structural-condition classification | 1D bridge-vibration signal | 15 | [Z24 Bridge](https://bwk.kuleuven.be/bwm/z24) |

The Z24 data are not public and are therefore not downloaded automatically.
Request access from the dataset owner and arrange the files under
`DatasetPDT/<class>/avt/` as expected by `b-z24.ipynb`.

## Repository contents

```text
.
├── b-AR.ipynb                 # AsphaltRegularity experiment
├── b-EPD.ipynb                # Electronic-parts experiment
├── b-SC.ipynb                 # Speech Commands experiment
├── b-ToF.ipynb                # Time-of-Flight experiment
├── b-z24.ipynb                # Z24 Bridge experiment
├── explainability_analysis.py # Shared FP-versus-BNN SHAP utilities
├── requirements.txt           # Training and LCE dependencies
└── README.md
```

Datasets, trained models, tuner state, generated C code, and analysis outputs
are intentionally excluded from version control. This keeps the repository
small and avoids redistributing restricted data. Consequently, the repository
contains the experiment definitions rather than pretrained weights or recorded
notebook outputs.

## Environment setup

Python 3.10 is recommended because the notebooks use TensorFlow 2.14.1.

```bash
conda create -n bnnforge python=3.10
conda activate bnnforge
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The shared explainability module additionally imports SHAP, pandas, Joblib, and
Pillow:

```bash
python -m pip install shap pandas joblib pillow
```

Start Jupyter from the repository root and run the selected notebook from top
to bottom:

```bash
jupyter lab
```

The first execution of the AR, EPD, SC, or ToF notebook requires network
access to retrieve its dataset. Hyperparameter searches and model training can
be time-consuming; the notebooks use early stopping and expose the relevant
search and training parameters in their respective cells.

## Outputs

The notebook export cells create the following LCE-compatible FlatBuffer
models:

| Experiment | Exported model |
| --- | --- |
| AR | `AR_model.tflite` |
| EPD | `EPD_model.tflite` |
| SC | `SC_model.tflite` |
| ToF | `ToF_model.tflite` |
| Z24 | `z24_model.tflite` |

These files are generated locally and ignored by Git.

## Explainability analysis

`explainability_analysis.py` compares full-precision (FP) and binarized TFLite
models with SHAP. It supports segmented KernelSHAP for high-dimensional 1D and
2D inputs and writes signed attribution plots to `explainability_output/`.

The standalone script expects all five model pairs in `--model_dir`:

```text
AR_fp.tflite       AR_bnn.tflite
EPD_fp.tflite      EPD_bnn.tflite
SC_fp.tflite       SC_bnn.tflite
ToF_fp.tflite      ToF_bnn.tflite
z24_model.tflite   z24_bnn.tflite
```

It also loads all five datasets at startup, so the complete input tree must be
available even when explaining one task through `run_multi`:

```text
inputs/
├── AR/AR_test_X.npy
├── AR/AR_test_y.npy
├── EPD/EPD_test_data.npy
├── EPD/EPD_test_label.npy
├── SC/SC_test_X.npy
├── SC/SC_test_y.npy
├── ToF/Tof_sample.csv
├── ToF/ToF_label.txt
├── Z24/z24_data.pkl
└── Z24/z24_label.pkl
```

Run the full FP-versus-BNN comparison with:

```bash
python explainability_analysis.py \
  --data_dir inputs \
  --model_dir path/to/models \
  --out_dir explainability_output \
  --sample 0 \
  --n_shap_bg 30
```

The explainability cells in the notebooks demonstrate how test samples are
serialized and how `run_multi` is called for a selected task.

## C code generation

The final cells of every notebook provide an optional C export workflow based
on [CBin-NN](https://github.com/Edge-Learning-Machine/CBin-NN). They:

1. save the trained model as an HDF5 (`.h5`) file;
2. clone CBin-NN if it is not already present;
3. copy the model into `CBin-NN/Code Generation/`;
4. update the converter's `model_name`; and
5. execute `model_converter.py`.

Run these cells only after training and consult the CBin-NN repository for its
compiler, runtime, and target-specific requirements.

## Raspberry Pi benchmarking

The notebooks export models through `larq_compute_engine`, whose optimized
operators target 64-bit ARM platforms including Raspberry Pi. Build the LCE
benchmark tool by following the
[official 64-bit ARM instructions](https://docs.larq.dev/compute-engine/end_to_end/#step-3-build-the-lce-benchmark-tool),
then benchmark a generated model with:

```bash
./lce_benchmark_model \
  --graph=/path/to/model.tflite \
  --num_threads=1 \
  --num_runs=100 \
  --warmup_runs=5 \
  --report_peak_memory_footprint=false
```

Use the same board, operating-system image, CPU governor, thread count, warm-up
count, and run count when comparing models.

## Reproducibility notes

- Dataset preparation, split logic, random seeds, model definition,
  hyperparameter search space, training schedule, and evaluation are contained
  in each notebook.
- Notebook outputs are stripped when committed; rerun the notebooks to
  reproduce tables, plots, and exported artifacts.
- TensorFlow is pinned to 2.14.1 in `requirements.txt`; record the resolved
  versions of Larq, Larq Compute Engine, and Keras Tuner for archival runs.
- The Z24 experiment requires separately authorized access to the source data.

## Citation

Citation information will be added when the associated paper is published.
