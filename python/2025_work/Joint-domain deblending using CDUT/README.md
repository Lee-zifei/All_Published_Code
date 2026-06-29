# Multi-Source Seismic Data Separation and Imaging Network

This repository contains the training and testing pipelines for a deep learning-network designed for seismic data separation (deblending) across different data domains and noise amplitudes.

## Requirements

To set up the environment and install the required dependencies, use the following `conda` command:

**Bash**

```
conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=10.2 -c pytorch
```


## Network Training

### 1. Base Training

To train the network, run `main.py` with the appropriate configuration file. Each YAML file corresponds to a specific data domain and a distinct amplitude of blending noise.

**Bash**

```
python main.py --cfg configs/xxxx.yaml --batch_size 16
```

### 2. Iterative Training

For iterative refinement, use the data generation script to produce the deblended outputs from the previous iteration, which will serve as the input for the next training cycle.

**Bash**

```
python set_iter_temp_data.py
```

## Network Testing & Inference

Run the testing scripts using the corresponding configuration files:

**Bash**

```
python <script_name>.py --cfg configs/xxxx.yaml
```

Choose the appropriate script based on your evaluation scenario:

| **Script Name**         | **Domain / Mode**                 | **Data / Acquisition Type**           |
| ----------------------------- | --------------------------------------- | ------------------------------------------- |
| **`main_crg.py`**     | CRG (Common Receiver Gather) separation | Dithered-source data                        |
| **`main_crg_est.py`** | CRG (Common Receiver Gather) separation | ISS (Independent Simultaneous Sources) data |
| **`main_csg.py`**     | CSG (Common Shot Gather) separation     | Standard blended data                       |
| **`main_test.py`**    | Joint CSG-CCG iterative separation      | Dithered-source data                        |
| **`main_test1.py`**   | Joint CSG-CCG iterative separation      | ISS (Independent Simultaneous Sources) data |

## Pre-trained Models & Datasets

Pre-trained models and datasets are available upon request. If you need access to these resources for validation or replication, please contact the author directly.
