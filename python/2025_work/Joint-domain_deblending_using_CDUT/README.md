# Joint-Domain Seismic Deblending with CDUT

This repository contains the training, iterative data-generation, and inference
code for joint-domain seismic deblending with CDUT.

The main entry points are listed below.

Initial model training:
- `main_training_crg.py`
- `main_training_csg.py`

Iterative training-data generation:
- `set_crg_iter.py`
- `set_csg_iter.py`

Different training stages or parameter settings can be selected by changing the
`--cfg` configuration file passed to each script.

- `main_test1.py`: PGS blended-data experiment.
- `main_test1_est.py`: Viking experiment with interference estimation.

Large training datasets, experimental datasets, and pretrained checkpoints are
not included in this source tree. Before running the scripts, download the
required files and place them in the expected locations shown below.

Download link for the training data in `data/`, pretrained checkpoints in
`output/`, and experimental datasets in `test/`:
https://drive.google.com/drive/folders/165mH9PYvdMZ8SqCR9P-02yCLEHvt-YUZ?usp=drive_link

After downloading the files, place the `data/`, `output/`, and `test/`
directories in the project root.

## Required Directory Layout

Checkpoint files:

```text
output/
  CDUTnet_2single_csg_0.4/default/ckpt_epoch_199.pth
  CDUTnet_2single_csg_0.4_temp/default/ckpt_epoch_199.pth
  CDUTnet_2single_crg/default/ckpt_epoch_199.pth
  CDUTnet_2single_crg_1_iter/default/ckpt_epoch_199.pth
  CDUTnet_2single_crg_1_iter_5/default/ckpt_epoch_199.pth
```

PGS data required by `main_test1.py`:

```text
test/PGS_blended_data_time/blend_data/
  d1.dat
  d2.dat
  d1b.dat
  d2b.dat
  delay_old.dat
```

Viking data required by `main_test1_est.py`:

```text
test/viking/blend_data/
  d1.dat
  d2.dat
  d1b.dat
  d2b.dat
  t1.dat
  t2.dat
```

## Environment

Install the Python dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

The scripts assume a CUDA-enabled PyTorch environment. If needed, adjust
`CUDA_VISIBLE_DEVICES` in the entry scripts to match your hardware setup.

## Usage

Train the network models:

```bash
python main_training_crg.py --cfg configs/CDUTnet_2single_crg_1.yaml --batch_size xx
python main_training_csg.py --cfg configs/CDUTnet_2single_csg_0.4.yaml --batch_size xx
```

Use trained models to update the training data and perform iterative training:

```bash
python set_crg_iter.py --cfg configs/CDUTnet_2single_crg_1.yaml --batch_size xx
python set_csg_iter.py --cfg configs/CDUTnet_2single_csg_0.4.yaml --batch_size xx
```

Run the test experiments:

```bash
python main_test1.py --cfg configs/CDUTnet_2single_crg_1.yaml
python main_test1_est.py --cfg configs/CDUTnet_2single_crg_1.yaml
```

## License

The source headers state that this project is released under GPL v3 or later.
