# HuBMAP-tissue-segmentation

## Structure of this repository
- `configs` is static and contains training hyperparameters, folder paths, the model architecture, metrics, flags;
- `data_processing` contains all scripts related to data loading;
- `evaluation` and `submission` contain scripts related to performance evaluation and submission for the challenge;
- `utils` will be used to store utilities, custom scripts that can be called anywhere;
- `models` embeds the deep learning models;
- `notebooks` contains all Jupyter notebooks, used for launching the whole pipeline.

## Requirements
Use `pip install -r requirements.txt` to install all modules necessary to run the project into your virtual environment.