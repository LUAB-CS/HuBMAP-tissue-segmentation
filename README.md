# HuBMAP-tissue-segmentation

## Structure of this repository
- `configs` is static and contains training hyperparameters, folder paths, the model architecture, metrics, flags;
- `data_processing` contains all scripts related to data loading;
- `evaluation` and `submission` contain scripts related to performance evaluation and submission for the challenge;
- `utils` will be used to store utilities, custom scripts that can be called anywhere;
- `models` embeds the deep learning models;
- `notebooks` contains all Jupyter notebooks, used for launching the whole pipeline;
- `data` is not tracked by git and should contain the data from Kaggle.

## Requirements
Use `pip install -r requirements.txt` to install all modules necessary to run the project into your virtual environment.

## How to run the code
Use the notebooks `example.ipynb` to launch experiments. The model and parameters have to be chosen there (see commented stuff).

To use multitasking Unet (with organ classification and pixel size prediction), switch to the branch `baptiste/predict_organ_and_p_size`. You can view trained model in `notebooks/evaluate_model.ipynb`.
