
REM This file does the training
@echo off
Color 1E

REM Prepare the dataset for training
python prepare_dataset.py

REM Representation Learning
python autoencoder.py
python autoencoder_ann.py
python vanilla_ann.py
python train_pca.py

REM Data Rebalance
python GAN.py
python data_rebalance.py

REM Other files to be run
python pca_eda.py
