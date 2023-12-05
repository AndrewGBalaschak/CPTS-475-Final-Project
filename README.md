# CPTS-475-Final-Project
This project aims to use a neural network to increase the dataset density of minority classes

## Dataset
We are using the NF-UQ-NIDS dataset, which can be downloaded at https://staff.itee.uq.edu.au/marius/NIDS_datasets/
Extract the files and place every file in a directory named "data" in the repo, without any subfolders.

## Data Pre-Processing
Run the `data-cleaning` script to clean the data, run the script with the `-reduce` flag to perform stratified subsampling.

## Classification
Open the `classifier` notebook and run the cells in order to train and evaluate the classifiers. The training step can be skipped if you prefer to load a pre-trained model for evaluation.
