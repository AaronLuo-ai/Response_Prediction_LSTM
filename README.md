Rectal Cancer Treatment Response Prediction
Overview
This project focuses on predicting the post-treatment response of rectal cancer patients using pre- and post-treatment MRI scans. The goal is to leverage machine learning techniques to analyze imaging data and provide insights into treatment outcomes, which can assist clinicians in making informed decisions.

Dataset
The dataset consists of MRI scans from rectal cancer patients, including both pre-treatment (MR1) and post-treatment (MR2) images. Each patient has corresponding image and mask files in .nrrd format. The dataset is structured as follows:

Pre-treatment MRI: cnda_pXXX_MR1_images.nrrd

Post-treatment MRI: cnda_pXXX_MR2_images.nrrd

Masks: Corresponding segmentation masks for each MRI scan.

Only patients with both pre- and post-treatment data are included in the analysis to ensure a complete dataset for training and evaluation.

Project Structure