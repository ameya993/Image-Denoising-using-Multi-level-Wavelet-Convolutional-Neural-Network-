This repository contains an implementation of an image denoising technique using Multi-level Wavelet Convolutional Neural Networks (MWCNN). The goal is to improve the quality of noisy images by utilizing wavelet transforms combined with Convolutional Neural Networks (CNNs).

Project Overview
Image denoising is an essential task in image processing, especially for real-world applications such as medical imaging, satellite imagery, and other computer vision applications where noise in images can reduce the quality of the data. This project demonstrates how a Multi-level Wavelet Convolutional Neural Network (MWCNN) can be used to remove noise from images effectively.

The model uses the wavelet transform to extract multi-scale features, which are then processed by convolutional neural networks for denoising.

Features
Wavelet Transform: The method uses multi-level wavelet decomposition to extract features at different scales.

Convolutional Neural Networks (CNNs): A CNN architecture is used to denoise the image by learning the underlying noise patterns.

Training & Testing: The model is trained on noisy images and tested to evaluate its performance on various types of noise.

Model Performance: The model's performance is evaluated using standard metrics like Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index (SSIM).

Dataset
The dataset used in this project is sourced from the SIDD (Smart Image Denoising Dataset), a high-quality dataset specifically designed for image denoising tasks. It contains noisy and clean image pairs, which are used for training and testing the denoising model.

You can access the SIDD dataset here: SIDD Dataset https://abdokamel.github.io/sidd/ small dataset. 

Features
Wavelet Transform: The method uses multi-level wavelet decomposition to extract features at different scales.

Convolutional Neural Networks (CNNs): A CNN architecture is used to denoise the image by learning the underlying noise patterns.

Training & Testing: The model is trained on noisy images and tested to evaluate its performance on various types of noise.

Model Performance: The model's performance is evaluated using standard metrics like Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index (SSIM).


Steps in the Notebook
Loading the dataset: The notebook loads noisy and clean images for training.

Preprocessing: Apply wavelet transform for feature extraction.

Model Building: Construct the Multi-level Wavelet Convolutional Neural Network.

Training: Train the model on noisy images.

Testing & Evaluation: Evaluate the model performance using PSNR and SSIM metrics.

Streamlit Web App: The notebook includes Streamlit code to create a simple web interface where users can upload a noisy image and see the denoised result in real-time.


RESULTS :  MWCNN  	Blind	sRGB	35.42dB/0.865 (PSNR/SSIM)

