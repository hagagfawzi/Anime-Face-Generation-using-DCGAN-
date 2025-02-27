# **Anime Face Generation using DCGAN** 🎨🚀

## **Overview**
This project implements a **Deep Convolutional Generative Adversarial Network (DCGAN)** using **TensorFlow** to generate high-quality **anime faces**. The model is trained on a dataset of **21,551 anime images**, leveraging convolutional layers and adversarial training for realistic face synthesis.

## **Dataset**
- The dataset consists of **21,551 anime face images** scraped from **www.getchu.com** and available on [Kaggle](https://www.kaggle.com/datasets/soumikrakshit/anime-faces).
- Faces are **cropped and resized (64×64)** using the [lbpcascade_animeface](https://github.com/nagadomi/lbpcascade_animeface) detection algorithm.
- Some **outliers** (e.g., bad crops, non-human faces) may be present.
- Feel free to contribute to this dataset by adding images of similar quality or adding image labels.

### **Dataset Description**
This dataset consists of **21,551 anime faces** scraped from **www.getchu.com**, cropped using the anime face detection algorithm from [lbpcascade_animeface](https://github.com/nagadomi/lbpcascade_animeface), and resized to **64×64** for convenience. Please cite both sources when using this dataset.

Some outliers are still present in the dataset:
- **Bad cropping results**
- **Some non-human faces**

## **Model Architecture**
The **DCGAN** consists of two networks:
1. **Generator**
   - Uses **transposed convolutional layers** to upsample noise into a realistic face.
   - Batch normalization and LeakyReLU activations for stable training.
2. **Discriminator**
   - A **CNN-based classifier** that distinguishes between real and generated images.
   - Uses **strided convolutions, batch normalization, and LeakyReLU activations**.

## **Training Details**
- **Noise Dimension:** 100
- **Loss Function:** Binary Cross-Entropy (BCE)
- **Optimizer:** Adam (learning rate = 0.0002, beta1 = 0.5)
- **Batch Size:** 64
- **Epochs:**500
- **Framework:** TensorFlow/Keras

## **Preprocessing & Augmentation**
- **Resizing** all images to **64×64**.
- **Normalization** (scaling pixel values between -1 and 1).
- **Data augmentation** (optional) to improve training stability.

## **Results**
- The model successfully generates **high-quality anime faces** with clear features.
- Training is **stable** with well-balanced **generator and discriminator losses**.
- Output samples improve as **training progresses**.

## **Installation & Usage**
### **Dependencies**
Install required libraries:
```bash
pip install tensorflow numpy matplotlib
```
### **Run Training**
```bash
python train.py
```
### **Generate Anime Faces**
After training, generate new faces:
```bash
python generate.py
```

## **Future Improvements**
- Train on a **larger dataset** for better diversity.
- Experiment with **StyleGAN** for higher-quality results.
- Optimize for **faster inference** and deployment.

## **References**
- [DCGAN Paper](https://arxiv.org/abs/1511.06434)
- [TensorFlow DCGAN Tutorial](https://www.tensorflow.org/tutorials/generative/dcgan)
- [lbpcascade_animeface](https://github.com/nagadomi/lbpcascade_animeface)
- [Kaggle Anime Faces Dataset](https://www.kaggle.com/datasets/soumikrakshit/anime-faces)
