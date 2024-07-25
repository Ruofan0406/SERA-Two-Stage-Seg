# SERA-Two-Stage-Seg
This is the reference code for the paper ["A two-stage CNN method for MRI image segmentation of prostate with lesion."](https://www.sciencedirect.com/science/article/abs/pii/S1746809423000435).

# Code Update Log
07/05/2024: Upload the code for building the classification and segmentation model proposed in the paper.

# Dataset Availability
The dataset used in this experiment is confidential due to the need to protect patient privacy.  
However, you can run the code using your own dataset. Here are the tips:   
For classification dataset: format: ./data/include/0_1.png 0_2.png ... ./data/exclude/0_0.png 0_10.png ...  
You put images of the same category in the same folder and images of different categories in different folders.  
For segmentation dataset: format: ./data/fold_1/1/image.bmp mask.bmp  
You put both the image and the mask under one folder.  

# Environment Preparation
It is recommended to build two different environments using Anaconda.  
You need GPU support, cuda version suggestion: cuda 11.8  
For classification, the code is built based on Keras(tensorflow~=2.14.0 & keras~=2.14.0).  
For Segmentation, the code is built based on PyTorch(11.8).    

# Paper Information
Please cite this paper after referring to the code.  

BibTex:  
@article{wang2023two,  
  title={A two-stage CNN method for MRI image segmentation of prostate with lesion},  
  author={Wang, Zixuan and Wu, Ruofan and Xu, Yanran and Liu, Yi and Chai, Ruimei and Ma, He},  
  journal={Biomedical Signal Processing and Control},  
  volume={82},  
  pages={104610},  
  year={2023},  
  publisher={Elsevier}  
}

# Contact Information
If you have any problems when using this code, please contact: rofanfanwu@gmail.com  
I will try my best to help you!
