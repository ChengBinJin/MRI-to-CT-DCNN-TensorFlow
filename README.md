# MRI-to-CT-DCNN-TensorFlow
This repository is an implementation of ["MR‐based synthetic CT generation using a deep convolutional neural network method." Medical physics 2017"](https://aapm.onlinelibrary.wiley.com/doi/pdf/10.1002/mp.12155) by Xiao Han.

<p align='center'>
  <img src="https://user-images.githubusercontent.com/37034031/56449995-fd0a5f80-635b-11e9-87c5-9383fe57b820.png" width=600)
</p>  

## Requirements
- tensorflow 1.13.1
- numpy 1.15.2
- opencv 3.4.3
- matplotlib 2.2.3
- pickleshare 0.7.4
- simpleitk 1.2.0
- scipy 1.1.0

# Preprocessing
<p align='center'>
  <img src="https://user-images.githubusercontent.com/37034031/56449965-b583d380-635b-11e9-97c1-fc3e691cae2e.png" width=800)
</p> 
  
<p align='center'>
  <img src="https://user-images.githubusercontent.com/37034031/56449970-c16f9580-635b-11e9-9737-0ab8326e4b40.png" width=800)
</p> 
  
<p align='center'>
  <img src="https://user-images.githubusercontent.com/37034031/56449971-c896a380-635b-11e9-8657-195451fb7336.png" width=800)
</p> 
  
- N4 bias field correction using [N4ITK](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5445030)
- Histogram matching using [Dynamic histogram warping algorithm](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=537491)
- Binary head mask using [Otsu auto-thresholding](https://pdfs.semanticscholar.org/fa29/610048ae3f0ec13810979d0f27ad6971bdbf.pdf)

# References
- Tustison, Nicholas J., et al. "N4ITK: improved N3 bias correction." IEEE transactions on medical imaging 29.6 (2010): 1310.
- Cox, Ingemar J., Sébastien Roy, and Sunita L. Hingorani. "Dynamic histogram warping of image pairs for constant image brightness." Proceedings., International Conference on Image Processing. Vol. 2. IEEE, 1995.
- Otsu, Nobuyuki. "A threshold selection method from gray-level histograms." IEEE transactions on systems, man, and cybernetics 9.1 (1979): 62-66.  
