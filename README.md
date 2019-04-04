# MRI-to-CT-DCNN-TensorFlow

# Preprocessing
<p align='center'>
  <img src="https://user-images.githubusercontent.com/37034031/55565646-2ceb1d80-5735-11e9-8f03-a21bea959aa5.png" width=800)
</p> 
  
 <p align='center'>
  <img src="https://user-images.githubusercontent.com/37034031/55565679-3d9b9380-5735-11e9-971d-2e75764f60e8.png" width=800)
</p> 
  
- N4 bias field correction using [N4ITK](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5445030)
- Histogram matching using [Dynamic histogram warping algorithm](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=537491)
- Binary head mask using [Otsu auto-thresholding](https://pdfs.semanticscholar.org/fa29/610048ae3f0ec13810979d0f27ad6971bdbf.pdf)

# References
- Tustison, Nicholas J., et al. "N4ITK: improved N3 bias correction." IEEE transactions on medical imaging 29.6 (2010): 1310.
- Cox, Ingemar J., Sébastien Roy, and Sunita L. Hingorani. "Dynamic histogram warping of image pairs for constant image brightness." Proceedings., International Conference on Image Processing. Vol. 2. IEEE, 1995.
- Otsu, Nobuyuki. "A threshold selection method from gray-level histograms." IEEE transactions on systems, man, and cybernetics 9.1 (1979): 62-66.  
