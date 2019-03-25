# MRI-to-CT-DCNN-TensorFlow

# Preprocessing
- Binary head mask using [Otsu auto-thresholding](https://pdfs.semanticscholar.org/fa29/610048ae3f0ec13810979d0f27ad6971bdbf.pdf)
- N4 bias field correction using [N4ITK](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5445030)

# References
- Otsu, Nobuyuki. "A threshold selection method from gray-level histograms." IEEE transactions on systems, man, and cybernetics 9.1 (1979): 62-66.
- Tustison, Nicholas J., et al. "N4ITK: improved N3 bias correction." IEEE transactions on medical imaging 29.6 (2010): 1310.
