#!/usr/bin/env python

from __future__ import print_function

import SimpleITK as sitk
import sys
import os

if len(sys.argv) < 2:
    print("Usage: N4BiasFieldCorrection inputImage " +
          "outputImage [shrinkFactor] [maskImage] [numberOfIterations] " +
          "[numberOfFittingLevels]" )
    sys.exit(1)

# 0: N4BiasFieldCorrection
# 1: inputImage
# 2: outputImage
# 3: [shrinkFactor]
# 4: [maskImage]
# 5: [numberOfIterations]
# 6: [numberOfFittingLevels]

inputImage = sitk.ReadImage(sys.argv[1])

if len(sys.argv) > 4:
    maskImage = sitk.ReadImage(sys.argv[4], sitk.sitkUint8)
else:
    maskImage = sitk.OtsuThreshold(inputImage, 0, 1, 200)

if len(sys.argv) > 3:
    inputImage = sitk.Shrink(inputImage, [int(sys.argv[3])] * inputImage.GetDimension())
    maskImage = sitk.Shrink(maskImage, [int(sys.argv[3])] * inputImage.GetDimension())

inputImage = sitk.Cast(inputImage, sitk.sitkFloat32)
corrector = sitk.N4BiasFieldCorrectionImageFilter()
numberFittingLevels = 4

if len(sys.argv) > 6:
    numberFittingLevels = int(sys.argv[6])

if len(sys.argv) > 5:
    corrector.SetMaximumNumberOfIterations([int(sys.argv[5])] *numberFittingLevels)

output = corrector.Execute(inputImage, maskImage)
sitk.WriteImage(output, sys.argv[2])

if not "SITK_NOSHOW" in os.environ:
    sitk.Show(output, "N4 Corrected")