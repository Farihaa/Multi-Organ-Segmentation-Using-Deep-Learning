# Multi-Organ-Segmentation-Using-Deep-Learning
NiftyNet framework based on Tensorfow was used for this project.
```bash
pip install niftynet
```
## Dataset
Dataset was acquired from MICCAI 2012, MRBrains18 and ADNI.

## Preprocessing
3D medical images were used as dataset and due to their large sizes and smaller number, they had to be resampled and augmented.
SimpleITK library was used for preprocessing as seen in [data_aug.py](https://github.com/Farihaa/Multi-Organ-Segmentation-Using-Deep-Learning/blob/master/data_aug.py)file.
```bash
pip install SimpleITK
```
Furthermore, data was increased by collecting it from various sites like MICCAI 2012, MRBRains18 and ADNI. Some images came with their annotations
but some of them had to be segmented manually for training. [ITKSNAP](http://www.itksnap.org/pmwiki/pmwiki.php) was used for manual annotations of 3D images.

## Deep Learning Network
HighRes3D Networks were used for segmentation of 3 regions and 8 regions respectively. Checkpoints have been uploaded.

## Results
Dice score of 0.957 and 0.776 was achieved for 3 brain regions and 8 brain regions respectively.
