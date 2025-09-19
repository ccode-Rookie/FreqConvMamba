# FreqConvMamba
This is the official code repository for "FreqConvMamba: Frequency-guided Hierarchical Hybrid SSM-CNN for Medical Image Segmentation".
## 0. Main Environments
```bash
conda create -n Your virtual environment python=3.10
conda activate Your virtual environment
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 
pip install packaging
pip install timm==1.0.15 
pip install pytest chardet yacs termcolor
pip install submitit tensorboardX
pip install causal_conv1d==1.4.0  
pip install mamba_ssm==2.2.2  
pip install 。。。Other packages
```


## 1. Prepare the dataset

### ISIC or TN3K or MMOTU datasets
- The ISIC17 and ISIC18 datasets, divided into a 7:3 ratio, can be found here {[Baidu](https://pan.baidu.com/s/1Y0YupaH21yDN5uldl7IcZA?pwd=dybm) or [GoogleDrive](https://drive.google.com/file/d/1XM10fmAXndVLtXWOt5G0puYSQyI2veWy/view?usp=sharing)}. 
- MMOTU can be downloaded from this place:{[MMOTU](https://figshare.com/articles/dataset/_zip/25058690?file=44222642)}
- After downloading the datasets, you are supposed to put them into './data/datasetname/' , and the file format reference is as follows. (take the ISIC17 dataset as an example.)

- './data/isic17/'
  - train
    - images
      - .png
    - masks
      - .png
  - val
    - images
      - .png
    - masks
      - .png

### Synapse datasets

- For the Synapse dataset, you could follow [Swin-UNet](https://github.com/HuCaoFighting/Swin-Unet) to download the dataset, or you could download them from {[Baidu](https://pan.baidu.com/s/1JCXBfRL9y1cjfJUKtbEhiQ?pwd=9jti)}.

- After downloading the datasets, you are supposed to put them into './data/Synapse/', and the file format reference is as follows.

- './data/Synapse/'
  - lists
    - list_Synapse
      - all.lst
      - test_vol.txt
      - train.txt
  - test_vol_h5
    - casexxxx.npy.h5
  - train_npz
    - casexxxx_slicexxx.npz
  
### ACDC datasets
This dataset can be accessed through this connection.The division of the dataset is also shown according to this connection.{[Baidu](
https://pan.baidu.com/s/1TSWteL9Z_rciGy5kQsW5Zg?pwd=1234)}.For this dataset, we rely on the data loading method of this model{[Github](
https://github.com/kathyliu579/CS-Unet)}, and all comparative 
experiments are trained using this method.

## 2. Train the FreqConvMamba
```bash
cd Your virtual environment
python train.py  # Train and test FreqConvMamba on the ISIC17 or ISIC18 or TN3K or MMOTU dataset.
python train_synapse.py  # Train and test FreqConvMamba on the Synapse dataset.
python trainacdc.py #Train and test FreqConvMamba on the ACDC dataset.
```

## 3. Obtain the outputs
- After trianing, you could obtain the results in './results/'

## 4. Acknowledgments

- We thank the authors of [
CS-Unet](https://github.com/kathyliu579/CS-Unet) and [VM-UNet](https://github.com/JCruan519/VM-UNet) for their open-source codes.
