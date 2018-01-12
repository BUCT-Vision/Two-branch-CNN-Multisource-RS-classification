# Two-branch-CNN-Multisource-RS-classification
This example implements the paper [Multisource Remote Sensing Data Classification Based on Convolutional Neural Network](http://ieeexplore.ieee.org/document/8068943/)

A two-branch CNN architecture for feasture fusion with HSI and other remote scensing imagery. Reach a quite high classification accuracy. Evaluated on the dataset of Houston, Trento, Salinas and Pavia. 

![](https://github.com/Hsuxu/Two-branch-CNN-Multisource-RS-classification/blob/master/figs/arch-01.PNG)

## Prerequisites
- System *Ubuntu 14.04 or upper* 
- Python 2.7 or 3.6
- Packages
```
pip install -r requirements.txt
```

## Usage
### dataset utilization
**Please modify line 10-22 in *data_util.py* for the dataset details.**

### Training
1. Train HSI
```
python main.py --train hsi --epochs 20 --modelname ./logs/weights/hsi.h5
```
2. Train LiDAR/VIS
```
python main.py --train lidar --epochs 20 --modelname ./logs/weights/lidar.h5
```
3. Train two branches
```
python main.py --train finetune --epochs 20 --modelname ./logs/weights/model.h5
```

## Dataset
Please contact <hsuxu820@gmail.com>

## Results
All the results are cited from original paper. More details can be found in the paper.

| dataset   | Kappa | OA       |
|-----------|-------|----------|
| Houston   | 0.8698| 87.98%   |
| Trento    | 0.9681| 97.92%   |
| Pavia     | 0.9883| 99.13%   |
| Salinas   | 0.9745| 97.72%   |

## Citation
```
@article{xu2017multisource,
  title={Multisource Remote Sensing Data Classification Based on Convolutional Neural Network},
  author={Xu, Xiaodong and Li, Wei and Ran, Qiong and Du, Qian and Gao, Lianru and Zhang, Bing},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2017},
  publisher={IEEE}
}
```
## TODO
1. pytorch version.
2. more flexiable dataset utilization
