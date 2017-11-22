# Dense 3D Regression for Hand Pose Estimation

This respository contains python codes of the paper. It is developped and tested on Debian GNU/Linux 8 64-bit.

## Requirements:
- tensorflow >= 1.4
- tfplot (for visualization on tf summary files)
- numpy
- opencv >= 2.4 (optional, for cpu visualization) 

## Data Preparations:
Download the datasets, create soft links for them to [exp/data](./exp/data) and run data/$dataset.py to create the TFRecord files. Details is described in [here](./exp/data).

## Usage:
Both training and testing function is provided by model/hourglass\_um\_crop\_tiny.py. Here is an example:
```bash
python model/hourglass_um_crop_tiny.py --dataset 'icvl' --batch_size 40 --num_stack 2 --fea_num 128 --debug_level 2
```
where the hyper parameter configuration is explained in the source python files.

## Results:
We provide the estimation results by the proposed method for [ICVL](./exp/result/icvl.txt), [NYU](./exp/result/nyu.txt), [MSRA15](./exp/result/msra.txt). They are xyz coordinates in mm, the 2D projection method is in the function _xyz2uvd_ from [here](data/util.py#L23)

## Pretrained Models:
Coming soon
