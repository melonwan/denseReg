# Dense 3D Regression for Hand Pose Estimation

This respository contains tensorflow implementation of the [paper](https://arxiv.org/abs/1711.08996). It is developped and tested on Debian GNU/Linux 8 64-bit.

## Requirements:
- python 2.7
- tensorflow == 1.3
- [tfplot](https://github.com/wookayin/tensorflow-plot) (for visualization on tf summary files)
- matplotlib >= 2.0.2 
- numpy
- opencv >= 2.4 (optional, for cpu visualization) 

## Data Preparations:
Download the datasets, create soft links for them to [exp/data](./exp/data) and run `python data/${dataset}.py` to create the TFRecord files. Details are [here](./exp/data).

## Usage:
Both training and testing functions are provided by `model/hourglass_um_crop_tiny.py`. Here is an example:
```bash
python model/hourglass_um_crop_tiny.py --dataset 'icvl' --batch_size 40 --num_stack 2 --fea_num 128 --debug_level 2 --is_train True
```
where the hyper parameter configuration is explained in the source python files.

## Results:
We provide the estimation results by the proposed method for [ICVL](./exp/result/icvl.txt), [NYU](./exp/result/nyu.txt), [MSRA15](./exp/result/msra.txt). They are xyz coordinates in mm, the 2D projection method is in the function _xyz2uvd_ from [here](data/util.py#L23). Check [here](https://github.com/xinghaochen/awesome-hand-pose-estimation/tree/master/evaluation) for comparison to other methods. Thanks @xinghaochen for providing the comparison.

## Pretrained Models:
Run the script to download and install the corresponding trained model of datasets. $ROOT denote the root path of this project.
```bash
cd $ROOT
./exp/scripts/fetch_icvl_models.sh
./exp/scripts/fetch_msra_models.sh
./exp/scripts/fetch_nyu_models.sh
```
To perform testing, simply run
```
python model/hourglass_um_crop_tiny.py --dataset 'icvl' --batch_size 3 --num_stack 2 --num_fea 128 --debug_level 2 --is_train False
python model/hourglass_um_crop_tiny.py --dataset 'nyu' --batch_size 3 --num_stack 2 --num_fea 128 --debug_level 2 --is_train False
python model/hourglass_um_crop_tiny.py --dataset 'msra' --pid 0 --batch_size 3 --num_stack 2 --num_fea 128 --debug_level 2 --is_train False
```
in which msra dataset should use pid to indicate which person to test on. In the [testing function](data/hourglass_um_crop_tiny.py#L23), the third augument is used to indicate which model with corresponding training step will be restored. We use step of -1 to indicate our pre-trained model.
