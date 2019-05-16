# An Empirical Study of Deep Models for Monocular Depth Estimation
This codebase implements the system described in the paper:

An Empirical Study of Deep Models for Monocular Depth Estimation
<!-- ****************(need to be change)
[Zhicheng Fang](),

In WACV 2020.

See the [project webpage](https://people.eecs.berkeley.edu/~tinghuiz/projects/SfMLearner/) for more details. 

![sample_results](misc/cityscapes_sample_results.gif)
**************** -->

## Preamble
This codebase was developed and tested with Pytorch 0.4.1, CUDA 9.1 and Ubuntu 16.04. And it is built based on [SfMLearner Pytorch version](https://github.com/ClementPinard/SfmLearner-Pytorch)

## Prerequisite

```bash
pip3 install -r requirements.txt
```

or install manually the following packages :

```
pytorch >= 0.4.1
imageio
scipy
argparse
tensorboardX
blessings
progressbar2
path.py
```

It is also advised to have python3 bindings for opencv for tensorboard visualizations

### What has been done

* Training has been tested on KITTI and NYU Depth v2.
* As for the multiscale loss, the loss weight is divided by `2.3` when downscaling instead of `2`. This is the results of empiric experiments, so the optimal value is clearly not carefully determined.

## Preparing training data
Preparation is roughly the same as in the SfMLearner Pytorch version.

For [KITTI](http://www.cvlibs.net/datasets/kitti/raw_data.php), first download the dataset using this [script](http://www.cvlibs.net/download.php?file=raw_data_downloader.zip) provided on the official website, and then run the following command. The `--with-depth` option will save resized copies of groundtruth to help you setting hyper parameters. The `--with-pose` will dump the sequence pose in the same format as Odometry dataset (see pose evaluation)
```bash
python3 data/prepare_train_data.py /path/to/raw/kitti/dataset/ --dataset-format 'kitti' --dump-root /path/to/resulting/formatted/data/ --width 416 --height 128 --num-threads 4 [--static-frames /path/to/static_frames.txt] [--with-depth] [--with-pose]
```

For [NYU](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html), first download the dataset using this [script](horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_raw.zip) provided on the official website, then follow the instruction below and corresponding file like process_raw.m is saved in data/nyudepth_preparation. 
```
How to process the training dataset:
1.) Extract the RAW dataset into a folder A (name not important)
2.) Download NYU Depth v2. toolbox from http://cs.nyu.edu/~silberman/code/toolbox_nyu_depth_v2.zip
3.) Extract scripts from the toolbox to folder 'tools' in folder A
4.) Run process_raw.m in folder A
5.) python nyud_raw_train_to_npy.py (modify the paths in that file to point to correct dirs, and also the resolution of training images can be modified here)

How to process the testing dataset:
1.) Download labeled NYU Depth v2. dataset from http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat
2.) Download splits.mat containing official train/test split http://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat
3.) Place all downloaded files into single folder
4.) Run script nyud_test_to_npy.py (modify the paths in that file to point to correct dirs)
```
<!-- For [Cityscapes](https://www.cityscapes-dataset.com/), download the following packages: 1) `leftImg8bit_sequence_trainvaltest.zip`, 2) `camera_trainvaltest.zip`. You will probably need to contact the administrators to be able to get it. Then run the following command
```bash
python3 data/prepare_train_data.py /path/to/cityscapes/dataset/ --dataset-format 'cityscapes' --dump-root /path/to/resulting/formatted/data/ --width 416 --height 171 --num-threads 4
```
Notice that for Cityscapes the `img_height` is set to 171 because we crop out the bottom part of the image that contains the car logo, and the resulting image will have height 128. -->

## Training
Once the data are formatted following the above instructions, you should be able to train the model by running the following command
```bash
python3 train.py /path/to/the/formatted/data/ -b4 -m0.2 -s0.1 --epoch-size 3000 --sequence-length 3 --log-output [--with-gt] --network disp_vgg_BN [--pretrained-enocoder] [--imagenet-normalization] --loss L1 --dataset nyu [--pretrained-disp /path/to/the/existing_weights/]
```
You can then start a `tensorboard` session in this folder by
```bash
tensorboard --logdir=checkpoints/
```
and visualize the training progress by opening [https://localhost:6006](https://localhost:6006) on your browser. If everything is set up properly, you should start seeing reasonable depth prediction after ~30K iterations when training on KITTI. As for the NYU Depth, image is not saved in tensorboard and it takes about over 40 epochs. This epoch means going through whole dataset, since our NYU Depth dataset consists of 67837 images and batchsize is 4, thus 17459 is the corresponding epochsize.

## Evaluation

Disparity map generation can be done with `run_inference.py`
```bash
python3 run_inference.py --pretrained /path/to/dispnet --dataset-dir /path/pictures/dir --output-dir /path/to/output/dir --network disp_vgg_BN [--imagenet-normalization]
```
Will run inference on all pictures inside `dataset-dir` and save a jpg of disparity (or depth) to `output-dir` for each one see script help (`-h`) for more options.

Disparity evaluation is avalaible
```bash
python3 test_disp.py --pretrained-dispnet /path/to/dispnet --pretrained-posenet /path/to/posenet --dataset-dir /path/to/KITTI_raw --dataset-list /path/to/test_files_list --network disp_vgg_BN [--imagenet-normalization]
```
eg. ```python3 test_disp.py --pretrained-dispnet checkpoints/kitti_sfm,epoch_size1000,seq5,s3.0,networkdisp_res,pretrained_encoderTrue,lossL1/02-21-10:43/dispnet_model_best.pth.tar --dataset-dir /scratch_net/kronos/zfang/dataset_kitti/kitti_original/kitti/ --dataset-list kitti_eval/test_files_eigen.txt --network disp_res --imagenet-normalization```

notice that imagenet-normalization is quite important if the encoder is pretrained on imagenet dataset.

Test file list is available in kitti eval folder. 

## Pretrained Nets

Currently not available
<!-- 
[Avalaible here](https://drive.google.com/drive/folders/1H1AFqSS8wr_YzwG2xWwAQHTfXN5Moxmx)
 -->
### KITTI Depth Results

|      specification      | Abs Rel | Sq Rel | RMSE  | RMSE(log) | Acc.1 | Acc.2 | Acc.3 |
|-------------------------|---------|--------|-------|-----------|-------|-------|-------|
| disp_vgg_BN with L1 loss| 0.105   | 0.723  | 4.537 | 0.186     | 0.873 | 0.959 | 0.983 | 

### NYU Depth Results

|      specification      | Abs Rel | Sq Rel | RMSE  | RMSE(log) | Acc.1 | Acc.2 | Acc.3 |
|-------------------------|---------|--------|-------|-----------|-------|-------|-------|
| disp_vgg_BN with L1 loss| 0.102   | 0.075  | 0.410 | 0.157     | 0.868 | 0.962 | 0.988 | 