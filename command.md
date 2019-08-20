# apply for render
qrsh -l gpu -l h_vmem=40G -q "gpu.middle.q@*"

cd /scratch_net/biwidl204/zfang/SfmLearner-Pytorch-master

#train network
CUDA_VISIBLE_DEVICES=$SGE_GPU python3 train.py /scratch_net/biwidl204_second/zfang/kitti_sfm/ -b4 -m0.0 -s0.0 --epoch-size 3000 --sequence-length 3 --log-output --with-gt

#test 
python3 test_disp.py --pretrained-dispnet checkpoints/kitti_1024_320,5epochs,epoch_size11000,networkdisp_vgg_BN,pretrained_encoderTrue,lossL1/06-13-09:48/dispnet_checkpoint.pth.tar --dataset-dir /scratch_net/airfox_second/zfang/kitti_original/kitti/ --dataset-list kitti_eval/test_files_eigen.txt --network disp_vgg_BN --imagenet-normalization [--img-height 320] [--img-width 1024]

#finetune over monodepth2 model(here resolution 640 192)
python3 train.py /scratch_net/airfox_second/zfang/kitti_supervised/kitti_640_192/ -b12 -m0.0 -s0.0 --epoch-size 3500 --sequence-length 3 --log-output --with-gt --network disp_vgg_BN --pretrained-encoder --imagenet-normalization --loss L1 --epochs 5 --lr 1e-5 --pretrained-disp /scratch_net/airfox_second/zfang/checkpoints/06-29-20:28/MS_640x192/models/weights_1/ --monodepth2 --record

#test finetuned model from monodepth2
python3 test_disp.py --pretrained-dispnet checkpoints/kitti_1024_320,5epochs,epoch_size8500,lr1e-05,networkdisp_vgg_BN,pretrained_encoderTrue,lossL1/07-01-11:36/weights_4/dispnet_checkpoint.pth.tar --dataset-dir /scratch_net/airfox_second/zfang/kitti_original/kitti/ --dataset-list kitti_eval/test_files_eigen.txt --network disp_vgg_BN --imagenet-normalization --img-height 320 --img-width 1024 --monodepth2 [--error] [--pic] 

#check tensorboard for tracking performance
tensorboard --logdir=checkpoints/
#for the remote usage in tensorboard, first ssh -L 16006:127.0.0.1:6006 olivier@my_server_ip, then go to that link http://127.0.0.1:16006

#check disp image output and monodepth2 model can be implemented by corresponding args
python3 run_inference.py --pretrained checkpoints/kitti_416_128,epoch_size3000,networkdisp_vgg_BN,pretrained_encoderTrue,lossL1/06-12-23:29/dispnet_checkpoint.pth.tar --dataset-dir /scratch_net/airfox_second/zfang/kitti_original/kitti/ --output-dir ../result/ --network disp_vgg_BN --imagenet-normalization --output-disp --dataset-list kitti_eval/test_files_eigen.txt [--monodepth2]