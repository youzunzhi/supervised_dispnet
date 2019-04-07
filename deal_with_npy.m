pred = readNPY('output/npy_result/disp_vgg/s0.5.npy');
%pred_zoomed =  readNPY('pred_zoomed.npy');
gt_depth = readNPY('gt_depth.npy');
%mask = readNPY('mask.npy');

pred_single = pred(1,:,:);
%pred_zoomed_single = pred_zoomed(1,:,:);
gt_single = gt_depth(1,:,:);
%mask_single = mask(1,:,:);

%plot
[X,Y] = meshgrid(1:416,1:128);surf(X,Y,squeeze(pred_single));