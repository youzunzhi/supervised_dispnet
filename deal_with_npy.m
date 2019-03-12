pred = readNPY('output/npy_result/dispnet/L1.npy');
pred_zoomed =  readNPY('pred_zoomed.npy');
gt_depth = readNPY('gt_depth.npy');
mask = readNPY('mask.npy');

pred_single = pred(1,:,:);
pred_zoomed_single = pred_zoomed(1,:,:);
gt_single = gt_depth(1,:,:);
mask_single = mask(1,:,:);