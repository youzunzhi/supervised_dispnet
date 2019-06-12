import torch
import torchvision.transforms
from scipy.misc import imresize
from scipy.ndimage.interpolation import zoom
import numpy as np
from path import Path
import argparse
from tqdm import tqdm
import pdb
#from models import DispNetS, Disp_res, Disp_vgg, Disp_vgg_feature, Disp_vgg_BN, FCRN, deeplab_depth, PoseExpNet
import models
# for depth ground truth
from imageio import imsave
from utils import tensor2array, get_depth_sid

#from datasets.nyu_depth_v2 import NYU_Depth_V2, transform_chw

parser = argparse.ArgumentParser(description='Script for DispNet testing with corresponding groundTruth',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--network", required=True, type=str, help="network type")
parser.add_argument('--imagenet-normalization', action='store_true', help='use imagenet parameter for normalization.')
parser.add_argument('--ordinal-c', default=80, type=int, metavar='N', help='DORN loss channel number')
parser.add_argument("--unsupervised", action='store_true', help="to have unsupervised loss")
#parser.add_argument("--dataset", default='kitti', type=str, help="dataset name")


parser.add_argument("--pretrained-dispnet", required=True, type=str, help="pretrained DispNet path")
parser.add_argument("--pretrained-posenet", default=None, type=str, help="pretrained PoseNet path (for scale factor)")

# width and height is set according to dataset and network (may be I will use other resolution for some network, currently based on dataset)
# parser.add_argument("--img-height", default=128, type=int, help="Image height")
# parser.add_argument("--img-width", default=416, type=int, help="Image width")
parser.add_argument("--no-resize", action='store_true', help="no resizing is done")
# parser.add_argument("--min-depth", default=1e-3)
# parser.add_argument("--max-depth", default=80)

parser.add_argument("--dataset-dir", default='.', type=str, help="Dataset directory")
parser.add_argument("--dataset-list", default=None, type=str, help="Dataset list file")
parser.add_argument("--output-dir", default=None, type=str, help="Output directory for saving predictions in a big 3D numpy file")

parser.add_argument("--gt-type", default='KITTI', type=str, help="GroundTruth data type", choices=['npy', 'png', 'KITTI', 'NYU','stillbox'])
parser.add_argument("--img-exts", default=['png', 'jpg', 'bmp'], nargs='*', type=str, help="images extensions to glob")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


@torch.no_grad()
def main():
    args = parser.parse_args()
    if args.gt_type == 'KITTI':
        from kitti_eval.depth_evaluation_utils import test_framework_KITTI as test_framework
    elif args.gt_type == 'NYU':
        from kitti_eval.depth_evaluation_utils import test_framework_NYU as test_framework
    elif args.gt_type == 'stillbox':
        from stillbox_eval.depth_evaluation_utils import test_framework_stillbox as test_framework

    #choose corresponding net type
    if args.network=='dispnet':
    	disp_net = models.DispNetS().to(device)
    elif args.network=='disp_res':
    	disp_net = models.Disp_res().to(device)
    elif args.network=='disp_res_50':
        disp_net = models.Disp_res_50().to(device)
    elif args.network=='disp_vgg':
    	disp_net = models.Disp_vgg_feature().to(device)
    elif args.network=='disp_vgg_BN':
        disp_net = models.Disp_vgg_BN().to(device)
    elif args.network=='FCRN':
        disp_net = models.FCRN().to(device)
    elif args.network=='res50_aspp':
        disp_net = models.res50_aspp().to(device)  
    elif args.network=='ASPP':
        disp_net = models.deeplab_depth().to(device)
    elif args.network=='disp_res_101':
        disp_net = models.Disp_res_101().to(device)  
    elif args.network=='DORN':
        disp_net = models.DORN().to(device)
    elif args.network=='disp_vgg_BN_DORN':
        disp_net = models.Disp_vgg_BN_DORN(ordinal_c=args.ordinal_c).to(device)
    else:
    	raise "undefined network"

    weights = torch.load(args.pretrained_dispnet)
    disp_net.load_state_dict(weights['state_dict'])
    disp_net.eval()

    if args.pretrained_posenet is None:
        print('no PoseNet specified, scale_factor will be determined by median ratio, which is kiiinda cheating\
            (but consistent with original paper)')
        seq_length = 0
    else:
        weights = torch.load(args.pretrained_posenet)
        seq_length = int(weights['state_dict']['conv1.0.weight'].size(1)/3)
        pose_net = PoseExpNet(nb_ref_imgs=seq_length - 1, output_exp=False).to(device)
        pose_net.load_state_dict(weights['state_dict'], strict=False)

    dataset_dir = Path(args.dataset_dir)

    if args.gt_type == 'KITTI':
        img_height = 128
        img_width = 416
        min_depth = 1e-3
        max_depth = 80

        if args.dataset_list is not None:
            with open(args.dataset_list, 'r') as f:
                test_files = list(f.read().splitlines())
        else:
            test_files = [file.relpathto(dataset_dir) for file in sum([dataset_dir.files('*.{}'.format(ext)) for ext in args.img_exts], [])]
        
        framework = test_framework(dataset_dir, test_files, seq_length, min_depth=1e-3, max_depth=80)
        test_length = len(test_files)

    elif args.gt_type == 'NYU':# this dataset_dir will add the 'nyu_depth_v2\labeled\npy' at the end
        framework = test_framework(dataset_dir, min_depth=1e-3, max_depth=10)
        # img_height = 480
        # img_width = 640
        img_height = 256
        img_width = 352
        min_depth = 1e-3
        max_depth = 10
        
        test_length = len(framework)
    else:
        raise 'undefined dataset'

    print('{} files to test'.format(test_length))
    errors = np.zeros((2, 7, test_length), np.float32)
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        output_dir.makedirs_p()

    #save gt_avg for run inference
    gt_scale = np.zeros(len(framework))

    for j, sample in enumerate(tqdm(framework)):
        
        #gt depth shape (1, 480, 640)
        #gt_depth = sample['gt_depth']#;pdb.set_trace()

        tgt_img = sample['tgt']#;pdb.set_trace()

        #ref_imgs = sample['ref']
        
        # #*************************************
        # nyu data setting is different, this nyu depth has color channel as the first channel
        if args.gt_type == 'NYU':
            tgt_img = np.transpose(tgt_img, (1, 2, 0))

        h,w,_ = tgt_img.shape
        if (not args.no_resize) and (h != img_height or w != img_width):
            tgt_img = imresize(tgt_img, (img_height, img_width)).astype(np.float32)
            #ref_imgs = [imresize(img, (img_height, img_width)).astype(np.float32) for img in ref_imgs]

        tgt_img = np.transpose(tgt_img, (2, 0, 1))
        #ref_imgs = [np.transpose(img, (2,0,1)) for img in ref_imgs]

        tgt_img = torch.from_numpy(tgt_img)#.unsqueeze(0)
       
        #for different normalize method
        if args.imagenet_normalization:
        	normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        else:
        	normalize = torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        
        if args.gt_type == 'NYU':
            tgt_img = normalize(tgt_img).unsqueeze(0).to(device)#;pdb.set_trace()
        elif args.gt_type == 'KITTI':
            tgt_img = normalize(tgt_img/255).unsqueeze(0).to(device)#;pdb.set_trace()

        # #*************************************
        # tgt_img = np.transpose(tgt_img, (1, 2, 0))
        # tgt_img = imresize(tgt_img, (img_height, img_width)).astype(np.float32)
        # tgt_img = np.transpose(tgt_img, (2, 0, 1));pdb.set_trace()

        # transform=NYU_Depth_V2.get_transform(training=False)
        # image = transform_chw(transform, tgt_img)
        # tgt_img = tgt_img.to(device)
        # gt_depth = gt_depth.to(device)
        # gt_depth = torch.squeeze(gt_depth[0,:,:])
        # #*************************************

        # that part is for the unsupervised setting 

        # for i, img in enumerate(ref_imgs):
        #     img = torch.from_numpy(img).unsqueeze(0)
        #     img = ((img/255 - 0.5)/0.5).to(device)
        #     ref_imgs[i] = img
        
        if args.network=='DORN' or args.network == 'disp_vgg_BN_DORN':
            pred_d, pred_ord = disp_net(tgt_img)
            pred_depth = torch.squeeze(get_depth_sid(pred_d, ordinal_c=args.ordinal_c, dataset=args.gt_type)).cpu().numpy()#;pdb.set_trace()
            pred_disp = 1/pred_depth
        else:
            pred_disp = disp_net(tgt_img).cpu().numpy()[0,0]

        if args.output_dir is not None:
            if j == 0:
                predictions = np.zeros((test_length, *pred_disp.shape))
            if args.network=='DORN' or args.network=='disp_vgg_BN_DORN':
                predictions[j] = pred_d
            else:
                predictions[j] = 1/pred_disp
                pred_depth = predictions[j]

        ##gt depth shape (1, 480, 640)
        gt_depth = sample['gt_depth']#;pdb.set_trace()

        #correspond with framework code setting about depth getitem
        if args.gt_type == 'NYU':
            gt_depth = gt_depth[0,:,:] 

        pred_depth = 1/pred_disp#;pdb.set_trace()

        pred_depth_zoomed = zoom(pred_depth,
                                 (gt_depth.shape[0]/pred_depth.shape[0],
                                  gt_depth.shape[1]/pred_depth.shape[1])
                                 ).clip(min_depth, max_depth)

        # #ground truth depth production
        # tensor_depth = torch.from_numpy(gt_depth).to(device)
        # #tensor_depth = tensor_depth.unsqueeze(1)
        # tensor_depth[tensor_depth == 0] = 1000
        # disp_to_show = (1/tensor_depth).clamp(0,10)
        
        #;pdb.set_trace()
        #print(disp_to_show.size())
        #disp_to_show = np.clip(1/tensor_depth, 0, 10)

#********************************************        
        # x=(disp_to_show > 0.002)
        # x_delete_edge=x[1:374,1:1241]
        # disp_to_show[0:373,0:1240]=disp_to_show[x_delete_edge]
        # disp_to_show[0:373,1:1241]=disp_to_show[x_delete_edge]
        # disp_to_show[0:373,2:1242]=disp_to_show[x_delete_edge]
        # disp_to_show[1:374,0:1240]=disp_to_show[x_delete_edge]
        # #disp_to_show[1:374,1:1241]=disp_to_show[x_delete_edge]
        # disp_to_show[1:374,2:1242]=disp_to_show[x_delete_edge]
        # disp_to_show[2:375,0:1240]=disp_to_show[x_delete_edge]
        # disp_to_show[2:375,1:1241]=disp_to_show[x_delete_edge]
        # disp_to_show[2:375,2:1242]=disp_to_show[x_delete_edge]
        # #wait for add in
#*************************************************

        # disp = (255*tensor2array(disp_to_show, max_value=None, colormap='bone',channel_first=False)).astype(np.uint8)
        # imsave(Path('groundtruth')/'{}_disp.png'.format(j), disp)

        #this kitti mask process generate a mask that has a crop procedure

        # if sample['mask'] is not None:
        #     pred_depth_zoomed = pred_depth_zoomed[sample['mask']]
        #     gt_depth = gt_depth[sample['mask']]

        if args.gt_type == 'KITTI':
            if sample['mask'] is not None:
                pred_depth_zoomed = pred_depth_zoomed[sample['mask']]
                gt_depth = gt_depth[sample['mask']]
        elif args.gt_type == 'NYU':
            valid = (gt_depth > min_depth) & (gt_depth < max_depth)
            pred_depth_zoomed = pred_depth_zoomed[valid]
            gt_depth = gt_depth[valid]

        if seq_length > 0:
            # Reorganize ref_imgs : tgt is middle frame but not necessarily the one used in DispNetS
            # (in case sample to test was in end or beginning of the image sequence)
            middle_index = seq_length//2
            tgt = ref_imgs[middle_index]
            reorganized_refs = ref_imgs[:middle_index] + ref_imgs[middle_index + 1:]
            _, poses = pose_net(tgt, reorganized_refs)
            mean_displacement_magnitude = poses[0,:,:3].norm(2,1).mean().item()

            scale_factor = sample['displacement'] / mean_displacement_magnitude
            errors[0,:,j] = compute_errors(gt_depth, pred_depth_zoomed*scale_factor)
        if args.unsupervised:
            scale_factor = np.median(gt_depth)/np.median(pred_depth_zoomed)
        else:
            scale_factor = 1#;pdb.set_trace()
        #gt_scale[j] = np.median(gt_depth)
        errors[1,:,j] = compute_errors(gt_depth, pred_depth_zoomed*scale_factor)#;pdb.set_trace()

        # #ground truth depth production
        # tensor_depth = torch.from_numpy(gt_depth).to(device)
        # tensor_depth = tensor_depth.unsqueeze(1)
        # tensor_depth[tensor_depth == 0] = 1000
        # disp_to_show = (1/tensor_depth).clamp(0,10);pdb.set_trace()
        # #print(disp_to_show.size())
        # #disp_to_show = np.clip(1/tensor_depth, 0, 10)
        # disp = (255*tensor2array(disp_to_show, max_value=None, colormap='bone',channel_first=False)).astype(np.uint8)
        # imsave(Path('groundtruth')/'{}_disp.png'.format(j), disp)

    mean_errors = errors.mean(2)
    error_names = ['abs_rel','sq_rel','rms','log_rms','a1','a2','a3']
    if args.pretrained_posenet:
        print("Results with scale factor determined by PoseNet : ")
        print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format(*error_names))
        print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(*mean_errors[0]))

    print("Results with scale factor determined by GT/prediction ratio (like the original paper) : ")
    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format(*error_names))
    print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(*mean_errors[1]))

    if args.output_dir is not None:
        np.save(output_dir/'predictions.npy', predictions)
    
    #save to check whether avg is needed in the prediction 
    output_file = Path('errors_test')
    np.save(output_file, errors[1,:,:])


#interpolate ground truth map
def lin_interp(shape, xyd):
    # taken from https://github.com/hunse/kitti
    m, n = shape
    ij, d = xyd[:, 1::-1], xyd[:, 2]
    f = LinearNDInterpolator(ij, d, fill_value=0)
    J, I = np.meshgrid(np.arange(n), np.arange(m))
    IJ = np.vstack([I.flatten(), J.flatten()]).T
    disparity = f(IJ).reshape(shape)
    return disparity

def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25   ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred)**2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


if __name__ == '__main__':
    main()
