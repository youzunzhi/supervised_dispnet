import torch
import torchvision.transforms
from imageio import imread, imsave
from scipy.misc import imresize
import numpy as np
from path import Path
import argparse
from tqdm import tqdm

#from models import DispNetS
import models
from utils import tensor2array

parser = argparse.ArgumentParser(description='Inference script for DispNet learned with \
                                 Structure from Motion Learner inference on KITTI and CityScapes Dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--network", default='disp_vgg', type=str, help="network type")
parser.add_argument('--imagenet-normalization', action='store_true', help='use imagenet parameter for normalization.')

parser.add_argument("--output-disp", action='store_true', help="save disparity img")
parser.add_argument("--output-depth", action='store_true', help="save depth img")
parser.add_argument("--pretrained", required=True, type=str, help="pretrained DispNet path")
parser.add_argument("--img-height", default=128, type=int, help="Image height")
parser.add_argument("--img-width", default=416, type=int, help="Image width")
parser.add_argument("--no-resize", action='store_true', help="no resizing is done")

parser.add_argument("--dataset-list", default=None, type=str, help="Dataset list file")
parser.add_argument("--dataset-dir", default='.', type=str, help="Dataset directory")
parser.add_argument("--output-dir", default='output', type=str, help="Output directory")

parser.add_argument("--img-exts", default=['png', 'jpg', 'bmp'], nargs='*', type=str, help="images extensions to glob")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


@torch.no_grad()
def main():
    args = parser.parse_args()
    if not(args.output_disp or args.output_depth):
        print('You must at least output one value !')
        return
    
    if args.network=='dispnet':
        disp_net = models.DispNetS().to(device)
    elif args.network=='disp_res':
        disp_net = models.Disp_res().to(device)
    elif args.network=='disp_vgg':
        disp_net = models.Disp_vgg_feature().to(device)
    elif args.network=='disp_vgg_BN':
        disp_net = models.Disp_vgg_BN().to(device)    
    else:
        raise "undefined network"

    weights = torch.load(args.pretrained)
    disp_net.load_state_dict(weights['state_dict'])
    disp_net.eval()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.makedirs_p()

    if args.dataset_list is not None:
        with open(args.dataset_list, 'r') as f:
            test_files = [dataset_dir/file for file in f.read().splitlines()]
    else:
        test_files = sum([dataset_dir.files('*.{}'.format(ext)) for ext in args.img_exts], [])

    print('{} files to test'.format(len(test_files)))

    for file in tqdm(test_files):

        img = imread(file).astype(np.float32)

        h,w,_ = img.shape
        if (not args.no_resize) and (h != args.img_height or w != args.img_width):
            img = imresize(img, (args.img_height, args.img_width)).astype(np.float32)
        img = np.transpose(img, (2, 0, 1))

        #for different normalize method
        if args.imagenet_normalization:
            normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        else:
            normalize = torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
 
        tensor_img = torch.from_numpy(img)#.unsqueeze(0)
        # tensor_img = ((tensor_img/255 - 0.5)/0.2).to(device)% why it is 0.2
        tensor_img = normalize(tensor_img/255).unsqueeze(0).to(device)# consider multiply by 2.5 to compensate

        output = disp_net(tensor_img)[0]

        if args.output_disp:
            disp = (255*tensor2array(output, max_value=None, colormap='bone', channel_first=False)).astype(np.uint8)
            imsave(output_dir/'{}_disp{}'.format(file.namebase,file.ext), disp)
        if args.output_depth:
            depth = 1/output
            depth = (255*tensor2array(depth, max_value=10, colormap='rainbow', channel_first=False)).astype(np.uint8)
            imsave(output_dir/'{}_depth{}'.format(file.namebase,file.ext), depth)


if __name__ == '__main__':
    main()
