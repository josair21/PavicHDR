#-*- coding:utf-8 -*-

import os.path as osp
import argparse
import time
import os
#from dataset.dataset_iccv23 import ICCV23_Test_Dataset
from dataset.dataset_sig17 import SIGGRAPH17_Test_Dataset, SIG17_Validation_Dataset
from torch.utils.data import DataLoader
from models.PavicHDR import PavicHDR

from train import test_single_img
from utils.utils import *

parser = argparse.ArgumentParser(description="Test Setting")
parser.add_argument("--dataset_dir", type=str, default='./our_data',
                        help='dataset directory')
parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
parser.add_argument('--test_batch_size', type=int, default=1, metavar='N',
                        help='testing batch size (default: 1)')
parser.add_argument('--num_workers', type=int, default=8, metavar='N',
                        help='number of workers to fetch data (default: 1)')
parser.add_argument('--patch_size', type=int, default=2000)
parser.add_argument('--ckpt', type=str, default='./ckpt_sctnet/')
parser.add_argument('--save_results', action='store_true', default=True)
parser.add_argument('--save_dir', type=str, default="./results_new/")
parser.add_argument('--model_arch', type=int, default=0)
times = []

def main():
    # Settings
    args = parser.parse_args()

    
    # pretrained_model
    print(">>>>>>>>> Start Testing >>>>>>>>>")
    print("Load weights from: ", args.ckpt)
    print(args.patch_size)

    # cuda and devices
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    print(device)

    model = PavicHDR().to(device)
    model = nn.DataParallel(model, device_ids = [0])
    model.load_state_dict(torch.load(f"{args.ckpt}")['state_dict'])
    model.eval()

    datasets = SIG17_Validation_Dataset(args.dataset_dir, crop = False)
    dataloader = DataLoader(
            dataset=datasets, batch_size=1, num_workers=1, shuffle=False
        )
    psnr_l = AverageMeter()
    ssim_l = AverageMeter()
    psnr_mu = AverageMeter()
    ssim_mu = AverageMeter()

    gpu_memory_list = []
    
    for i, images in enumerate(dataloader):
        start_memory = torch.cuda.memory_allocated() / (1024 ** 2)

        # Comenzar a medir tiempo
        img0_c = images["input0"].to(device)
        img1_c = images["input1"].to(device)
        img2_c = images["input2"].to(device)
        label = images['label'].to(device)
        start_time = time.time()
        with torch.no_grad():
            pred_img = model(img0_c, img1_c, img2_c)
        end_time = time.time()


        pred_hdr = pred_img
        # Medición del consumo de memoria
        end_memory = torch.cuda.memory_allocated() / (1024 ** 2)
        gpu_memory_list.append(end_memory - start_memory)

        # psnr-l and psnr-\muz

        pred_img = pred_img[0].cpu().numpy().astype(np.float64).transpose(1, 2, 0)
        label = label[0].cpu().numpy().astype(np.float64).transpose(1, 2, 0)

        scene_psnr_l = peak_signal_noise_ratio(label, pred_img, data_range=1.0)
        
        label_mu = range_compressor(label)
        pred_img_mu = range_compressor(pred_img)

        scene_psnr_mu = peak_signal_noise_ratio(label_mu, pred_img_mu, data_range=1.0)
        pred_img = np.clip(pred_img * 255.0, 0.0, 255.0)
        label = np.clip(label * 255.0, 0.0, 255.0)
        pred_img_mu = np.clip(pred_img_mu * 255.0, 0.0, 255.0)
        label_mu = np.clip(label_mu * 255.0, 0.0, 255.0)

        scene_ssim_l = calculate_ssim(pred_img, label)  # H W C data_range=0-255
        scene_ssim_mu = calculate_ssim(pred_img_mu, label_mu)

        print(
           f" {i} | "
           f"PSNR_mu: {scene_psnr_mu:.4f}  PSNR_l: {scene_psnr_l:.4f} | SSIM_mu: {scene_ssim_mu:.4f}  SSIM_l: {scene_ssim_l:.4f} Seconds: {end_time - start_time:.4f}  GPU: {end_memory - start_memory:.2f}"
        )

        psnr_l.update(scene_psnr_l)
        ssim_l.update(scene_ssim_l)
        psnr_mu.update(scene_psnr_mu)
        ssim_mu.update(scene_ssim_mu)
        times.append(end_time - start_time)

        
        # save results
        if args.save_results:
            args.save_dir = f"{args.ckpt[:-4]}"
            if not osp.exists(args.save_dir):
                os.makedirs(args.save_dir)
                
            pred_hdr = pred_hdr[0].cpu().permute(1, 2, 0).numpy()
            cv2.imwrite(os.path.join(args.save_dir, '{}_pred.png'.format(i)), pred_img_mu)
            cv2.imwrite(os.path.join(args.save_dir, '{}_pred.hdr'.format(i)), pred_hdr[:, :, ::-1])
            cv2.imwrite(os.path.join(args.save_dir, '{}_gt.png'.format(i)), label_mu)
            
        del pred_img, img0_c, img1_c, img2_c, label
        torch.cuda.empty_cache()
            
    print("Average PSNR_mu: {:.4f}  PSNR_l: {:.4f}".format(psnr_mu.avg, psnr_l.avg))
    print("Average SSIM_mu: {:.4f}  SSIM_l: {:.4f}".format(ssim_mu.avg, ssim_l.avg))
    print(f"Average time {np.mean(times)}")
    print(f"Average GPU Mem {np.array(gpu_memory_list).mean()}")
    print(">>>>>>>>> Finish Testing >>>>>>>>>")

        


if __name__ == '__main__':
    main()
