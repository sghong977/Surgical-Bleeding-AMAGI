import argparse
import os
import os.path as osp
import json
import time
from tqdm import tqdm 

from PIL import Image
import numpy as np

import torch
import torchvision.datasets as datasets
from torchvision import transforms
import torch.utils.data as data

import mmcv
from mmaction.datasets.pipelines import Compose
from mmaction.models import build_model
# Multi GPU
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.parallel import collate

def parse_args():
    parser = argparse.ArgumentParser(description='workflow recognition')
    parser.add_argument("--local_rank", type=int, help="Local rank. Necessary for using the torch.distributed.launch utility.")
    parser.add_argument('--data_path', default='.', help='dataset prefix')
    parser.add_argument('--output_prefix', default='', help='output prefix')
    parser.add_argument('--task_type', default='active_bleeding', help='active_bleeding')
    parser.add_argument('--data_list', default=None, help='video list of the dataset, the format should be')
    parser.add_argument(
        '--frame_interval',
        type=int,
        default=30,
        help='the sampling frequency of frame in the untrimed video')
    parser.add_argument(
        '--temporal_stride',
        type=int,
        default=30,
        help='clip in frame interval')
    parser.add_argument('--ckpt', default='.pth', help='checkpoint for feature extraction')
    parser.add_argument(
        '--batch_size', type=int, default=256, help='input batch size')
    parser.add_argument(
        '--numModel', type=int, default=1, help='number of the model')
        
    parser.add_argument(
        '--config_file', type=str, default='ckpt/phase.py', help='config_file')
    parser.add_argument(
        '--num_class', type=int, default=2, help='active bleeding : 2')
    parser.add_argument(
        '--kfold', type=int, default=1, help='cross validation')
    parser.add_argument(
        '--multigpu', type=int, default=1, help='cross validation')
    args = parser.parse_args()
    return args


class mmaction_inference():
    def __init__(self, args):
        self.args = args
        self.config = mmcv.Config.fromfile(args.config_file)

        self.ckpt = args.ckpt
        self.num_class = args.num_class
        self.kfold = args.kfold
     
        self.data_path = args.data_path


        self.datalist = open(args.data_list).readlines()
        self.datalist = [x.strip() for x in self.datalist]
        
        # multi gpu
        self.device = torch.device("cuda:{}".format(args.local_rank) if torch.cuda.is_available() else "cpu")
        
        self.data_set()
        self.model_set()
    def data_set(self):
        # Data Setting
        self.img_norm_cfg = self.config['img_norm_cfg']
        self.img_norm_cfg['mean'] = [i / 255.0 for i in self.img_norm_cfg['mean']]
        self.img_norm_cfg['std'] = [i / 255.0 for i in self.img_norm_cfg['std']]
      
        self.clip_len = self.config['data']['test']['pipeline'][0]['clip_len'] # clip len

        self.transform = transforms.Compose([
                transforms.Scale(224),
                transforms.CenterCrop(224),
                # transforms.PILToTensor(),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.img_norm_cfg['mean'], std=self.img_norm_cfg['std']),
                # transforms.ToPILImage(),
                
        ])
    

        kfold = self.kfold
        if kfold == 1:
            test_patient = [3, 4, 6, 13, 17, 18, 22, 116, 208, 303]
        elif kfold == 2:
            test_patient = [1, 7, 10, 19, 56, 74, 100, 117, 203, 304]
        elif kfold == 3:
            test_patient = [5, 48, 76, 94, 202, 204, 206, 209, 301, 305]


    def model_set(self):
        # Model Setting
        model = build_model(self.config['model'])
        state_dict = torch.load(self.ckpt)['state_dict']
        model.load_state_dict(state_dict)
        # self.model = model.cuda()
        if self.args.multigpu > 1:
            self.model = MMDataParallel(
                model.cuda(0), device_ids=[self.args.local_rank], output_device=self.args.local_rank)
            self.model = self.model.to(self.device)
        else:
            self.model = model.cuda()
        
    
    def forward(self):
        prog_bar = mmcv.ProgressBar(len(self.datalist))
        probability = dict()
        for videoID in self.datalist:
            videoID = videoID.strip()
            frame_dir = os.path.join(self.data_path, videoID)
            output_dir = os.path.join(self.args.output_prefix, videoID)

            if not osp.exists(output_dir):
                os.system(f'mkdir -p {output_dir}')
            start = time.time()
   
            print('\nstart', videoID)
            inference_time_output_file = self.args.task_type + '_time.txt'
            inference_time_output_file = osp.join(output_dir, inference_time_output_file)
            output_file = self.args.task_type + '.json'
            output_file = osp.join(output_dir, output_file)
            
            # first frame 
            framelist = sorted(os.listdir(frame_dir))[::self.args.frame_interval]
            probability[videoID] = np.zeros((len(framelist), self.num_class))
            first_dataset = GastricDataset(data_path=frame_dir, datalist=framelist, temporal_stride=self.args.temporal_stride, windows=self.clip_len, transform=self.transform)
            # first_loader = torch.utils.data.DataLoader(first_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=8, pin_memory=True)
     
            # middle frame
            framelist = sorted(os.listdir(frame_dir))[::self.args.frame_interval]
            framelist = [framelist[0]] * ((self.clip_len - 1) // 2) + framelist[:-1* ((self.clip_len - 1) // 2)]
   
            middle_dataset = GastricDataset(data_path=frame_dir, datalist=framelist, temporal_stride=self.args.temporal_stride, windows=self.clip_len, transform=self.transform)
            # middle_loader = torch.utils.data.DataLoader(middle_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    
            # last frame
            framelist = sorted(os.listdir(frame_dir))[::self.args.frame_interval]
            framelist = [framelist[0]] * (self.clip_len - 1) + framelist[:-1* (self.clip_len - 1)]
            last_dataset = GastricDataset(data_path=frame_dir, datalist=framelist, temporal_stride=self.args.temporal_stride, windows=self.clip_len, transform=self.transform)
            # last_loader = torch.utils.data.DataLoader(last_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=8, pin_memory=True)
            
            dataset = ConcatDataset(first_dataset, middle_dataset, last_dataset)
            if self.args.multigpu > 1:
                rank, world_size = get_dist_info()
                sampler = DistributedSampler(
                    dataset, world_size, rank, shuffle=False, seed=None)
            else:
                sampler=None
            data_loader = torch.utils.data.DataLoader(dataset,sampler=sampler,
                batch_size=self.args.batch_size, shuffle=False, num_workers=6, pin_memory=True)
            
            with torch.no_grad():
                self.model.eval()
                idx = 0
                for batch_idx, (images1,images2,images3) in enumerate(tqdm(data_loader)):#zip(first_loader, middle_loader, last_loader), total=len(first_loader))):
                    # images1, images2, images3 = images1.cuda(), images2.cuda(), images3.cuda()
                    images = torch.cat((images1.unsqueeze(1), images2.unsqueeze(1), images3.unsqueeze(1)), dim=1).cuda()

                    prob = self.model(images, return_loss=False, infer_3d=True) #+ self.model(images2, return_loss=False, infer_3d=True) + self.model(images3, return_loss=False, infer_3d=True)
              
                    probability[videoID][idx:idx+images1.size(0)] = prob.squeeze().cpu().numpy()
                    idx += images1.size(0)
                results = self.save2json(probability[videoID])

            with open(output_file, "w") as json_file:
                json.dump(results, json_file)
            end = time.time()
            with open(inference_time_output_file, 'w') as f:
                f.write(str(end - start))
            prog_bar.update()

    def save2json(self, prob):
        result = dict()
        result['labels'] = {0 : 'Normal', 1: 'Actvie Bleeding'}
        result['prediction'] = dict()
        
        predictions = []
        for idx in range(len(prob)):
            predictions.append(str(np.argmax(prob[idx])))
        result['result'] = predictions
        probabilitys = []
        for idx in range(len(prob)):
            probabilitys.append(list(prob[idx]))
        result['prob'] = probabilitys
        result['frameInterval'] =  str(self.args.frame_interval) 
        return result

def main():
    args = parse_args()
    inference = mmaction_inference(args)
    inference.forward()

    # enumerate Untrimmed videos, extract feature from each of them
    
    
 
    
    
    
class ConcatDataset(data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)
    

class GastricDataset(data.Dataset):
    "Characterizes a dataset for PyTorch"
    def __init__(self, data_path, datalist, temporal_stride, windows, transform=None):
        "Initialization"
        self.data_path = data_path
        self.transform = transform
        self.temporal_stride = temporal_stride
        self.windows = windows
        self.datalist = datalist

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.datalist)
    

    def read_images(self, data):  
        X = []
        ts = self.temporal_stride
        frameNum = int(data[-10:-4])

        iamge = None 
        for i in range(0, self.windows):
           
            image_path = os.path.join(self.data_path, 'frame' + str((i*ts) \
                + frameNum).zfill(10) + '.jpg')
            if os.path.exists(image_path):
                image =Image.open(image_path)
                if self.transform is not None:
                    image = self.transform(image)

            X.append(image)
        X = torch.stack(X, dim=0)

        return X

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample

        data = self.datalist[index]

        # Load data
        X = self.read_images(data)     # (input) spatial images
        # BatchSize, Channel, Temporal, Width, Height
        X = X.permute(1, 0, 2, 3)
        return X
    

if __name__ == '__main__':
    main()
