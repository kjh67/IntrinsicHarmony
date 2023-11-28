"""Dataset class template

This module provides a template for users to implement custom datasets.
You can specify '--dataset_mode template' to use this dataset.
The class name should be consistent with both the filename and its dataset_mode option.
The filename should be <dataset_mode>_dataset.py
The class name should be <Dataset_mode>Dataset.py
You need to implement the following functions:
    -- <modify_commandline_options>:　Add dataset-specific options and rewrite default values for existing options.
    -- <__init__>: Initialize this dataset class.
    -- <__getitem__>: Return a data point and its metadata information.
    -- <__len__>: Return the number of images.
"""
import os.path
import torch
import torchvision.transforms.functional as tf
import torch.nn.functional as F
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from util import util
import glob
from torchvision.transforms import v2
import torch.nn as nn



class BlurEstimation(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_conv =  nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(inplace=True),
                nn.MaxPool2d(2)
            )

        self.conv1 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
            # nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),

                nn.LeakyReLU(inplace=True),
                nn.MaxPool2d(2)
        )

        self.conv2 = nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
            #  nn.Conv2d(256, 256, kernel_size=3, padding=1),
                            nn.BatchNorm2d(256),

                nn.LeakyReLU(inplace=True),
                nn.MaxPool2d(2)
        )

        self.conv3 = nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=3, padding=1),
                nn.LeakyReLU(inplace=True),
                nn.MaxPool2d(2)
        )

        self.conv4 = nn.Sequential(
                nn.Conv2d(512, 1024, kernel_size=3, padding=1),
                nn.LeakyReLU(inplace=True),
                nn.MaxPool2d(2)
        )

        self.fc = nn.Linear(65536,1)

    def forward(self, x):
        x = self.input_conv(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class CustomBlurDataset(BaseDataset):
    """A template dataset class for you to implement custom datasets."""
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--is_train', type=bool, default=True, help='whether in the training phase')
        parser.set_defaults(max_dataset_size=float("inf"), new_dataset_option=2.0)  # specify dataset-specific default values
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)
        self.image_paths = []
        self.isTrain = opt.isTrain
        self.image_size = opt.crop_size

        # initialise the blur prediction model
        self.blur_model = BlurEstimation()
        self.blur_model.load_state_dict(torch.load(opt.blur_path))
        self.blur_model.eval()
        
        # won't be used for training, only testing
        # if opt.isTrain==True:
        #     print('loading training file')
        #     self.trainfile = opt.dataset_root+'/train.txt'
        #     with open(self.trainfile,'r') as f:
        #             for line in f.readlines():
        #                 self.image_paths.append(os.path.join(opt.dataset_root,'composite_images',line.rstrip()))
        #     print(self.image_paths)
        print('loading test file')
        self.image_paths = glob.glob(opt.dataset_root+'/composite_images/*')
        # self.imnames = [fname.split('/')[-1] for fname in self.files]
        # for imname in self.imnames:
        #             self.image_paths.append(os.path.join(opt.dataset_root,'composite_images',imname.rstrip()))
        # get the image paths of your dataset;
          # You can call sorted(make_dataset(self.root, opt.max_dataset_size)) to get all the image paths under the directory self.root
        # define the default transform function. You can use <base_dataset.get_transform>; You can also define your custom transform function
        transform_list = [
            transforms.ToTensor(),
            transforms.Normalize((0, 0, 0), (1, 1, 1))
        ]
        self.transforms = transforms.Compose(transform_list)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.

        Step 1: get a random image path: e.g., path = self.image_paths[index]
        Step 2: load your data from the disk: e.g., image = Image.open(path).convert('RGB').
        Step 3: convert your data to a PyTorch tensor. You can use helpder functions such as self.transform. e.g., data = self.transform(image)
        Step 4: return a data point as a dictionary.
        """
        path = self.image_paths[index] # Gets a random path FROM ONE WE KNOW TO EXIST
        # get the files from the masks and real images folders with the same names
        #name_parts=path.split('_')
        mask_path = self.image_paths[index].replace('composite_images','masks')
        #mask_path = mask_path.replace(('_'+name_parts[-1]),'.png')
        target_path = self.image_paths[index].replace('composite_images','real_images')
        #target_path = target_path.replace(('_'+name_parts[-2]+'_'+name_parts[-1]),'.jpg')

        #print(f'image path: {path}')
        #print(f'mask path: {mask_path}')
        #print(f'target_path: {target_path}')

        comp = Image.open(path).convert('RGB').resize((256,256), Image.NEAREST)
        real = Image.open(target_path).convert('RGB').resize((256,256), Image.NEAREST)
        mask = Image.open(mask_path).convert('1').resize((256,256), Image.NEAREST)

        # if np.random.rand() > 0.5 and self.isTrain:
        #     comp, mask, real = tf.hflip(comp), tf.hflip(mask), tf.hflip(real)


        # MAKE BLUR ADJUSTMENT: need to get the blur of the background
        # then apply the same blur to the right region of the foreground
        with torch.no_grad():
            reshaped_image = np.float32((np.asarray(comp)).transpose((-1, 0, 1)))    
            reshaped_image = torch.tensor(reshaped_image).unsqueeze(0) / 255
            sigma = float(self.blur_model(reshaped_image).detach().numpy()[0][0])
            kernelsize = int(sigma*3)
            if not kernelsize%2: kernelsize+= 1
            print(f"Blur parameter: {sigma}")
            if sigma>0:
                adjustment = v2.GaussianBlur(kernel_size=(kernelsize,kernelsize), sigma=sigma)
                # only apply adjustment to the foreground, then put back into the background
                blurredcomp = adjustment(comp * np.tile(np.expand_dims(np.array(mask)/255, -1), (1,1,3))) \
                    + comp * (1 - np.tile(np.expand_dims(np.array(mask)/255, -1), (1,1,3)))
                # overwrite the raw composite image with the one with blurred foreground
                blurredcomp.save('/content/drive/MyDrive/IntrinsicHarmony/results/iih_base_lt_gd_allihd/test_latest/images/'+str(index)+'blurred.jpg')
                comp = Image.fromarray(np.uint8(blurredcomp), mode='RGB')

        if comp.size[0] != self.image_size:
            comp = tf.resize(comp, [self.image_size, self.image_size])
            mask = tf.resize(mask, [self.image_size, self.image_size])
            real = tf.resize(real, [self.image_size,self.image_size])
        
        comp = self.transforms(comp)
        mask = tf.to_tensor(mask)
        real = self.transforms(real)

        inputs=torch.cat([comp,mask],0)
        
        return {'inputs': inputs, 'comp': comp, 'real': real,'img_path':path,'mask':mask}

    def __len__(self):
        """Return the total number of images."""
        return len(self.image_paths)
