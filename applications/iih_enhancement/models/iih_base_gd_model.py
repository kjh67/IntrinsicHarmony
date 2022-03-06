import torch
import os
import itertools
import torch.nn.functional as F
from util import distributed as du
# import pytorch_colors as colors
from .base_model import BaseModel
from util import util
from . import harmony_networks as networks
import util.ssim as ssim


class IIHBaseGDModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm='instance', netG='base_gd', dataset_mode='adobe5k')
        if is_train:
            parser.add_argument('--lambda_L1', type=float, default=50.0, help='weight for L1 loss')
            parser.add_argument('--lambda_R', type=float, default=100., help='weight for R gradient loss')
            parser.add_argument('--lambda_ssim', type=float, default=50., help='weight for L L2 loss')
            parser.add_argument('--lambda_ifm', type=float, default=100, help='weight for pm loss')
            
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.opt = opt
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G','G_L1','G_R','G_R_SSIM',"IF"]
            
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['enhanced','real','fake','reconstruct','illumination']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['G']
        self.opt.device = self.device
        self.netG = networks.define_G(opt.netG, opt.init_type, opt.init_gain, self.opt)
        self.cur_device = torch.cuda.current_device()
        self.ismaster = du.is_master_proc(opt.NUM_GPUS)
        print(self.netG)  
        
        if self.isTrain:
            # if self.ismaster == 0:
            util.saveprint(self.opt, 'netG', str(self.netG))  
            # define loss functions
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionL2 = torch.nn.MSELoss()
            self.criterionSSIM = ssim.SSIM()
            self.criterionDSSIM_CS = ssim.DSSIM(mode='c_s').to(self.device)
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        self.fake = input['fake'].to(self.device)
        self.real = input['real'].to(self.device)
        self.image_paths = input['img_path']
        self.real_r = F.interpolate(self.real, size=[32,32])
        self.real_gray = util.rgbtogray(self.real_r)
    def forward(self):
        self.reconstruct, self.enhanced, self.illumination, self.ifm_mean = self.netG(self.fake)
    def backward_G(self):
        self.loss_IF = self.criterionDSSIM_CS(self.ifm_mean, self.real_gray)*self.opt.lambda_ifm

        self.loss_G_L1 = self.criterionL1(self.reconstruct, self.fake)*self.opt.lambda_L1
        self.loss_G_R = self.criterionL2(self.enhanced, self.real)*self.opt.lambda_R
        self.loss_G_R_SSIM = (1-self.criterionSSIM(self.enhanced, self.real))*self.opt.lambda_ssim
        self.loss_G = self.loss_G_L1 + self.loss_G_R + self.loss_G_R_SSIM + self.loss_IF
        self.loss_G.backward()
        
    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights