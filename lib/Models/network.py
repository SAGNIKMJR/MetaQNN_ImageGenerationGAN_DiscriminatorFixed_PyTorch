import torch
import torch.nn as nn
import torch.nn.functional as F
import collections
from lib.Utility import FeatureOperations as FO

# TODO: bias = False

class WRNBasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropout=0.0, batchnorm=1e-3):
        super(WRNBasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        """ 
            NOTE: batch norm in commented lines
        """
        self.bn1 = nn.BatchNorm2d(in_planes, eps=batchnorm)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        """ 
            NOTE: batch norm in commented lines
        """
        self.bn2 = nn.BatchNorm2d(out_planes, eps=batchnorm)
        self.relu2 = nn.ReLU(inplace=True)

        self.droprate = dropout
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1,
                                                                stride=stride, padding=0, bias=False) or None

    def forward(self, x):
        """ 
            NOTE: batch norm in commented lines
        """
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
            # x = F.relu(x)

        else:
            out = self.relu1(self.bn1(x))
            # out = F.relu(x)
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        # out = self.relu2(self.conv1(out if self.equalInOut else x))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)

        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class net(nn.Module):

    def __init__(self, state_list, state_space_parameters, net_input, bn_val, do_drop):
        super(net, self).__init__()
        self.state_list = state_list
        self.state_space_parameters = state_space_parameters
        self.batch_size = net_input.size(0)
        self.num_colors = net_input.size(1)
        self.image_size = net_input.size(2)
        self.bn_value = bn_val
        self.do_drop = do_drop
        self.gpu_usage = 32 * self.batch_size * self.num_colors * self.image_size * self.image_size
        temp_defeature_list = []
        temp_declassifier_list = []  
        defeature_list = []
        declassifier_list = []  										#  as mentioned in state_string_utils.py
        convT_no = wrnT_bb_no = fc_no = relu_no = drop_no = bn_no = 0 
        feature = 1
        out_channel = self.num_colors
        no_feature = self.num_colors*((self.image_size)**2)
        last_image_size = self.image_size
        print('-' * 80)
        print('GAN')
        print('-' * 80)
        for state_no, state in enumerate(self.state_list):
            if state_no == len(self.state_list)-1:
                break
            if state.layer_type == 'fc':
                feature = 0
            if feature == 1:
                if state.layer_type == 'wrn':
                    in_channel = out_channel
                    out_channel = state.filter_depth
                    no_feature = ((state.image_size)**2)*(out_channel)
                    last_image_size = state.image_size
                    # TODO: fix padding, will work for stride = 1 only
                    temp_defeature_list.append((0, out_channel, in_channel, state.filter_size, 0))
                    self.gpu_usage += 32*(3*3*in_channel*out_channel + 3*3*out_channel*out_channel + int(in_channel!=out_channel)*in_channel*out_channel)
                    self.gpu_usage += 32*self.batch_size*state.image_size*state.image_size*state.filter_depth*(2 + int(in_channel!=out_channel))

                    # feature_list.append(('dropout' + str(wrn_bb_no), nn.Dropout2d(p = self.do_drop)))   
                elif state.layer_type == 'conv':
                    in_channel = out_channel
                    out_channel = state.filter_depth
                    no_feature = ((state.image_size)**2)*(out_channel)
                    # TODO: include option for 'SAME'
                    # TODO: fix padding, will work for stride = 1 only
                    if ((last_image_size - state.filter_size)/state.stride + 1)%2 != 0:
                        temp_defeature_list.append((1, out_channel, in_channel, state.filter_size,1)) 
                        last_image_size = (last_image_size - state.filter_size + 2)/state.stride + 1
                    else:
                        temp_defeature_list.append((1, out_channel, in_channel, state.filter_size,0)) 
                        last_image_size = (last_image_size - state.filter_size)/state.stride + 1          
                    self.gpu_usage += 32*(state.image_size * state.image_size * state.filter_depth * self.batch_size \
                                        + in_channel * out_channel * state.filter_size * state.filter_size) 
            else:
                if state.layer_type == 'fc':
                    in_feature = no_feature
                    no_feature = (state.fc_size)
                    temp_declassifier_list.append((no_feature, in_feature))
                    self.gpu_usage += 32 * (no_feature * self.batch_size + in_feature * no_feature)
        # NOTE: features for the latent space counted twice, thus subtracted once
        # self.gpu_usage -= 32 * (no_feature * self.batch_size + in_feature * no_feature)
        self.final_image_size = last_image_size
        self.fc_no = len(temp_declassifier_list)
        self.input_size = self.state_list[-1].fc_size
        for i in range(len(declassifier_list)):
            fc_no += 1
            index = len(temp_declassifier_list) -1 - i
            if i == 0:
                declassifier_list.append(('fc' + str(fc_no), nn.Linear(self.input_size, temp_declassifier_list[index][1])))
            else:
                declassifier_list.append(('fc' + str(fc_no), nn.Linear(temp_declassifier_list[index][0], temp_declassifier_list[index][1])))
            bn_no += 1
            declassifier_list.append(('batchNorm' + str(bn_no), nn.BatchNorm1d(temp_declassifier_list[index][1])))
            relu_no += 1
            if len(temp_defeature_list) == 0 and i==(len(temp_declassifier_list)-1):
                declassifier_list.append(('relu' + str(relu_no), nn.Sigmoid()))
            else:    
                declassifier_list.append(('relu' + str(relu_no), nn.ReLU(inplace = True)))
        if len(temp_declassifier_list) == 0:
            fc_no += 1
            declassifier_list.append(('fc' + str(fc_no), nn.Linear(self.input_size, out_channel*(last_image_size**2))))
            bn_no += 1
            declassifier_list.append(('batchNorm' + str(bn_no), nn.BatchNorm1d(out_channel*(last_image_size**2))))            
            relu_no += 1
            declassifier_list.append(('relu' + str(relu_no), nn.ReLU(inplace = True)))
        for i in range(len(temp_defeature_list) - 1):
            index = len(temp_defeature_list) - 1 - i 
            if temp_defeature_list[index][0] == 1: 
                convT_no += 1
                defeature_list.append(('convT' + str(convT_no), nn.ConvTranspose2d(temp_defeature_list[index][1], \
                                        temp_defeature_list[index][2], temp_defeature_list[index][3], stride = 2, padding = temp_defeature_list[index][4])))
                bn_no += 1
                defeature_list.append(('batchNorm' + str(bn_no), nn.BatchNorm2d(temp_defeature_list[index][2])))
                relu_no += 1
                defeature_list.append(('relu' + str(relu_no), nn.ReLU(inplace = True)))
            elif temp_defeature_list[index][0] == 0:
                wrnT_bb_no += 1
                defeature_list.append(('wrnT_bb_' + str(wrnT_bb_no), WRNBasicBlock(temp_defeature_list[index][1], \
                                        temp_defeature_list[index][2], stride = 1)))
        if len(temp_defeature_list)>0:
            convT_no += 1
            defeature_list.append(('convT' + str(convT_no), nn.ConvTranspose2d(temp_defeature_list[0][1], \
                                    temp_defeature_list[0][2], temp_defeature_list[0][3], stride = 2, padding = temp_defeature_list[0][4])))
            defeature_list.append(('sigmoid', nn.Sigmoid()))

        self.convT_no = convT_no
        self.wrnT_bb_no = wrnT_bb_no
        self.declassifiers_list = nn.Sequential(collections.OrderedDict(declassifier_list))
        self.defeatures_list = nn.Sequential(collections.OrderedDict(defeature_list))
        self.gpu_usage /= (8.*1024*1024*1024)

    def forward(self, z):
        if (self.convT_no>=1 or self.wrnT_bb_no>=1 or self.fc_no>=1): 
            x = self.declassifiers_list(z) 
            x = self.defeatures_list(x.view(x.size(0), -1, self.final_image_size, self.final_image_size))
            x = x.view(x.size(0),-1,self.image_size,self.image_size)
            return x

class discriminator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self, state_space_parameters, net_input, discriminator_classes):
        super(discriminator, self).__init__()
        self.state_space_parameters = state_space_parameters
        self.image_size = net_input.size(2)
        self.features_list = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride = 2, padding = 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride = 2, padding = 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.classifiers_list = nn.Sequential(
            nn.Linear(128 * (self.image_size // 4) * (self.image_size // 4), 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, discriminator_classes),
            nn.Sigmoid(),
        )
    def forward(self, x):
        x = self.features_list(x)
        x = x.view(-1, 128 * (self.image_size // 4) * (self.image_size // 4))
        x = self.classifiers_list(x)
        return x