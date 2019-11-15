import os
import sys
import teca_py
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time


# Implementation of Google's Deeplab-V3-Plus
# source: https://arxiv.org/pdf/1802.02611.pdf
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, rate=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               dilation=rate, padding=rate, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.rate = rate

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, nInputChannels, block, layers, os=16, pretrained=False, state_dict_resnet=None):
        self.inplanes = 64
        super(ResNet, self).__init__()
        if os == 16:
            strides = [1, 2, 2, 1]
            rates = [1, 1, 1, 2]
            blocks = [1, 2, 4]
        elif os == 8:
            strides = [1, 2, 1, 1]
            rates = [1, 1, 2, 2]
            blocks = [1, 2, 1]
        else:
            raise NotImplementedError

        self.state_dict_resnet = state_dict_resnet

        # Modules
        self.conv1 = nn.Conv2d(nInputChannels, 64, kernel_size=7, stride=2, padding=3,
                                bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], rate=rates[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], rate=rates[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], rate=rates[2])
        self.layer4 = self._make_MG_unit(block, 512, blocks=blocks, stride=strides[3], rate=rates[3])

        self._init_weight()

        if pretrained:
            self._load_pretrained_model()

    def _make_layer(self, block, planes, blocks, stride=1, rate=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, rate, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_MG_unit(self, block, planes, blocks=[1,2,4], stride=1, rate=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, rate=blocks[0]*rate, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(block(self.inplanes, planes, stride=1, rate=blocks[i]*rate))

        return nn.Sequential(*layers)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        low_level_feat = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x, low_level_feat

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _load_pretrained_model(self):
        #resnet_sd_path = os.path.join(teca_py.get_teca_data_root(),
        #    "resnet101-5d3b4d8f-state_dict.pth")

        #state_dict = teca_py.teca_load_state_dict(self.comm, resnet_sd_path)
        #self.load_state_dict(state_dict)
        self.load_state_dict(self.state_dict_resnet)


def ResNet101(nInputChannels=3, os=16, pretrained=False, state_dict_resnet=None):
    model = ResNet(nInputChannels, Bottleneck, [3, 4, 23, 3],
        os, pretrained=pretrained, state_dict_resnet=state_dict_resnet)
    return model
    

class ASPP_module(nn.Module):
    def __init__(self, inplanes, planes, rate):
        super(ASPP_module, self).__init__()
        if rate == 1:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 3
            padding = rate
        self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=rate, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
class DeepLabv3_plus(nn.Module):
    def __init__(self, nInputChannels=3, n_classes=21, os=16,
                pretrained=False, state_dict_resnet=None, _print=True):
        if _print:
            print("Constructing DeepLabv3+ model...")
            print("Number of classes: {}".format(n_classes))
            print("Output stride: {}".format(os))
            print("Number of Input Channels: {}".format(nInputChannels))
        super(DeepLabv3_plus, self).__init__()

        # Atrous Conv
        self.resnet_features = ResNet101(nInputChannels, os, pretrained=pretrained, state_dict_resnet=state_dict_resnet)

        # ASPP
        if os == 16:
            rates = [1, 6, 12, 18]
        elif os == 8:
            rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = ASPP_module(2048, 256, rate=rates[0])
        self.aspp2 = ASPP_module(2048, 256, rate=rates[1])
        self.aspp3 = ASPP_module(2048, 256, rate=rates[2])
        self.aspp4 = ASPP_module(2048, 256, rate=rates[3])

        self.relu = nn.ReLU()

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(2048, 256, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(256),
                                             nn.ReLU())

        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)

        # adopt [1x1, 48] for channel reduction.
        self.conv2 = nn.Conv2d(256, 48, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(48)

        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Conv2d(256, n_classes, kernel_size=1, stride=1))

    def forward(self, input):
        x, low_level_features = self.resnet_features(input)
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)

        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = F.interpolate(x, size=(int(math.ceil(input.size()[-2]/4)),
                                int(math.ceil(input.size()[-1]/4))), mode='bilinear', align_corners=True)

        low_level_features = self.conv2(low_level_features)
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)


        x = torch.cat((x, low_level_features), dim=1)
        x = self.last_conv(x)

        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

# End of DeepLabv3_plus implementation


class teca_deeplabv3p_ar_detect(teca_py.teca_model_segmentation):
    """
    This algorithm detects Atmospheric Rivers using the DeepLabv3+
    architecture
    """
    def data_preprocess(self, input_data, lat, lon):
        '''
        Necessary data preprocessing before inputing it into
        the model. The preprocessing is padding the data and 
        converting it into 3 channels
        '''
        input_data = np.reshape(input_data, [1, lat, lon])

        target_shape = 128 * np.ceil(lat/128.0)
        target_shape_diff = target_shape - lat
        padding_amount_lat = (int(np.ceil(target_shape_diff / 2.0)),
            int(np.floor(target_shape_diff / 2.0)))
        input_data = np.pad(input_data, ((0,0), padding_amount_lat, (0,0)),
            'constant', constant_values=0)

        target_shape = 64 * np.ceil(lon/64.0)
        target_shape_diff = target_shape - lon
        padding_amount = (int(np.ceil(target_shape_diff / 2.0)),
            int(np.floor(target_shape_diff / 2.0)))
        input_data = np.pad(input_data, ((0,0), (0,0), padding_amount),
            'constant', constant_values=0)

        input_data = input_data.astype('float32')

        transformed_input_data = np.zeros(
            (1, 3, input_data.shape[1], input_data.shape[2])
        ).astype('float32')
        for i in range(3):
            transformed_input_data[0, i,...] = input_data

        #transformed_input_data = np.reshape(transformed_input_data
        #    , [1, len(lat), len(lon)])

        return transformed_input_data


    def build_model(self, state_dict_deeplab_file=None, state_dict_resnet_file=None):
        '''
        Load model from file system. If multi-threading is used rank 0
        loads the model file and broadcasts it to the other ranks
        '''
        if not state_dict_deeplab_file:
            deeplab_sd_path = \
                "pre_trained_deeplab_resnet_IVT_artmip_probabilistic_adam_epoch_001_loss_perceptual_loss_160987.8365_time_corr_split_state_dict_final.pt"
            state_dict_deeplab_file = os.path.join(
                teca_py.get_teca_data_root(),
                deeplab_sd_path
            )

        if not state_dict_resnet_file:
            resnet_sd_path = \
                "resnet101-5d3b4d8f-state_dict.pth"
            state_dict_resnet_file = os.path.join(
                teca_py.get_teca_data_root(),
                resnet_sd_path
            )

        comm = self.get_communicator()

        state_dict_deeplab = self.load_state_dict(state_dict_deeplab_file)
        state_dict_resnet = self.load_state_dict(state_dict_resnet_file)
        
        model = DeepLabv3_plus(n_classes=1, pretrained=True, _print=False, state_dict_resnet=state_dict_resnet)
        model.load_state_dict(state_dict_deeplab)

        self.set_model(model)


    def get_execute_callback(self):
        """
        return a teca_algorithm::execute function
        """
        def execute(port, data_in, req):
            """
            expects an array of an input variable to run through
            the torch model and get the segmentation results as an
            output.
            """
            in_mesh = teca_py.as_teca_cartesian_mesh(data_in[0])

            if in_mesh is None:
                raise ValueError("ERROR: empty input, or not a mesh")
            if self.model is None:
                raise ValueError("ERROR: pretrained model has not been specified")

            md = in_mesh.get_metadata()
            ext = md["extent"]

            lat = int(ext[3]-ext[2]+1)
            lon = int(ext[1]-ext[0]+1)

            arrays = in_mesh.get_point_arrays()

            var_array = arrays[self.variable_name]

            start_time = time.time()
            self.var_array = self.data_preprocess(var_array, lat, lon)
            end_time = time.time()

            super_execute = super(teca_deeplabv3p_ar_detect, self).get_execute_callback()

            out_mesh = super_execute(port, data_in, req)

            return out_mesh
            '''
            var_array = torch.from_numpy(var_array).to(self.device)

            with torch.no_grad():
                pred = torch.sigmoid(self.model(var_array))

            if pred is None:
                if rank == 0:
                    raise Exception("ERROR: Model failed to get predictions")

            out_mesh = teca_py.teca_cartesian_mesh.New()
            out_mesh.shallow_copy(in_mesh)

            pred = teca_py.teca_variant_array.New(pred.numpy().ravel())
            out_mesh.get_point_arrays().set(self.pred_name, pred)

            return out_mesh
            '''
        return execute
