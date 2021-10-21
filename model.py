import math

import torch
import torchvision
from torch import nn
from torch.nn import functional as F

#from pytorch_metric_learning import losses, reducers
#from pytorch_metric_learning.distances import CosineSimilarity



class Backbone(nn.Module):
    def __init__(self, backbone, embedding_size, bn_mom = 0.05):
        super().__init__()
        
        '''backbone for the architecture. 
            Supported backbones: ResNets, ResNeXts, DenseNets (from torchvision), EfficientNets.'''
        if backbone.startswith('densenet'):
            channels = 96 if backbone == 'densenet161' else 64
            first_conv = nn.Conv2d(3, channels, 7, 2, 3, bias=False) # changed to 4
            pretrained_backbone = getattr(torchvision.models, backbone)(pretrained=True)
            self.features = pretrained_backbone.features
            self.features.conv0 = first_conv
            self.features_num = pretrained_backbone.classifier.in_features
        elif backbone.startswith('resnet') or backbone.startswith('resnext'):
            first_conv = nn.Conv2d(3, 64, 7, 2, 3, bias=False) # changed to 3
            pretrained_backbone = getattr(torchvision.models, backbone)(pretrained=True)
            self.features = nn.Sequential(
                first_conv,
                pretrained_backbone.bn1,
                pretrained_backbone.relu,
                pretrained_backbone.maxpool,
                pretrained_backbone.layer1,
                pretrained_backbone.layer2,
                pretrained_backbone.layer3,
                pretrained_backbone.layer4,
            )
            self.features_num = pretrained_backbone.fc.in_features
        elif backbone.startswith('efficientnet'):
            from efficientnet_pytorch import EfficientNet
            self.efficientnet = EfficientNet.from_pretrained(backbone)
            first_conv = nn.Conv2d(6, self.efficientnet._conv_stem.out_channels, kernel_size=3, stride=2, padding=1, bias=False)
            self.efficientnet._conv_stem = first_conv
            self.features = self.efficientnet.extract_features
            self.features_num = self.efficientnet._conv_head.out_channels
        else:
            raise ValueError('wrong backbone')
        
        
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        num_features = self.features_num
        
        self.neck = nn.Sequential(
            nn.BatchNorm1d(num_features),
            nn.Linear(num_features, embedding_size, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(embedding_size),
            nn.Linear(embedding_size, embedding_size, bias=False),
            nn.BatchNorm1d(embedding_size),
        )
            
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                m.momentum = bn_mom
                
    @torch.cuda.amp.autocast()
    def forward(self, x):
        #x = self.features.conv0(x)
        out_backbone = self.features(x)
        z = self.gap(out_backbone)
        
        z = z.view(z.size(0), -1)
        
        #if self.concat_cell_type:
        #    x = torch.cat([x, s], dim=1)

        embedding = self.neck(z)
        return embedding



class BackboneOnly(nn.Module):
    def __init__(self, backbone, embedding_size, stain, bn_mom = 0.05):
        super().__init__()
        
        '''backbone for the architecture. 
            Supported backbones: ResNets, ResNeXts, DenseNets (from torchvision), EfficientNets.'''
            
        if stain == 1:
            stain_ch = 4
        elif stain == 2:
            stain_ch = 3
        elif stain == 3:
            stain_ch = 4
        else:
            stain_ch = 7
            
        if backbone.startswith('densenet'):
            channels = 96 if backbone == 'densenet161' else 64
            first_conv = nn.Conv2d(stain_ch, channels, 7, 2, 3, bias=False)
            pretrained_backbone = getattr(torchvision.models, backbone)(pretrained=True)
            self.features = pretrained_backbone.features
            self.features.conv0 = first_conv
            self.features_num = pretrained_backbone.classifier.in_features
        elif backbone.startswith('resnet') or backbone.startswith('resnext'):
            first_conv = nn.Conv2d(stain_ch, 64, 7, 2, 3, bias=False)
            pretrained_backbone = getattr(torchvision.models, backbone)(pretrained=True)
            self.features = nn.Sequential(
                first_conv,
                pretrained_backbone.bn1,
                pretrained_backbone.relu,
                pretrained_backbone.maxpool,
                pretrained_backbone.layer1,
                pretrained_backbone.layer2,
                pretrained_backbone.layer3,
                pretrained_backbone.layer4,
            )
            self.features_num = pretrained_backbone.fc.in_features
        elif backbone.startswith('efficientnet'):
            from efficientnet_pytorch import EfficientNet
            self.efficientnet = EfficientNet.from_pretrained(backbone)
            first_conv = nn.Conv2d(6, self.efficientnet._conv_stem.out_channels, kernel_size=3, stride=2, padding=1, bias=False)
            self.efficientnet._conv_stem = first_conv
            self.features = self.efficientnet.extract_features
            self.features_num = self.efficientnet._conv_head.out_channels
        else:
            raise ValueError('wrong backbone')
        
        
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        '''
        num_features = self.features_num
        
        self.neck = nn.Sequential(
            nn.BatchNorm1d(num_features),
            nn.Linear(num_features, embedding_size, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(embedding_size),
            nn.Linear(embedding_size, embedding_size, bias=False),
            nn.BatchNorm1d(embedding_size),
        )
            '''
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                m.momentum = bn_mom
                
    #@torch.cuda.amp.autocast()
    def forward(self, x):
        #x = self.features.conv0(x)
        out_backbone = self.features(x)
        z = self.gap(out_backbone)
        
        z = z.view(z.size(0), -1)
        
        #if self.concat_cell_type:
        #    x = torch.cat([x, s], dim=1)

        #embedding = self.neck(z)
        embedding = z
        return embedding




'''
class Embedder(nn.Module):
    def __init__(self, num_features, embedding_size = 1024, concat_cell_type = True, bn_mom = 0.05):
        super().__init__()
        
        self.concat_cell_type = concat_cell_type
        
        #features_num = num_features + (4 if self.concat_cell_type else 0)
        
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        self.neck = nn.Sequential(
            nn.BatchNorm1d(num_features),
            nn.Linear(num_features, embedding_size, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(embedding_size),
            nn.Linear(embedding_size, embedding_size, bias=False),
            nn.BatchNorm1d(embedding_size),
        )
        
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                m.momentum = bn_mom
    
    @torch.cuda.amp.autocast()    
    def forward(self, out_backbone):
        #x = self.features(out_backbone)
        
        #x = F.adaptive_avg_pool2d(out_backbone, (1, 1))
        #x = nn.AdaptiveAvgPool2d(out_backbone, (1, 1))
        x = self.gap(out_backbone)
        
        x = x.view(x.size(0), -1)
        #if self.concat_cell_type:
        #    x = torch.cat([x, s], dim=1)

        embedding = self.neck(x)
        return embedding
        '''

class Classifier(nn.Module):
    def __init__(self, embedding_size, classes, head_hidden = None, bn_mom = 0.05):
        super().__init__()
        
        # hidden layers sizes in the head. Defaults to absence of hidden layers
        if head_hidden is None:
            self.head = nn.Linear(embedding_size, classes)
        else:
            self.head = []
            for input_size, output_size in zip([embedding_size] + head_hidden, head_hidden):
                self.head.extend([
                    nn.Linear(input_size, output_size, bias=False),
                    nn.BatchNorm1d(output_size),
                    nn.ReLU(),
                ])
            self.head.append(nn.Linear(head_hidden[-1], classes))
            self.head = nn.Sequential(*self.head)

        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                m.momentum = bn_mom
    
    #@torch.cuda.amp.autocast()
    def forward(self, embeddings):
        out = self.head(embeddings)
        return out

''' 
class Model(nn.Module):
    def __init__(self, backbone, embedding_size = 1024, concat_cell_type = True, classes = 1139, bn_mom = 0.05):
        super().__init__()
        
        self.backbone = Backbone(backbone)
        self.embedder = Embedder(embedding_size, concat_cell_type, features_num = self.backbone.features_num)
        self.embedding = None
        self.classifier = Classifier(backbone)
        
    def eval_forward(self, x, s):
        embedding = self.embedder.embed(x, s)
        output = self.model.classify(embedding)
        return output
        
        
        return cosine
        
    def embed(self, x, s):
        out_backbone = self.backbone(x)
        return self.embedder.embed(out_backbone, s)


class ModelAndLoss(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.model = Model(args)
        
        #reducer = reducers.DoNothingReducer()
        self.metric_crit =  losses.ArcFaceLoss(args.classes, args.embedding_size, margin=28.6, scale=30.0)
        #self.metric_crit = ArcFaceLoss()
        
        self.crit = DenseCrossEntropy()

    def train_forward(self, x, s, y):
        embedding = self.model.embed(x, s)

        #metric_output = self.model.metric_classify(embedding)
        #metric_loss = self.metric_crit(metric_output, y)
        
        #print(embedding.shape)
        #print(y.shape)
        shape = y.shape    
            
        metric_loss = self.metric_crit(embedding, y.resize_((shape[0])))
        
        #print(metric_loss)
        
        output = self.model.classify(embedding)
        loss = self.crit(output, y.resize_(shape))

        acc = (output.max(1)[1] == y.resize_(shape).max(1)[1]).float().mean().item()

        coeff = self.args.metric_loss_coeff
        return loss * (1 - coeff) + metric_loss / 2 * coeff, acc
    
        #return loss, acc

    def eval_forward(self, x, s):
        embedding = self.model.embed(x, s)
        output = self.model.classify(embedding)
        return output

    def embed(self, x, s):
        return self.model.embed(x, s)


class DenseCrossEntropy(nn.Module):
    def forward(self, x, target):
        x = x.float()
        target = target.float()
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        loss = -logprobs * target
        loss = loss.sum(-1)
        return loss.mean()


class ArcFaceLoss(nn.modules.Module):
    def __init__(self, s=30.0, m=0.5):
        super().__init__()
        self.crit = DenseCrossEntropy()
        self.s = s
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, logits, labels):
        logits = logits.float()
        cosine = logits
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        output = (labels * phi) + ((1.0 - labels) * cosine)
        output *= self.s
        loss = self.crit(output, labels)
        return loss / 2


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, features):
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        return cosine
        
    '''
