"""
Author: Zhenbo Xu
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import torch
import torch.nn as nn
import models.erfnet as erfnet
import torch.nn.functional as F
import math


def LocationEmbedding(f_g, dim_g=64, wave_len=1000):
    x_min, y_min, x_max, y_max = torch.chunk(f_g, 4, dim=1)
    cx = (x_min + x_max) * 0.5
    cy = (y_min + y_max) * 0.5
    w = (x_max - x_min) + 1.
    h = (y_max - y_min) + 1.
    position_mat = torch.cat((cx, cy, w, h), -1)

    feat_range = torch.arange(dim_g / 8).cuda()
    dim_mat = feat_range / (dim_g / 8)
    dim_mat = 1. / (torch.pow(wave_len, dim_mat))

    dim_mat = dim_mat.view(1, 1, -1)
    position_mat = position_mat.view(f_g.shape[0], 4, -1)
    position_mat = 100. * position_mat

    mul_mat = position_mat * dim_mat
    mul_mat = mul_mat.view(f_g.shape[0], -1)
    sin_mat = torch.sin(mul_mat)
    cos_mat = torch.cos(mul_mat)
    embedding = torch.cat((sin_mat, cos_mat), -1)
    return embedding


class BranchedERFNet(nn.Module):
    def __init__(self, num_classes, input_channel=3, encoder=None):
        super().__init__()

        print('Creating branched erfnet with {} classes'.format(num_classes))

        if (encoder is None):
            self.encoder = erfnet.Encoder(sum(num_classes), input_channel=input_channel)
        else:
            self.encoder = encoder

        self.decoders = nn.ModuleList()
        for n in num_classes:
            self.decoders.append(erfnet.Decoder(n))

    def init_output(self, n_sigma=1):
        with torch.no_grad():
            output_conv = self.decoders[0].output_conv
            print('initialize last layer with size: ',
                  output_conv.weight.size())

            output_conv.weight[:, 0:2, :, :].fill_(0)
            output_conv.bias[0:2].fill_(0)

            output_conv.weight[:, 2:2+n_sigma, :, :].fill_(0)
            output_conv.bias[2:2+n_sigma].fill_(1)

    def forward(self, input, only_encode=False):
        if only_encode:
            return self.encoder.forward(input, predict=True)
        else:
            output = self.encoder(input)

        return torch.cat([decoder.forward(output) for decoder in self.decoders], 1)


class PointFeatFuse3P(nn.Module):
    # three path
    def __init__(self, num_points=250, ic=7, oc=64, maxpool=True):
        super(PointFeatFuse3P, self).__init__()
        self.conv1 = torch.nn.Conv1d(2, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 128, 1)

        self.e_conv1 = torch.nn.Conv1d(ic, 64, 1)
        self.e_conv2 = torch.nn.Conv1d(64, 128, 1)
        self.e_conv3 = torch.nn.Conv1d(128, 256, 1)

        self.t_conv1 = torch.nn.Conv1d(3, 64, 1)
        self.t_conv2 = torch.nn.Conv1d(64, 128, 1)
        self.t_conv3 = torch.nn.Conv1d(128, 128, 1)

        self.conv4 = torch.nn.Conv1d(512, 256, 1)
        self.conv5 = torch.nn.Conv1d(256, 512, 1)
        self.conv6 = torch.nn.Conv1d(512, oc, 1)

        if maxpool:
            self.pool = torch.nn.MaxPool1d(num_points)
        else:
            self.pool = torch.nn.AvgPool1d(num_points)

        self.num_points = num_points

    def forward(self, x, emb, t, withInd=False):
        x = F.leaky_relu(self.conv1(x))
        emb = F.leaky_relu(self.e_conv1(emb))
        t = F.leaky_relu(self.t_conv1(t))

        x = F.leaky_relu(self.conv2(x))
        emb = F.leaky_relu(self.e_conv2(emb))
        t = F.leaky_relu(self.t_conv2(t))

        x = F.leaky_relu(self.conv3(x))
        emb = F.leaky_relu(self.e_conv3(emb))
        t = F.leaky_relu(self.t_conv3(t))

        pointfeat_2 = torch.cat((x, emb, t), dim=1)

        x1 = F.leaky_relu(self.conv4(pointfeat_2))
        x1 = F.leaky_relu(self.conv5(x1))
        x1 = F.leaky_relu(self.conv6(x1))
        if withInd:
            return self.pool(x1).squeeze(-1), torch.max(x1, dim=2)[1]
        return self.pool(x1).squeeze(-1)


class PoseNetFeatOffsetEmb(nn.Module):
    # bn with border
    def __init__(self, num_points=250, ic=7, border_points=200, border_ic=6, output_dim=64, category=False):
        super(PoseNetFeatOffsetEmb, self).__init__()
        self.category = category
        bc = 256
        self.borderConv = PointFeatFuse3P(ic=border_ic, oc=bc, num_points=border_points)

        self.conv1 = torch.nn.Conv1d(2, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 256, 1)
        self.conv1_bn = nn.BatchNorm1d(64)
        self.conv2_bn = nn.BatchNorm1d(128)
        self.conv3_bn = nn.BatchNorm1d(256)

        self.e_conv1 = torch.nn.Conv1d(ic, 64, 1)
        self.e_conv2 = torch.nn.Conv1d(64, 128, 1)
        self.e_conv3 = torch.nn.Conv1d(128, 256, 1)
        self.e_conv1_bn = nn.BatchNorm1d(64)
        self.e_conv2_bn = nn.BatchNorm1d(128)
        self.e_conv3_bn = nn.BatchNorm1d(256)

        self.conv4 = torch.nn.Conv1d(512, 256, 1)
        self.conv5 = torch.nn.Conv1d(256, 512, 1)
        self.conv6 = torch.nn.Conv1d(512, 64, 1)
        self.conv4_bn = nn.BatchNorm1d(256)
        self.conv5_bn = nn.BatchNorm1d(512)

        self.conv7 = torch.nn.Conv1d(512, 256, 1)
        self.conv8 = torch.nn.Conv1d(256, 512, 1)
        self.conv9 = torch.nn.Conv1d(512, 64, 1)
        self.conv7_bn = nn.BatchNorm1d(256)
        self.conv8_bn = nn.BatchNorm1d(512)

        self.conv_weight = torch.nn.Conv1d(128, 1, 1)

        self.last_emb = nn.Sequential(
            nn.Linear(704+bc, 256),
            nn.LeakyReLU(),
            nn.Linear(256, output_dim)
        )
        self.ap1 = torch.nn.AvgPool1d(num_points)
        self.mp2 = torch.nn.MaxPool1d(num_points)
        self.num_points = num_points

    def forward(self, inp, borders, spatialEmbs, with_weight=False):
        x, emb = inp[:,-2:], inp[:,:-2]
        x = F.leaky_relu(self.conv1_bn(self.conv1(x)))
        emb = F.leaky_relu(self.e_conv1_bn(self.e_conv1(emb)))

        x = F.leaky_relu(self.conv2_bn(self.conv2(x)))
        emb = F.leaky_relu(self.e_conv2_bn(self.e_conv2(emb)))

        x = F.leaky_relu(self.conv3_bn(self.conv3(x)))          # B,256,N
        emb = F.leaky_relu(self.e_conv3_bn(self.e_conv3(emb)))  # B,256,N

        pointfeat_2 = torch.cat((x, emb), dim=1)

        x1 = F.leaky_relu(self.conv4_bn(self.conv4(pointfeat_2)))
        x1 = F.leaky_relu(self.conv5_bn(self.conv5(x1)))
        x1 = F.leaky_relu(self.conv6(x1))                       # B,64,N
        ap_x1 = self.ap1(x1).squeeze(-1)                        # B,64

        x2 = F.leaky_relu(self.conv7_bn(self.conv7(pointfeat_2)))
        x2 = F.leaky_relu(self.conv8_bn(self.conv8(x2)))
        x2 = F.leaky_relu(self.conv9(x2))                       # B,64,N
        mp_x2 = self.mp2(x2).squeeze(-1)                        # B,64

        weightFeat = self.conv_weight(torch.cat([x1, x2], dim=1))   #B,1,N
        weight = torch.nn.Softmax(2)(weightFeat)
        weight_x3 = (weight.expand_as(pointfeat_2) * pointfeat_2).sum(2)

        if with_weight:
            border_feat, bg_inds = self.borderConv(borders[:, 3:5], borders[:, :3], borders[:, 5:], withInd=with_weight)
            x = torch.cat([ap_x1, mp_x2, weight_x3, border_feat, spatialEmbs], dim=1)
            outp = self.last_emb(x)
            return outp, weight, bg_inds
        else:
            border_feat = self.borderConv(borders[:, 3:5], borders[:, :3], borders[:, 5:])

        x = torch.cat([ap_x1, mp_x2, weight_x3, border_feat, spatialEmbs], dim=1)
        outp = self.last_emb(x)
        return outp


class TrackerOffsetEmb(nn.Module):
    # for uv offset and category
    def __init__(self, margin=0.3, num_points=250, border_ic=6, env_points=200, category=False, outputD=64, v23=False):
        super().__init__()
        self.point_feat = PoseNetFeatOffsetEmb(num_points=num_points, ic=3, border_points=env_points, border_ic=border_ic, output_dim=outputD, category=True)
        self.num_points = num_points
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.embedding = LocationEmbedding

    def init_output(self, n_sigma=1):
        pass

    def compute_triplet_loss(self, inputs, targets):
        """
        Args:
            inputs: feature matrix with shape (n_samples*3, feat_dim)
            targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)
        #print('n:', n) # 72
        # inputs = 1. * inputs / (torch.norm(inputs, 2, dim=-1, keepdim=True).expand_as(inputs) + 1e-12)
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        #print('dist:', dist.size())
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        loss = torch.zeros([1]).cuda()
        if mask.float().unique().shape[0] > 1:
            dist_ap, dist_an = [], []
            for i in range(n):
                dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
                dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
            dist_ap = torch.cat(dist_ap)
            dist_an = torch.cat(dist_an)
            # Compute ranking hinge loss
            y = torch.ones_like(dist_an)
            #print('dist_an:', dist_an.size()) # 72
            loss = self.ranking_loss(dist_an, dist_ap, y).unsqueeze(0)
        return loss

    def forward(self, points, labels, xyxys, infer=False, visualize=False):

        #print('points', points.size()) # ([1, 72, 1500, 8]): (batch_size, n_sample*3, n_points, RGBXYCCC)
        #print('xyxys', xyxys.size()) # ([1, 72, 4]): (batch_size, n_sample*3, bbox_coords)
        #print('labels', labels.size()) # ([1, 72]): (batch_size, n_sample*3)

        points, xyxys = points[0], xyxys[0]
        xy_embeds = self.embedding(xyxys)
        envs = points[:,self.num_points:]
        points = points[:,:self.num_points, :5]
        if infer:
            return self.inference(points, envs, xy_embeds)
        elif visualize:
            embeds, point_weights, bg_inds = self.point_feat(points.transpose(2, 1).contiguous(), envs.transpose(2, 1).contiguous(), xy_embeds, with_weight=True)
            return embeds, point_weights, bg_inds
        else:
            embeds = self.point_feat(points.transpose(2, 1).contiguous(), envs.transpose(2, 1).contiguous(), xy_embeds)
            labels = labels[0]
            print('embeds', embeds.size()) # ([72, 32]): (n_sample*3, embed_dim)
            return self.compute_triplet_loss(embeds, labels)

    def inference(self, points, envs, embeds):
        # assert points.shape[0] == 1
        embeds = self.point_feat(points.transpose(2, 1).contiguous(), envs.transpose(2, 1).contiguous(), embeds)
        return embeds




"""
@vtsai01
using PointTrack output(point_feat) to train a Transformer to generate a embedding for next appear instance.
"""
class TransformerTrackerEmb(nn.Module):
    # for uv offset and category
    def __init__(self, margin=0.3, num_points=250, border_ic=6, env_points=200, category=False, outputD=64, v23=False):
        super().__init__()
        torch.autograd.set_detect_anomaly(True)
        print('TransformerTrackerEmb')
        self.point_feat = PoseNetFeatOffsetEmb(num_points=num_points, ic=3, border_points=env_points, border_ic=border_ic, output_dim=outputD, category=True)
        self.num_points = num_points
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.embedding = LocationEmbedding
        self.tranformer_model = TransformerModel(outputD, 2, 200, 2, posenc_max_len=5000)
        self.outputD = outputD
    def init_output(self, n_sigma=1):
        pass

    def compute_triplet_loss(self, inputs, targets):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels(id) with shape (num_classes)
        """
        n = inputs.size(0)
        # inputs = 1. * inputs / (torch.norm(inputs, 2, dim=-1, keepdim=True).expand_as(inputs) + 1e-12)
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        loss = torch.zeros([1]).cuda()
        if mask.float().unique().shape[0] > 1:
            dist_ap, dist_an = [], []
            for i in range(n):
                dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
                dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
            dist_ap = torch.cat(dist_ap)
            dist_an = torch.cat(dist_an)
            # Compute ranking hinge loss
            y = torch.ones_like(dist_an)
            loss = self.ranking_loss(dist_an, dist_ap, y).unsqueeze(0)
        return loss

    def forward(self, points=None, labels=None, xyxys=None, framestamp=None, embeds=None, current_frame=None, infer=False, visualize=False, infer_transformer_only=False):
        #print('forwarding')
        if infer_transformer_only:
            #framestamp = framestamp[0]
            # embeds: (3, ?, 32)
            # framestamp: (3, ?)
            #print('f', framestamp)
            inds = current_frame - framestamp
            print('inds', inds)
            output = self.tranformer_model(embeds, inds)
            return output[-1, :]
        else:
            if points is not None and xyxys is not None:
                points, xyxys = points[0], xyxys[0]
            if framestamp is not None:
                framestamp = framestamp[0]
            xy_embeds = self.embedding(xyxys)
            envs = points[:,self.num_points:]
            points = points[:,:self.num_points, :5]
            if infer:
                return self.inference(points, envs, xy_embeds)
            elif visualize:
                embeds, point_weights, bg_inds = self.point_feat(points.transpose(2, 1).contiguous(), envs.transpose(2, 1).contiguous(), xy_embeds, with_weight=True)
                return embeds, point_weights, bg_inds
            else:
                embeds = self.point_feat(points.transpose(2, 1).contiguous(), envs.transpose(2, 1).contiguous(), xy_embeds)
                labels = labels[0]
                triplet_losses = self.compute_triplet_loss(embeds, labels)
                #print('embeds', embeds.size()) # ([72, 32]): (n_sample*3, embed_dim)
                #print('framestamp', framestamp)
                #print('labels', labels)
                #####################
                # append the embeds #
                #####################
                # if
                # emb: (72, 32) 
                #   -> (96, 32)
                #   => (4, 24, 32)
                # then
                # src: (3, 24, 32)
                # tgt: (1, 24, 32)
                ##################### 
                embeds_re = torch.reshape(embeds, (-1, 4, self.outputD)).permute(1, 0, 2)
                labels_re = torch.reshape(labels, (-1, 4)).permute(1, 0)
                fstamp_re = torch.reshape(framestamp, (-1, 4)).permute(1, 0)

                #print('---'*30)
                fstamp_src = fstamp_re[:3, :]
                fstamp_tgt = fstamp_re[-1, :]
                labels = labels_re[:3, :]
                src = embeds_re[:3, :]
                tgt = embeds_re[-1, :]
                #print('labels', labels)
                #print('fstamp', fstamp_src)
                inds = fstamp_tgt - fstamp_src
                #print('inds',inds.size(),  inds)
                ###########################################################################################################
                #    For transformer,
                #    src:(S, N, E), tgt:(T, N, E) where S and T are seq length, N is batch size, E is feature number
                #    In this case dim should be src:(10, N, 64) which is  (frames, n_samples, feature_dims)
                ###########################################################################################################

                output = self.tranformer_model(src, inds)  # we have to bulid a customize transformer because of the poisition encoding.
                #print('transformer_output SIZE', output.size())

                y = torch.ones_like(tgt)
                transformer_losses = self.ranking_loss(output[-1, :], tgt, y).unsqueeze(0)
            
                return triplet_losses + transformer_losses

    def inference(self, points, envs, embeds):
        # assert points.shape[0] == 1
        embeds = self.point_feat(points.transpose(2, 1).contiguous(), envs.transpose(2, 1).contiguous(), embeds)
        return embeds

class TransformerModel(nn.Module):

    def __init__(self, emb_dim, nhead, nhid, nlayers, posenc_max_len=5000,  dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(emb_dim, dropout, posenc_max_len)
        encoder_layers = TransformerEncoderLayer(emb_dim, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        self.encoder = nn.Linear(emb_dim, emb_dim)
        self.decoder = nn.Linear(emb_dim, emb_dim)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.bias.data.zero_()
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, inds):

        src = F.relu(self.encoder(src))
        src = self.pos_encoder(src, inds)
        output = self.transformer_encoder(src)
        output = self.decoder(output)

        return output

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x, inds):
        #print('-inds', inds.size()) #
        #print('==', inds[:, :, None].size())
        #print('x',  x.size())
        #print(('pe', self.pe.size()))
        #print('==', self.pe.repeat(1, x.size(1), 1)[inds[:, :, None]].size())
        #x = x + self.pe.repeat(1, x.size(1), 1)[inds[:, :, None]] 
        x = x + self.pe.squeeze()[inds]
        return self.dropout(x)