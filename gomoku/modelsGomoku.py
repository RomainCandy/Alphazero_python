"""
All models for the learning Agent
"""


import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
import os
import sys
from . import loggers as lg
sys.path.append('..')
from utils import AverageMeter


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)


# Residual block
# https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/deep_residual_network/main.py
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.cnn_block = nn.Sequential(
            conv3x3(in_channels, out_channels, stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            conv3x3(out_channels, out_channels),
            nn.BatchNorm2d(out_channels)
        )
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.cnn_block(x)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = F.relu(out, inplace=True)
        return out


# ResNet
class ResNet(nn.Module):
    def __init__(self, block, game):
        super(ResNet, self).__init__()
        self.game = game
        self.height = game.height
        self.length = game.length
        self.action_size = len(game.state.action_possible)

        self.in_channels = 256
        self.preprocessing = nn.Sequential(
            conv3x3(2, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.layer1 = self.make_layer(block, 256, 9)
        # self.layer2 = self.make_layer(block, 32, layers[0])
        # self.layer3 = self.make_layer(block, 64, layers[1])
        self.policy_conv = nn.Sequential(
            nn.Conv2d(256, 2, kernel_size=1, stride=1),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
        )
        self.policy = nn.Linear(self.length * self.height * 2, self.action_size)

        self.value_conv = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )

        self.value = nn.Sequential(
            nn.Linear(self.length * self.height, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = [(block(self.in_channels, out_channels, stride, downsample))]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, board):
        board = board.view(-1, 2, self.height, self.length).float()
        out = self.preprocessing(board)
        out = self.layer1(out)
        out_pol = self.policy_conv(out)
        out_pol = out_pol.view(-1, self.length*self.height*2)
        out_pol = self.policy(out_pol)
        out_val = self.value_conv(out)
        out_val = out_val.view(-1, self.length*self.height)
        out_val = self.value(out_val)
        return out_pol, out_val


class WrapperNet:
    def __init__(self, game):
        self.net = ResNet(ResidualBlock, game)
        self.height = game.height
        self.length = game.length
        self.action_size = len(game.state.action_possible)

    def train(self, memory, batch_size=32, epochs=2):
        # optimizer = optim.Adam(self.net.parameters(), weight_decay=0.4)
        optimizer = optim.SGD(self.net.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.4)
        criterion = Loss()
        self.net.train()
        losses = AverageMeter()
        num_batch = len(memory) // batch_size
        lg.logger_model.info('Loop : {}'.format(num_batch))
        for _ in range(epochs):
            for k in range(num_batch):
                optimizer.zero_grad()
                state, pi_target, v_target = list(zip(*memory))
                board = torch.cat([torch.from_numpy(x) for x in state]).view(-1, 2, self.height, self.length)
                pi_target = torch.Tensor(pi_target)
                v_target = torch.Tensor(v_target)
                out_pi, out_v = self.net(board)
                loss = criterion(out_pi, out_v, pi_target, v_target)
                losses.update(loss.item(), out_pi.size(0))
                loss.backward()
                optimizer.step()
            lg.logger_model.info('Loss avg : {:.4f}'.format(losses.avg))
        lg.logger_model.info('-'*100)

    def predict(self, state):
        self.net.eval()
        board = torch.from_numpy(state).float()
        moves_probabilities, value = self.net(board)
        moves_probabilities = moves_probabilities.view(-1).detach().numpy()
        return moves_probabilities, value.item()

    def save_checkpoint(self, folder, filename):
        filepath = os.path.join(folder, filename)
        torch.save({
            'state_dict': self.net.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder, filename):
        filepath = os.path.join(folder, filename)
        checkpoint = torch.load(filepath)
        self.net.load_state_dict(checkpoint['state_dict'])


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, m_proba, v, target_pi, target_v):
        zero = torch.zeros_like(m_proba)
        where = torch.eq(target_pi, zero)
        neg = -100.0 * where.float()
        m_proba = torch.where(where, neg, m_proba)
        m_proba = F.log_softmax(m_proba, 1)
        l2 = -torch.sum(target_pi*m_proba, 1).mean()
        l1 = F.mse_loss(v, target_v.view(-1, 1))
        lg.logger_model.info('policy loss: {:.4f}\t\tvalue loss: {:.4f}'.format(l2.item(), l1.item()))
        return l1 + l2
