"""
All models for the learning Agent
"""


import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
import os
from utils import AverageMeter
# import loggers as lg


class Net(nn.Module):
    """
    This neural network takes as an
    input the raw board representation s of the position and its history, and outputs both
    move probabilities and a value.
    The vector of move probabilities p represents the probability of selecting each move
    The value v is a scalar evaluation, estimating the probability of the current player winning from position s
    """
    def __init__(self, game, num_channels=512, dropout=.3):
        super(Net, self).__init__()
        self.game = game
        self.height = game.height
        self.length = game.length
        self.action_size = len(game.state.action_possible)
        self.num_channels = num_channels
        self.conv = nn.Sequential(
            nn.Conv2d(2, num_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
            nn.Conv2d(num_channels, num_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
            nn.Conv2d(num_channels, num_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
            nn.Conv2d(num_channels, num_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
        )
        self.features = nn.Sequential(
            nn.Linear(self.num_channels*self.length*self.height, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.pi = nn.Linear(512, self.action_size)
        self.value = nn.Linear(512, 1)

    def forward(self, board):
        board = board.view(-1, 2, self.height, self.length).float()
        conv = self.conv(board)
        conv = conv.view(-1, self.num_channels*self.height*self.length)
        features = self.features(conv)
        move_probabilities = self.pi(features)
        v = self.value(features)
        return move_probabilities, F.tanh(v)

    def predict(self, board):
        self.eval()
        moves_probabilities, value = self.forward(board)
        moves_probabilities = F.softmax(moves_probabilities, dim=1).view(-1).detach().numpy()
        return moves_probabilities, value.item()

    def _trainer(self, memory, batch_size, criterion, optimizer):
        self.train()
        transitions = memory.sample(batch_size)
        train_loader = torch.Tensor(transitions)
        for state, target_pi, target_v in train_loader:
            optimizer.zero_grad()
            pi, v = self.forward(state)
            loss = criterion(pi, v, target_pi, target_v)
            loss.backward()
            optimizer.step()

    def training(self, memory, batch_size, criterion, optimizer, epochs):
        for epoch in range(epochs):
            self._trainer(memory, batch_size, criterion, optimizer)


class Net2(nn.Module):
    def __init__(self, game):
        super(Net2, self).__init__()
        self.game = game
        self.height = game.height
        self.length = game.length
        self.action_size = len(game.state.action_possible)
        self.features = nn.Sequential(
            nn.Conv2d(2, 10, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(10, 20, 3, stride=1, padding=1),
            nn.ReLU()
        )
        self.pi = nn.Linear(20*self.height*self.length, self.action_size)
        self.value = nn.Linear(20*self.height*self.length, 1)

    def forward(self, board):
        board = board.view(-1, 2, self.height, self.length).float()
        features = self.features(board)
        features = features.view(-1, 20 * self.height * self.length)
        move_probabilities = self.pi(features)
        v = self.value(features)
        return move_probabilities, F.tanh(v)


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
    def __init__(self, game, logger, filename=None):
        self.game = game
        self.filename = filename
        self.net = ResNet(ResidualBlock, game)
        self.height = game.height
        self.length = game.length
        self.action_size = len(game.state.action_possible)
        self.logger = logger
        if filename is not None:
            self.load_checkpoint('.', filename)

    # def train2(self, memory, batch_size, epochs=10):
    #     optimizer = optim.Adam(self.net.parameters(), weight_decay=0.4)
    #     criterion = Loss()
    #     self.net.train()
    #     losses = AverageMeter()
    #     num_batch = len(memory) // batch_size
    #     lg.logger_model.info('Loop : {}'.format(num_batch))
    #     for _ in range(epochs):
    #         for k in range(num_batch):
    #             optimizer.zero_grad()
    #             # examples = memory.sample(min(len(memory), batch_size))
    #             examples = memory[k*batch_size:((k+1) * batch_size)]
    #             state, pi_target, v_target = list(zip(*examples))
    #             board = torch.cat([torch.from_numpy(x) for x in state]).view(-1, 2, self.height, self.length)
    #             pi_target = torch.Tensor(pi_target)
    #             v_target = torch.Tensor(v_target)
    #             out_pi, out_v = self.net(board)
    #             # out_pi = F.log_softmax(out_pi, 1)
    #             loss = criterion(out_pi, out_v, pi_target, v_target)
    #             losses.update(loss.item(), out_pi.size(0))
    #             loss.backward()
    #             optimizer.step()
    #     lg.logger_model.info('Loss avg : {:.4f}'.format(losses.avg))
    #     lg.logger_model.info('-'*100)

    def train(self, memory, batch_size=32, epochs=2):
        # optimizer = optim.Adam(self.net.parameters(), weight_decay=0.4)
        optimizer = optim.SGD(self.net.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.4)
        criterion = Loss(self.logger)
        self.net.train()
        losses = AverageMeter()
        num_batch = len(memory) // batch_size
        self.logger.info('Loop : {}'.format(num_batch))
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
            self.logger.info('Loss avg : {:.4f}'.format(losses.avg))
        self.logger.info('-'*100)

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

    def __reduce__(self):
        return self.__class__, (self.game, self.length, self.filename)


class Loss(nn.Module):
    def __init__(self, logger):
        super(Loss, self).__init__()
        self.logger = logger

    def forward(self, m_proba, v, target_pi, target_v):
        # zero = torch.zeros_like(m_proba)
        # where = torch.eq(target_pi, zero)
        # neg = -100.0 * where.float()
        # m_proba = torch.where(where, neg, m_proba)
        m_proba = F.log_softmax(m_proba, 1)
        l2 = -torch.sum(target_pi*m_proba, 1).mean()
        l1 = F.mse_loss(v, target_v.view(-1, 1))
        self.logger.info('policy loss: {:.4f}\t\tvalue loss: {:.4f}'.format(l2.item(), l1.item()))
        return l1 + l2
