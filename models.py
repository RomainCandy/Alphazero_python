import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
import os
from utils import AverageMeter
import loggers as lg


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


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, m_proba, v, target_pi, target_v):
        # m_proba = F.log_softmax(m_proba, 1)
        l2 = -torch.sum(target_pi*m_proba)/target_pi.size(0)
        # l2 = F.kl_div(m_proba, target_pi)
        l1 = F.smooth_l1_loss(v, target_v.view(-1, 1))
        return l1 + l2


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


class WrapperNet:
    def __init__(self, game):
        self.net = Net(game)
        self.height = game.height
        self.length = game.length
        self.action_size = len(game.state.action_possible)

    def train(self, memory, batch_size, epochs=10):
        optimizer = optim.Adam(self.net.parameters(), weight_decay=0.4)
        criterion = Loss()
        self.net.train()
        losses = AverageMeter()
        num_batch = len(memory) // batch_size
        lg.logger_model.info('Loop : {}'.format(num_batch))
        for _ in range(epochs):
            for k in range(num_batch):
                optimizer.zero_grad()
                # examples = memory.sample(min(len(memory), batch_size))
                examples = memory.memory[k*batch_size:((k+1) * batch_size)]
                state, pi_target, v_target = list(zip(*examples))
                board = torch.cat([torch.from_numpy(x) for x in state]).view(-1, 2, self.height, self.length)
                pi_target = torch.Tensor(pi_target)
                v_target = torch.Tensor(v_target)
                out_pi, out_v = self.net(board)
                out_pi = F.log_softmax(out_pi, 1)
                loss = criterion(out_pi, out_v, pi_target, v_target)
                losses.update(loss.item(), out_pi.size(0))
                loss.backward()
                optimizer.step()
            lg.logger_model.info('Loss avg : {}'.format(losses.avg))
        lg.logger_model.info('-'*100)

    def predict(self, state):
        self.net.eval()
        board = torch.from_numpy(state).float()
        moves_probabilities, value = self.net(board)
        moves_probabilities = F.softmax(moves_probabilities, dim=1).view(-1).detach().numpy()
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
