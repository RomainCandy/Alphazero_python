import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    """
    This neural network takes as an
    input the raw board representation s of the position and its history, and outputs both
    move probabilities and a value.
    The vector of move probabilities p represents the probability of selecting each move
    The value v is a scalar evaluation, estimating the probability of the current player winning from position s
    """
    def __init__(self, length=7, height=6):
        super(Net, self).__init__()
        self.length = length
        self.height = height
        self.features = nn.Linear(length * height, length)
        self.value = nn.Linear(length * height, 1)

    def forward(self, board):
        board = board.float()
        out = board.view(-1, self.length * self.height)
        move_probabilities = self.features(out)
        v = self.value(out)
        return move_probabilities, F.tanh(v)

    def predict(self, board):
        self.eval()
        moves_probabilities, value = self.forward(board)
        moves_probabilities = F.softmax(moves_probabilities, dim=1).view(-1).detach().numpy()
        return moves_probabilities, value.item()

    def _trainer(self, train_loader, criterion, optimizer):
        self.train()
        for state, target_pi, target_v in train_loader:
            optimizer.zero_grad()
            pi, v = self.forward(state)
            loss = criterion(pi, v, target_pi, target_v)
            loss.backward()
            optimizer.step()

    def training(self, train_loader, criterion, optimizer, epochs):
        for epoch in range(epochs):
            self._trainer(train_loader, criterion, optimizer)


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, m_proba, v, target_pi, target_v):
        return F.mse_loss(m_proba, target_v) + F.cross_entropy(v, target_pi)
