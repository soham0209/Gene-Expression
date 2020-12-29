import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import sys


class CNN1D(nn.Module):
    def __init__(self, input_dim, filter_size, output_dim):
        super(CNN1D, self).__init__()
        self.conv = nn.Conv2d(1, 1, stride=1, kernel_size=(1, filter_size))
        self.maxpool = nn.MaxPool2d(kernel_size=(1, 2))
        self.act = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.2)
        dense_input = (input_dim - filter_size + 1) // 2
        self.dense = nn.Sequential(nn.Linear(dense_input, 64), nn.ReLU(), nn.Linear(64, 32),
                                   nn.ReLU(), nn.Linear(32, 2))

    def forward(self, x):
        x = torch.unsqueeze(torch.unsqueeze(x, 0), 0)
        # print(x.shape)
        x = self.conv(x)
        x = self.maxpool(self.act(x))
        x = torch.squeeze(torch.squeeze(x, 0), 0)
        x = self.dropout(x)
        x = self.dense(x)
        # label = torch.argmax(x, dim=1)
        return x


class trainData(Dataset):

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


# test data
class testData(Dataset):

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.max(y_pred, 1)[1]
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)
    return acc


if __name__ == '__main__':
    lr = 0.01
    EPOCHS = 2
    BATCH_SIZE = 20
    FILTER_SIZE = 100
    OUT_SIZE = 2
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print('Device =', device)
    # Toy data
    fname = 'Human_Dengue'
    type_data = 'var'
    if len(sys.argv) > 1:
        fname = sys.argv[1]
    if len(sys.argv) > 2:
        type_data = sys.argv[2]
    if fname == 'Human_bowel':
        OUT_SIZE = 3
    ip_fname = fname + '/' + fname + '_' + type_data + '_genex.npy'
    print('Loaded ', ip_fname)
    data_x = np.load(ip_fname, allow_pickle=True).astype(float)
    data_y = np.load(fname + '/' + fname + '_labels.npy', allow_pickle=True).astype(float)
    X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.33, random_state=69, shuffle=True)
    train_data = trainData(torch.Tensor(X_train),
                           torch.Tensor(y_train))
    test_data = testData(torch.Tensor(X_test), torch.Tensor(y_test))

    # Data loader
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True)

    model = CNN1D(data_x.shape[1], FILTER_SIZE, OUT_SIZE).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    for e in range(1, EPOCHS + 1):
        epoch_loss = 0
        epoch_acc = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()

            y_pred = model(X_batch)
            # y_pred = y_pred.type(torch.FloatTensor)
            # y_bc = y_batch.type(torch.long)
            loss = criterion(y_pred, y_batch.long())
            acc = binary_acc(y_pred, y_batch)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()

        print(f'Epoch {e + 0:03}: | Loss: {epoch_loss / len(train_loader):.5f} | Acc: {epoch_acc / len(train_loader):.3f}')
    print('#### TEST ACC #####')
    model.eval()
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch.long())
        acc = binary_acc(y_pred, y_batch)
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    print(f'Epoch {e + 0:03}: | Loss: {epoch_loss / len(train_loader):.5f} | Acc: {epoch_acc / len(train_loader):.3f}')


