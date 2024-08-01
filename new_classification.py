import torch
from torch import nn
from torch import optim
import numpy as np
import datetime
import matplotlib.pyplot as plt
import torch.nn.functional as F

class New_Classification_Net(nn.Module):
    def __init__(self):
        super(New_Classification_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, (8, 128), padding=0)
        self.conv2 = nn.Conv2d(8, 16, (8, 64), padding=0)
        self.conv3 = nn.Conv2d(16, 32, (4, 32), padding=0)
        self.conv4 = nn.Conv2d(32, 64, (3, 8), padding=0)

        self.fc1 = nn.Linear(576, 10)
        self.fc2 = nn.Linear(10, 2)

        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.pool2 = nn.MaxPool2d((1, 2), stride=2)
        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(64)

        self.act = nn.ReLU()
        self.soft = nn.Softmax()
        self.drop = nn.Dropout(p=0.25)
        self.flat = nn.Flatten()


    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.pool1(out)
        out = self.drop(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act(out)
        out = self.pool1(out)
        out = self.drop(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.act(out)
        out = self.pool1(out)
        out = self.drop(out)

        out = self.conv4(out)
        out = self.bn4(out)
        out = self.act(out)
        out = self.drop(out)

        out = self.flat(out)

        out = self.fc1(out)
        out = self.act(out)
        out = self.drop(out)

        out = self.fc2(out)
        # out = self.soft(out)

        return out


def train(numb_epoch, device, train_x, train_y, val_x, val_y):
    lr = 1e-4
    batch_size = 64
    max_accurancy = 0
    cnn = New_Classification_Net().to(device)
    cec = nn.CrossEntropyLoss()
    optimazer = optim.Adam(cnn.parameters(), lr=lr)
    loss_train_history = []
    loss_valid_history = []
    accuracy_valid_history = []
    i = 1
    val_y = val_y.to(device)
    val_y = val_y.to(torch.int64)
    # val_y = torch.unsqueeze(val_y, 1)


    for epoch in range(numb_epoch):
        cnn.train()
        order_train = np.random.permutation(len(train_x))
        order_val = np.arange(0, len(val_x))
        loss_train = 0.0
        for start_index in range(0, len(train_x), batch_size):
            batch_indexes = order_train[start_index:start_index + batch_size]

            X_batch = train_x[batch_indexes].to(device)
            y_batch = train_y[batch_indexes].to(device)

            y_batch = y_batch.to(torch.int64)

            # y_batch = torch.unsqueeze(y_batch, 1)

            preds = cnn.forward(X_batch)
            # preds = F.softmax(preds, dim=-1)
            # print(preds)


            # print(torch.argmax(preds, dim=1))
            loss = cec(preds, y_batch)
            optimazer.zero_grad()
            loss.backward()
            optimazer.step()
            loss_train += loss.item()

        loss_train_history.append(loss_train)
        cnn.eval()
        with torch.inference_mode():
            for start_index in range(0, len(val_x), batch_size):
                batch_indexes = order_val[start_index:start_index + batch_size]
                X_batch_val = val_x[batch_indexes].to(device)
                test_preds_batch = cnn.forward(X_batch_val)
                if start_index == 0:
                    test_preds = test_preds_batch
                else:
                    test_preds = torch.concat([test_preds, test_preds_batch], dim=0)



            loss_valid_history.append(cec(test_preds, val_y).data.cpu())
            accuracy = (test_preds.argmax(dim=1) == val_y).float().mean().data.cpu()
            accuracy_valid_history.append(accuracy)
            print(test_preds.shape)

            if accuracy > max_accurancy:
                max_accurancy = accuracy
                checkpoint = {'state_dict': cnn.state_dict(), 'optimizer': optimazer.state_dict()}
            if epoch == 25 * i:
                torch.save(checkpoint, f'saves\\count_sm\\new_classification_e6_{epoch}.pt')
                i += 1
            print('{} Epoch {}, Accurancy {}'.format(datetime.datetime.now(), epoch, accuracy))
    return loss_train_history, loss_valid_history, accuracy_valid_history


if __name__ == '__main__':
    file_train_Mat = open('D:\\networking\\data\\Grs3_55_seria_13_obr\\classification\\train_Mat', 'rb')
    file_train_Mask = open('D:\\networking\\data\\Grs3_55_seria_13_obr\\classification\\train_Mask', 'rb')
    file_valid_Mat = open('D:\\networking\\data\\Grs3_55_seria_13_obr\\classification\\valid_Mat', 'rb')
    file_valid_Mask = open('D:\\networking\\data\\Grs3_55_seria_13_obr\\classification\\valid_Mask', 'rb')

    train_Mat_x = np.load(file_train_Mat)
    train_Mask = np.load(file_train_Mask)
    val_Mat_x = np.load(file_valid_Mat)
    val_Mask = np.load(file_valid_Mask)

    file_train_Mat.close()
    file_train_Mask.close()
    file_valid_Mat.close()
    file_valid_Mask.close()

    train_Mat_x = torch.from_numpy(np.asarray(train_Mat_x))
    train_Mask = torch.from_numpy(np.asarray(train_Mask))
    val_Mat_x = torch.from_numpy(np.asarray(val_Mat_x))
    val_Mask = torch.from_numpy(np.asarray(val_Mask))

    train_Mask_treck = train_Mask[:, 2]
    val_Mask_treck = val_Mask[:, 2]

    device = torch.device("cuda")

    numb_epoch = 126

    # print(train_Mask_treck)

    lt, lv, ac = train(numb_epoch=numb_epoch, device=device, train_x=train_Mat_x, train_y=train_Mask_treck, val_x=val_Mat_x,
                       val_y=val_Mask_treck)

    plt.plot(lt)
    plt.show()

    plt.plot(lv)
    plt.show()

    plt.plot(ac)
    plt.show()






