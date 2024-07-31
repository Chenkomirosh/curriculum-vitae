import torch
from torch import nn
from torch import optim

import numpy as np
import matplotlib.pyplot as plt
from math import *
import seaborn as sns

class SMS_count_Net(nn.Module):
    def __init__(self):
        super(SMS_count_Net, self).__init__()
        self.conv1 = nn.Conv3d(1, 100, (1, 2, 64), padding=0)
        self.conv2 = nn.Conv3d(100, 200, (1, 2, 12), padding=0)
        self.conv3 = nn.Conv3d(200, 300, (1, 4, 6), padding=0)
        self.conv4 = nn.Conv3d(300, 400, (2, 3, 3), padding=0)
        self.pool = nn.MaxPool3d((1, 1, 2), stride=(1, 1, 2))
        self.act1 = nn.ReLU()
        self.act2 = nn.Tanh()
        self.drop = nn.Dropout(0.4)
        self.flat = nn.Flatten()
        self.liner1 = nn.Linear(3200, 500)
        self.liner2 = nn.Linear(500, 1)
        self.liner3 = nn.Linear(1000, 1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)
        out = self.pool(out)
        out = self.conv2(out)
        out = self.act1(out)
        out = self.pool(out)
        out = self.conv3(out)
        out = self.act1(out)
        out = self.pool(out)
        out = self.conv4(out)
        out = self.act1(out)
        # out = self.pool(out)
        # out = self.flat(out)
        # # print(out.size())
        # out = self.liner1(out)
        # out = self.act2(out)
        # out = self.drop(out)
        # out = self.liner2(out)
        # out = self.act2(out)
        # out = self.drop(out)
        # out = self.liner3(out)
        return out

class Concat_Net(nn.Module):
    def __init__(self):
        super(Concat_Net, self).__init__()
        self.conv1 = nn.Conv3d(400, 500, (1, 3, 3), padding=0)
        self.conv2 = nn.Conv3d(500, 600, (1, 3, 3), padding=0)
        self.conv3 = nn.Conv3d(600, 700, (1, 2, 4), padding=0)
        self.liner1 = nn.Linear(700, 1)
        self.drop = nn.Dropout(0.25)
        self.flat = nn.Flatten()
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.act(out)
        out = self.drop(out)
        out = self.conv2(out)
        out = self.act(out)
        out = self.drop(out)
        out = self.conv3(out)
        out = self.act(out)
        out = self.flat(out)
        out = self.liner1(out)
        return out

class Concat_Net_inception(nn.Module):
    def __init__(self):
        super(Concat_Net_inception, self).__init__()
        self.conv1_1 = nn.Conv3d(400, 200, (1, 1, 1), padding='same')
        self.conv3_3 = nn.Conv3d(400, 200, (1, 3, 3), padding='same')
        self.conv5_5 = nn.Conv3d(400, 200, (1, 5, 5), padding='same')
        self.drop = nn.Dropout(0.25)
        self.flat = nn.Flatten()
        self.act = nn.ReLU()

    def forward(self, x):
        out1 = self.conv1_1(x)
        out1 = self.act(out1)
        out1 = self.drop(out1)

        out3 = self.conv3_3(x)
        out3 = self.act(out3)
        out3 = self.drop(out3)

        out5 = self.conv3_3(x)
        out5 = self.act(out5)
        out5 = self.drop(out5)

        out = torch.concat([out1, out3, out5], dim=3)


def train(numb_epoch, train_x, train_y, val_x, val_y):
    lr = 1e-5
    device = torch.device("cuda")
    batch_size = 64
    test_accuracy_history = []
    test_loss_history = []
    cnn2 = Concat_Net().to(device)
    cnn1 = SMS_count_Net()
    state = torch.load('D:\\networking\\saves\\count_sm\\count_in_smss_249.pt')
    cnn1.load_state_dict(state['state_dict'])
    cnn1.to(device)
    cnn1.eval()
    cec = nn.MSELoss()
    optimazer = optim.Adam(cnn2.parameters(), lr=lr)
    val_y = val_y.to(device)
    val_y = torch.unsqueeze(val_y, 1)

    max_accurancy = 1000
    i = 1
    for epoch in range(numb_epoch):
        order_train = np.random.permutation(len(train_x))
        order_val = np.arange(0, len(val_x))
        torch.set_grad_enabled(True)
        cnn2.train()
        for start_index in range(0, len(train_x), batch_size):
            batch_indexes = order_train[start_index:start_index + batch_size]
            X_batch = train_x[batch_indexes].to(device)
            y_batch = train_y[batch_indexes].to(device)
            with torch.no_grad():
                intermediate_preds1 = cnn1(X_batch[:, :, :, 0, :, :])
                intermediate_preds2 = cnn1(X_batch[:, :, :, 1, :, :])
                intermediate_preds3 = cnn1(X_batch[:, :, :, 2, :, :])
                intermediate_preds4 = cnn1(X_batch[:, :, :, 3, :, :])
                intermediate_preds5 = cnn1(X_batch[:, :, :, 4, :, :])
                intermediate_preds6 = cnn1(X_batch[:, :, :, 5, :, :])
                intermediate_preds = torch.concat(
                    (intermediate_preds1,
                     intermediate_preds2,
                     intermediate_preds3,
                     intermediate_preds4,
                     intermediate_preds5,
                     intermediate_preds6),
                    dim=3
                )

            preds = cnn2(intermediate_preds)
            y_batch = y_batch.to(torch.float)
            y_batch = torch.unsqueeze(y_batch, 1)
            loss_value = cec(preds, y_batch)
            optimazer.zero_grad()
            loss_value.backward()
            optimazer.step()
        # print(loss_value)
        torch.set_grad_enabled(False)
        cnn2.eval()
        for start_index in range(0, len(val_x), batch_size):
            batch_indexes = order_val[start_index:start_index + batch_size]
            X_batch_val = val_x[batch_indexes].to(device)
            # y_batch_val = train_y[batch_indexes].to(device)
            # y_batch_val = torch.unsqueeze(y_batch_val, 1)
            # y_batch_val = y_batch_val.to(torch.float)
            val_intermediate_preds1 = cnn1(X_batch_val[:, :, :, 0, :, :])
            val_intermediate_preds2 = cnn1(X_batch_val[:, :, :, 1, :, :])
            val_intermediate_preds3 = cnn1(X_batch_val[:, :, :, 2, :, :])
            val_intermediate_preds4 = cnn1(X_batch_val[:, :, :, 3, :, :])
            val_intermediate_preds5 = cnn1(X_batch_val[:, :, :, 4, :, :])
            val_intermediate_preds6 = cnn1(X_batch_val[:, :, :, 5, :, :])
            val_intermediate_preds = torch.concat(
                (val_intermediate_preds1,
                 val_intermediate_preds2,
                 val_intermediate_preds3,
                 val_intermediate_preds4,
                 val_intermediate_preds5,
                 val_intermediate_preds6),
                dim=3
            )
            test_preds_batch = cnn2.forward(val_intermediate_preds)
            if start_index == 0:
                test_preds = test_preds_batch
            else:
                test_preds = torch.concat((test_preds, test_preds_batch), dim=0)
        test_loss_history.append(cec(test_preds, val_y).data.cpu())
        accuracy = ((test_preds - val_y).std().mean().float().data.cpu())
        test_accuracy_history.append(accuracy)
        # if accuracy <= max_accurancy:
        #     # best_model = copy.deepcopy(cnn)
        #     max_accurancy = accuracy
        #     checkpoint = {'state_dict': cnn2.state_dict(), 'optimizer': optimazer.state_dict()}
        # if epoch == 100 * i:
        #     torch.save(checkpoint,
        #                f'saves\\count_sm\\count_in_decor_{epoch}.pt')
        #     max_accurancy = 1000
        #     i += 1
        print(epoch)
        print(accuracy)
        # print('preds', test_preds[95])
        # print('real', val_y[95])
    return test_accuracy_history, test_loss_history


if __name__ == '__main__':
    file_train_Mat = open('D:\\networking\\train_Mat_1_6_sm_000_799_3d_img_sms', 'rb')
    file_train_Mask = open('D:\\networking\\train_Mask_1_6_sm_000_799_3d_img_sms', 'rb')
    file_valid_Mat = open('D:\\networking\\test_Mat_1_6_sm_000_799_3d_img_sms', 'rb')
    file_valid_Mask = open('D:\\networking\\test_Mask_1_6_sm_000_799_3d_img_sms', 'rb')

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

    a, l = train(numb_epoch=101, train_x=train_Mat_x, train_y=train_Mask_treck, val_x=val_Mat_x,
                 val_y=val_Mask_treck)
    plt.plot(a)
    plt.show()
    plt.plot(l)
    plt.show()

