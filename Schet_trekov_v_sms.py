import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim

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
        self.drop = nn.Dropout(0.25)
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
        out = self.pool(out)
        out = self.flat(out)
        # print(out.size())
        out = self.liner1(out)
        out = self.act2(out)
        out = self.drop(out)
        out = self.liner2(out)
        # out = self.act2(out)
        # out = self.drop(out)
        # out = self.liner3(out)
        return out


def train(numb_epoch, device, train_x, train_y, val_x, val_y):
    lr = 1e-5
    batch_size = 500
    test_accuracy_history = []
    test_loss_history = []
    cnn = SMS_count_Net().to(device)
    cec = nn.MSELoss()
    optimazer = optim.Adam(cnn.parameters(), lr=lr)
    val_x = val_x.to(device)
    val_y = val_y.to(device)
    val_y = torch.unsqueeze(val_y, 1)
    val_y = val_y.to(torch.float)
    max_accurancy = 1000
    i = 1
    cnn.train()
    for epoch in range(numb_epoch):
        order = np.random.permutation(len(train_x))

        for start_index in range(0, len(train_x), batch_size):
            optimazer.zero_grad()

            batch_indexes = order[start_index:start_index + batch_size]

            X_batch = train_x[batch_indexes].to(device)
            y_batch = train_y[batch_indexes].to(device)

            preds = cnn.forward(X_batch)

            y_batch = y_batch.to(torch.float)
            y_batch = torch.unsqueeze(y_batch, 1)

            loss_value = cec(preds, y_batch)
            loss_value.backward()

            optimazer.step()
        with torch.no_grad():
            test_preds = cnn.forward(val_x)
            test_loss_history.append(cec(test_preds, val_y).data.cpu())
            accuracy = ((test_preds - val_y).std().mean().float().data.cpu())
            test_accuracy_history.append(accuracy)

        if accuracy <= max_accurancy:
            # best_model = copy.deepcopy(cnn)
            max_accurancy = accuracy
            checkpoint = {'state_dict': cnn.state_dict(), 'optimizer': optimazer.state_dict()}
        if epoch == 50 * i - 1:
            torch.save(checkpoint,
                       f'saves\\count_sm\\count_in_smsss_{epoch}.pt')
            max_accurancy = 1000
            i += 1
        print(epoch)
        print(accuracy)
        print('preds', test_preds[100])
        print('real', val_y[100])
        # torch.save(checkpoint, 'Schet_trekov_x_img_1000_ephoh_MSELOSS_LR4.pt')
        # torch.save(cnn, 'CLassNet.pkl')
    return test_accuracy_history, test_loss_history

if __name__ == '__main__':
    # file_train_Mat = open('D:\\networking\\data\\df\\train_Mat_by_sms_0_321_run', 'rb')
    # file_train_Mask = open('D:\\networking\\data\\df\\train_Mask_by_sms_0_321_run', 'rb')
    file_valid_Mat = open('D:\\networking\\data\\df\\test_Mat_by_sms_0_321_run', 'rb')
    file_valid_Mask = open('D:\\networking\\data\\df\\test_Mask_by_sms_0_321_run', 'rb')

    # train_Mat_x = np.load(file_train_Mat)
    # train_Mask = np.load(file_train_Mask)
    val_Mat_x = np.load(file_valid_Mat)
    val_Mask = np.load(file_valid_Mask)

    train_Mat_x = torch.load('D:\\networking\\data\\df\\tr_in_big.pt')
    train_Mask = torch.load('D:\\networking\\data\\df\\tr_out_big.pt')
    print(type(train_Mask))

    # file_train_Mat.close()
    # file_train_Mask.close()
    file_valid_Mat.close()
    file_valid_Mask.close()
    #
    # train_Mat_x = torch.from_numpy(np.asarray(train_Mat_x))
    # train_Mask = torch.from_numpy(np.asarray(train_Mask))
    val_Mat_x = torch.from_numpy(np.asarray(val_Mat_x))
    val_Mask = torch.from_numpy(np.asarray(val_Mask))

    train_Mask_treck = train_Mask[:, 2]
    val_Mask_treck = val_Mask[:, 2]
    # print(train_Mask_treck)

    device = torch.device('cuda')
    # print(train_Mask[252][1])
    # print(train_Mask[252])
    #
    # ax = sns.heatmap(train_Mat_x[252][0][0], annot=True)
    # plt.show()

    a, l = train(numb_epoch=250, device=device, train_x=train_Mat_x, train_y=train_Mask_treck, val_x=val_Mat_x,
                 val_y=val_Mask_treck)

    plt.plot(a)
    plt.show()
    plt.plot(l)
    plt.show()