import seaborn as sns; sns.set()
import numpy as np
import random



def hex2binary(line):
    scale = 16
    num_of_bits = 8
    mask = np.empty((8, 256), dtype='int')
    mask_i = 0
    for i in range(8):
        h = 0
        for j in range(32):
            binary = bin(int(line[mask_i], scale))[2:].zfill(num_of_bits)
            mask_i +=1
            for k in range(8):
                mask[i][h] = binary[7-k]
                h +=1
    return mask

def file_len(file_name):
    f = open(file_name)
    lines = 0
    for line in f:
        lines += 1
    f.close()
    return  lines


if __name__ == '__main__':
    file_names = ['data\\Grs3_55_seria_13_obr\\classification\\Matr_dan_good_and_bad_X.dat',
                   'data\\Grs3_55_seria_13_obr\\classification\\Matr_dan_good_and_bad_Y.dat']

    Mat = np.ones((92177, 1, 2, 64, 256), dtype='f')
    Mask = np.ones((92177, 3), dtype='f')

    f1 = open(file_names[0])
    f2 = open(file_names[1])

    len_new = 0

    for event in range(file_len(file_names[0])):
        line_X = f1.readline().split('\t')
        line_Y = f2.readline().split('\t')
        step = 0

        for sm in range(8):
            binary_mask_X = hex2binary(line_X[3 + 256 * sm: 3 + 256 * (sm + 1)])
            binary_mask_Y = hex2binary(line_Y[3 + 256 * sm: 3 + 256 * (sm + 1)])

            for plsk in range(8):
                for strip in range(256):
                    Mat[event][0][0][step][strip] = binary_mask_X[plsk][strip]
                    Mat[event][0][1][step][strip] = binary_mask_Y[plsk][strip]
                step += 1

        Mask[event][0] = float(line_X[0])
        Mask[event][1] = float(line_X[1])
        if str(line_X[2]) == 'good':
            Mask[event][2] = 1.0
        if str(line_X[2]) == 'bad':
            Mask[event][2] = 0.0

    f1.close()
    f2.close()

    combined_list = list(zip(Mat, Mask))
    random.shuffle(combined_list)
    Mat_rand, Mask_rand = zip(*combined_list)

    train_Mat = Mat_rand[0:92177 - 10000]
    valid_Mat = Mat_rand[92177 - 10000:]

    train_Mask = Mask_rand[0:92177 - 10000]
    valid_Mask = Mask_rand[92177 - 10000:]

    f_train_Mat = open('data\\Grs3_55_seria_13_obr\\classification\\train_Mat_3d', 'wb')
    f_valid_Mat = open('data\\Grs3_55_seria_13_obr\\classification\\valid_Mat_3d', 'wb')
    f_train_Mask = open('data\\Grs3_55_seria_13_obr\\classification\\train_Mask_3d', 'wb')
    f_valid_Mask = open('data\\Grs3_55_seria_13_obr\\classification\\valid_Mask_3d', 'wb')

    np.save(f_train_Mat, train_Mat)
    np.save(f_train_Mask, train_Mask)
    np.save(f_valid_Mat, valid_Mat)
    np.save(f_valid_Mask, valid_Mask)

    f_train_Mat.close()
    f_valid_Mat.close()
    f_train_Mask.close()
    f_valid_Mask.close()