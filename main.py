from matplotlib import image
import matplotlib.pyplot as plt
import numpy as np

'''
getting the image from user
'''


def get_image():
    image_data = image.imread('noisy.jpg')
    return image_data


if __name__ == '__main__':
    pic_data = get_image()
    size = pic_data.shape
    '''In this section we break the noisy matrix into three color dimensions'''
    first_dim = np.zeros((size[0], size[1]), dtype='uint8')
    second_dim = np.zeros((size[0], size[1]), dtype='uint8')
    third_dim = np.zeros((size[0], size[1]), dtype='uint8')
    for i in range(size[0]):
        for j in range(size[1]):
            first_dim[i][j] = pic_data[i][j][0]
            second_dim[i][j] = pic_data[i][j][1]
            third_dim[i][j] = pic_data[i][j][2]
    '''In this section we break the matrixes into USV matrixes'''
    UG, SG, VG = np.linalg.svd(first_dim)
    UR, SR, VR = np.linalg.svd(second_dim)
    UB, SB, VB = np.linalg.svd(third_dim)
    RG = np.zeros((UG.shape[1], VG.shape[0]), dtype='float64')
    RR = np.zeros((UR.shape[1], VR.shape[0]), dtype='float64')
    RB = np.zeros((UB.shape[1], VB.shape[0]), dtype='float64')
    '''In this section we delete the noises by making diameter after k = 18'''
    changed_G = np.diag(SG[:18])
    changed_R = np.diag(SR[:18])
    changed_B = np.diag(SB[:18])
    for i in range(RG.shape[0]):
        for j in range(RG.shape[1]):
            if i == j and i < 18:
                RG[i][j] = changed_G[i][j]
                RR[i][j] = changed_R[i][j]
                RB[i][j] = changed_B[i][j]
    Main_Matrix1 = UG.dot(RG).dot(VG)
    Main_Matrix2 = UR.dot(RR).dot(VR)
    Main_Matrix3 = UB.dot(RB).dot(VB)
    Main_Matrix = np.zeros((size[0], size[1], 3), dtype='uint8')
    '''make the matrix with new S again'''
    for i in range(Main_Matrix1.shape[0]):
        for j in range(Main_Matrix1.shape[1]):
            Main_Matrix[i][j][0] = Main_Matrix1[i][j]
            Main_Matrix[i][j][1] = Main_Matrix2[i][j]
            Main_Matrix[i][j][2] = Main_Matrix3[i][j]
    plt.imshow(Main_Matrix)
    plt.show()
    plt.imsave('path.jpeg', Main_Matrix)

