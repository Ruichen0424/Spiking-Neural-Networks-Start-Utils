import os
import numpy as np
from scipy.io import loadmat, savemat


def pad_image(image, padding, mode='constant', pad_value=0):
    if padding > 0:
        padded_image = np.pad(image, ((padding, padding), (padding, padding)), mode, constant_values=pad_value)
    else:
        padded_image = image
    return padded_image


def unpad_image(image, padding):
    if padding > 0:
        unpadded_image = image[padding:-padding, padding:-padding]
    else:
        unpadded_image = image
    return unpadded_image


def get_index(img_size, stride, padding, Ker_H, Ker_W):

    H_in = img_size[0]
    W_in = img_size[1]
    H_out = int((H_in - Ker_H + 2 * padding) / stride) + 1
    W_out = int((W_in - Ker_W + 2 * padding) / stride) + 1

    img = np.zeros((H_in, W_in))
    img_p = pad_image(img, padding)
    Ker_index = (np.array(range(0, Ker_H * Ker_W)) + 1).reshape(Ker_H, Ker_W)
    out_index = np.zeros((H_in*W_in, H_out*W_out))
    
    count = 0
    for row in range(0, img_p.shape[0]+1-Ker_H, stride):
        for col in range(0, img_p.shape[1]+1-Ker_W, stride):
            img_temp = img_p.copy()
            img_temp[row:row+Ker_H, col:col+Ker_W] = Ker_index
            img_temp = unpad_image(img_temp, padding)
            img_temp = img_temp.reshape(-1)
            out_index[:, count] = img_temp
            count += 1
    assert count == H_out * W_out, 'Wrong!'
    print(f'out_index is generated.')

    return out_index.astype(int)


def conv2d2matrix_singlechannel(weight, out_index):
    
    assert len(weight.shape) == 2, f'weight.shape must be [Ker_H, Ker_W], but got {weight.shape}'

    weight = np.concatenate((np.array([0]).reshape(-1, 1), weight.reshape(-1, 1)), 0).reshape(-1)
    matrix = weight[out_index]

    return matrix.astype(np.float32)


def conv2d2matrix_mulcin(weight, out_index):

    assert len(weight.shape) == 3, f'weight.shape must be [C_in, Ker_H, Ker_W], but got {weight.shape}'

    C_in = weight.shape[0]
    matrix = []
    [matrix.append(conv2d2matrix_singlechannel(weight[i], out_index)) for i in range(C_in)]
    matrix = np.concatenate(matrix, 0)

    return matrix


def conv2d2matrix(weight, img_size, stride, padding):
    '''
        将卷积操作的参数矩阵w转换为w_2d二维矩阵的形式, 使得卷积能够由矩阵乘法实现。
        y = Conv2d(w, x)   =>   y_vector = x_vector * w_2d, 其中x_vector和y_vector是x和y的一维展开
        w_2d的物理意义: 行为输入, 即输出神经元的输入突触; 列为输出, 即输出神经元

        参数:
            weight: 卷积操作的参数矩阵w, shape为[C_out, C_in, Ker_H, Ker_W]
            img_size: 输入图片的尺寸, shape为[H_in, W_in]
            stride: 卷积操作的步长
            padding: 卷积操作的padding

        返回:
            matrix: 卷积操作的参数矩阵w转换为w_2d二维矩阵的形式,
                    shape为[C_in * H_in * W_in, C_out * H_out * W_out]

        用法：
            matrix = conv2d2matrix(weight, img_size=[32, 32], stride=1, padding=1)
    '''
    assert len(weight.shape) == 4, f'weight.shape must be [C_out, C_in, Ker_H, Ker_W], but got {weight.shape}'

    C_out = weight.shape[0]
    Ker_H = weight.shape[2]
    Ker_W = weight.shape[3]
    
    out_index = get_index(img_size, stride, padding, Ker_H, Ker_W)
    matrix = []
    [matrix.append(conv2d2matrix_mulcin(weight[i], out_index)) for i in range(C_out)]
    matrix = np.concatenate(matrix, 1)
    print(f'The matrix is generated.')

    return matrix.astype(np.float32)


def linear2matrix(weight):
    '''
        将全连接操作的参数矩阵w转换为w_2d二维矩阵的形式, 使得全连接能够由矩阵乘法实现。
        由于w本身就为二维矩阵, 因此只需将w转置返回即可 (转置即保证物理意义相同)
        w_2d的物理意义: 行为输入, 即输出神经元的输入突触; 列为输出, 即输出神经元

        参数:
            weight: 全连接操作的参数矩阵w, shape为[C_out, C_in]
        
        返回:
            matrix: 全连接操作的参数矩阵w转换为w_2d二维矩阵的形式, shape为[C_in, C_out]

        用法：
            matrix = linear2matrix(weight)
    '''
    assert len(weight.shape) == 2, f'weight.shape must be [C_out, C_in], but got {weight.shape}'
    matrix = weight.T

    return matrix.astype(np.float32)


def maxpool2d2matrix(img_size, ker_size, stride, padding):
    '''
        将最大池化操作转换为矩阵乘法实现。
        由于脉冲神经网络的输出仅为0/1, 显然使用参数全为'1'的卷积操作即可实现对应大小的最大池化操作
        w_2d的物理意义: 行为输入, 即输出神经元的输入突触; 列为输出, 即输出神经元

        参数:
            img_size: 输入图片的尺寸, shape为[C, H_in, W_in]
            ker_size: 最大池化操作的核尺寸, shape为[Ker_H, Ker_W]
            stride: 最大池化操作的步长
            padding: 最大池化操作的padding

        返回:
            matrix: 最大池化操作的参数矩阵w转换为w_2d二维矩阵的形式,
                    shape为[C_in, H_out, W_out]

        用法：
            matrix = maxpool2d2matrix(img_size=[3, 32, 32], ker_size=[2, 2], stride=2, padding=0)
    '''
    assert len(img_size) == 3, f'img_size must be [C, H, W], but got {img_size}'

    C_in = img_size[0]
    img_size = img_size[1:]
    Ker_H = ker_size[0]
    Ker_W = ker_size[1]

    equ_weight = np.zeros((C_in, C_in, Ker_H, Ker_W)).astype(np.float32)
    for i in range(C_in):
        equ_weight[i, i, :, :] = np.ones((Ker_H, Ker_W)).astype(np.float32)

    matrix = conv2d2matrix(equ_weight, img_size, stride, padding)

    return matrix.astype(np.float32)


def rewrite_mat_with_2d(file_name, param):
    '''
        将file_name(.mat)文件中的Labels转换为Labels_2d, 并在源文件中写入键值为Labels_2d的字典
        用于将原始的Labels转换为2d矩阵形式, 使得脉冲神经网络能够直接使用矩阵乘法实现卷积操作

        参数:
            file_name: .mat文件的路径
            param: 卷积操作的参数, 为一个列表, 其中第一个元素为img_size, 第二个元素为stride, 第三个元素为padding
                   详见maxpool2d2matrix函数的参数
    '''
    data = loadmat(file_name)
    if 'Labels' in data:
        weight = data['Labels']
        print(f'weight.shape: {weight.shape}')

        if len(weight.shape) == 2:
            weight_2d = linear2matrix(weight)
        elif len(weight.shape) == 4:
            weight_2d = conv2d2matrix(weight, param[0], param[1], param[2])
        else:
            raise ValueError(f'weight.shape must be [C_out, C_in, Ker_H, Ker_W] or [C_out, C_in], but got {weight.shape}')
        
        print(f'weight_2d.shape: {weight_2d.shape}')
        data['Labels_2d'] = weight_2d.astype(np.int32)
        savemat(file_name, data)
        print(f'Labels_2d is saved in {file_name}\n')
    else:
        print(f'No Labels in {file_name}\n')


def get_all_file_names(folder_path, tag='weight'):
    file_names = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if tag in file:
                file_names.append(file)
    return sorted(file_names)


def rewrite_mat_with_folder(folder_path, param=None, tag='weight'):
    '''
        将文件夹folder_path中所有包含字符tag的文件进行rewrite_mat_with_2d转换

        参数：
            folder_path: .mat文件所在文件夹
            param: 为一个list, 要求len(all_files) == len(param), 每一个param格式参见rewrite_mat_with_2d
            tag: folder_path中需要处理的文件包含名称
    '''
    all_files = get_all_file_names(folder_path, tag=tag)
    assert len(all_files) == len(param), f'len(all_files) != len(param)'

    for i in range(len(all_files)):
        file_name = all_files[i]
        print(file_name)
        rewrite_mat_with_2d(folder_path+file_name, param[i])