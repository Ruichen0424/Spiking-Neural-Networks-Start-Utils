import os
import copy
import torch
from scipy.io import savemat
from sklearn.cluster import KMeans
from spikingjelly.activation_based import neuron, monitor

def cul_weight(labels, centroids):                         # 由labels和centroids还原weight, labels.shape = weight.shape
    weight = torch.zeros_like(labels).float()
    for i, c in enumerate(centroids.numpy().squeeze()):
        weight[labels == i] = c.item()
    return weight

def k_means_cpu(weight, n_clusters, init='k-means++', n_init=1, max_iter=300):      # 对weight做聚类
    org_shape = weight.shape
    weight = weight.reshape(-1, 1)
    if n_clusters > weight.numel():
        n_clusters = weight.numel()

    k_means = KMeans(n_clusters=n_clusters, init=init, n_init=n_init, max_iter=max_iter)
    k_means.fit(weight)

    centroids = torch.from_numpy(k_means.cluster_centers_)
    labels = k_means.labels_
    labels = torch.from_numpy(labels.reshape(org_shape)).int()
    weight_out = cul_weight(labels, centroids)
    return centroids, labels, weight_out

def net_kmeans_with_quantization(net, data_save_path=None, n_clusters=15, init='k-means++', n_init=1, max_iter=300, bit=None, neuron_type=neuron.IFNode):
    '''
        对网络net每一层参数做聚类, 当bit不为None时, 对网络参数做整数bit-bit量化
        若data_save_path不为None, 则保存聚类后的参数至该路径
        对于每一层网络保存一个.mat文件, 文件名为该层网络的名称, 内容包含：
            Centroids/Bit_Centroids: 聚类中心/量化后的聚类中心
            Labels: 聚类标签 (下标从1开始), Labels.shape =  Weight.shape
            Weight/Bit_Weight: 聚类后的参数/量化后的聚类后的参数
        保存网络阈值至sn_threshold.mat/sn_threshold_quantization.mat, 表示每一层类型为neuron_type的神经元的阈值
        注: 网络允许最后若干层不为SNN, 若如此, sn_threshold.mat/sn_threshold_quantization.mat中不包含最后若干层的数据
            网络中所有层的bias必须为False, 网络中必须不含BatchNorm层
            聚类算法具备一定随机性, 为获得好的结果请勿固定随机数种子并多次运行取最优结果

        参数：
            net: 待聚类的网络
            data_save_path: 保存聚类后的参数的路径， 若不为None则将结果保存至该路径
            n_clusters: 聚类中心数量, 详见sklearn.cluster.KMeans
            init: 聚类中心初始化方式, 详见sklearn.cluster.KMeans
            n_init: 聚类中心初始化次数, 详见sklearn.cluster.KMeans
            max_iter: 聚类最大迭代次数, 详见sklearn.cluster.KMeans
            bit: 网络参数量化位数, 为None时不量化, 否则对参数做bit-bit整数量化
            neuron_type: 网络中神经元类型, 默认为IFNode

        输出：
            net: 聚类后的网络
            Para_name: 网络参数名称列表
            Centroids/Bit_Centroids: 聚类中心/量化后的聚类中心
            Labels: 聚类标签 (下标从1开始), Labels.shape =  Weight.shape
            Weight/Bit_Weight: 聚类后的参数/量化后的聚类后的参数
            Threshold/Bit_Threshold: 网络阈值/量化后的网络阈值

        用法：
            net, Para_name, Centroids, Labels, Weight, Threshold
              = net_kmeans_with_quantization(net, data_save_path, n_clusters, init, n_init, max_iter, None, neuron_type)
        或  net, Para_name, Bit_Centroids, Labels, Bit_Weight, Bit_Threshold
                = net_kmeans_with_quantization(net, data_save_path, n_clusters, init, n_init, max_iter, bit, neuron_type)
    '''
    net = copy.deepcopy(net.cpu())
    Centroids = []
    Labels = []
    Para_name = []
    Weight = []
    Threshold = []

    fr_monitor = monitor.OutputMonitor(net, neuron_type)
    all_N_name = fr_monitor.monitored_layers
    all_N_name = [all_N_name[i].replace('.', '_') for i in range(len(all_N_name))]
    fr_monitor.remove_hooks()

    for name, param in net.named_parameters():          # 聚类所有参数
        centroids, labels, weight = k_means_cpu(param.data, n_clusters, init, n_init, max_iter)
        param.data = weight
        Centroids.append(centroids)
        Labels.append(labels)
        Para_name.append(name.replace('.', '_'))
        Weight.append(weight)
        print(f'Layer: {name} Kmeans Done!')
    
    for module in net.modules():                        # 读取所有neuron_type层阈值
        if isinstance(module, neuron_type):
            Threshold.append(module.v_threshold)
    print('All Kmeans Done!')

    if bit is not None:                                 # bit不为None时量化网络参数
        print('\nStart quantifying the network.')
        Bit_Centroids = []
        Bit_Weight = []
        Bit_Threshold = []

        count = 0
        radio = 2 ** (bit - 1) - 1
        for name, param in net.named_parameters():      # 量化所有参数
            threshold = radio / torch.max(torch.abs(Centroids[count]))
            centroids = torch.round(Centroids[count] * threshold)
            Bit_Threshold.append(threshold)
            Bit_Centroids.append(centroids)

            weight = cul_weight(Labels[count], centroids)
            param.data = weight
            Bit_Weight.append(weight)
            count += 1

        assert count == len(Centroids), "The number of Centroids is not equal to the number of parameters!"

        count = 0
        for module in net.modules():                    # 配置网络量化后所有neuron_type层的等效阈值
            if isinstance(module, neuron_type):
                Bit_Threshold[count] = torch.round(Bit_Threshold[count] * module.v_threshold)
                module.v_threshold = Bit_Threshold[count]
                count += 1
        
        print('Quantization Done!')

    if data_save_path is not None:                      # 保存聚类后的参数

        print('\nStart saving data.')
        if not os.path.exists(data_save_path):
            os.makedirs(data_save_path)

        torch.save(net.state_dict(), data_save_path + f'net_{n_clusters}_clusters_{bit}_bit.h5')

        if bit is None:
            mdic_threshold = {}
            for i in range(len(Para_name)):
                mdic_weight = {}
                mdic_weight['Centroids'] = Centroids[i].numpy()
                mdic_weight['Labels'] = (Labels[i] + 1).numpy()
                mdic_weight['Weight'] = Weight[i].numpy()
                savemat(data_save_path + Para_name[i] + f'_{n_clusters}_clusters_{bit}_bit.mat', mdic_weight)

            mdic_threshold = {f'{all_N_name[i]}': Threshold[i] for i in range(len(all_N_name))}
            savemat(data_save_path + f'sn_threshold_{n_clusters}_clusters_{bit}_bit.mat', mdic_threshold)
        
        else:
            mdic_threshold = {}
            for i in range(len(Para_name)):
                mdic_weight = {}
                mdic_weight['Bit_Centroids'] = Bit_Centroids[i].numpy()
                mdic_weight['Labels'] = (Labels[i] + 1).numpy()
                mdic_weight['Bit_Weight'] = Bit_Weight[i].numpy()
                savemat(data_save_path + Para_name[i] + f'_{n_clusters}_clusters_{bit}_bit.mat', mdic_weight)

            mdic_threshold = {f'{all_N_name[i]}': Bit_Threshold[i] for i in range(len(all_N_name))}
            savemat(data_save_path + f'sn_threshold_{n_clusters}_clusters_{bit}_bit.mat', mdic_threshold)
        print(f'Save data to {data_save_path} Done!')

    if torch.cuda.is_available():
        net = net.cuda()

    if bit is None:
        return net, Para_name, Centroids, Labels, Weight, Threshold
    else:
        return net, Para_name, Bit_Centroids, Labels, Bit_Weight, Bit_Threshold