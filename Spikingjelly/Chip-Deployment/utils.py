from Test_fr import *
from Net_Kmeans_with_Quantization import *
from Conv2d2Matrix import *
from Save_N_sv import *

from spikingjelly.activation_based import neuron


def all_hardware_test(net, param, data, dataloader, neuron_type=neuron.IFNode, n_clusters=15, bit=8, param_path='./Param_datas/', N_sv_path='./N_sv/'):
    '''
        用法: all_hardware_test(net, param, Test_data, test_data_loader, neuron_type=neuron.IFNode, n_clusters=15, bit=8)
    '''
    _, allN_fr, all_N_name, acc = test_fr(net, dataloader, neuron_type=neuron.IFNode, Output=True, Mean=True)
    print(f'acc={acc}')
    plot_fr(all_N_name, allN_fr, tag='sn', ylim=(0, 1))

    net_kmean_q,_,_,_,_,_ = net_kmeans_with_quantization(net, n_clusters=n_clusters, bit=bit, data_save_path=param_path)
    _, allN_fr, all_N_name, acc = test_fr(net_kmean_q, dataloader, neuron_type=neuron.IFNode, Output=True, Mean=True)
    print(f'After kmeans_with_quantization, acc={acc}')
    plot_fr(all_N_name, allN_fr, tag='sn', ylim=(0, 1))

    rewrite_mat_with_folder(param_path, param=param)

    save_N_sv(net_kmean_q, N_sv_path, dataset=data, num_example=2, neuron_type=neuron.IFNode)