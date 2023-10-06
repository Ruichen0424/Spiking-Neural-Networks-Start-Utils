import os
import torch
from scipy.io import savemat
from spikingjelly.activation_based import neuron, monitor, functional

def save_N_sv(net, N_sv_path="./N_sv/", dataset=None, num_example=2, neuron_type=neuron.IFNode, Timestep=None, data=None):
    '''
        记录网络net中每一层类型为neuron_type的神经元在data数据上每个时间的膜电位和脉冲序列, 并将结果保存在N_sv_path文件夹下。
        其中N_s.mat记录了脉冲序列, N_v.mat记录了膜电位序列, 文件保存的格式均为 [T, N, *]。
        Input_data.mat记录了输入数据data, Output_data.mat记录了网络net在输入数据data上的输出。

        输入数据data可由参数data直接给定, 其为None时将从dataset中取出前num_example个样本拼接起来作为data。
        data的数据格式为 [T, N, *] 或 [N, *], dataset的数据格式为 [T, *] 或 [*]
        为 [N, *] 或 [*] 时, 若给定参数Timestep, 则将data.shape变换为 [T, N, *]; 若Timestep=None, 则维持data.shape=[N, *]。

        参数:
            net: 网络
            N_sv_path: 保存N_s.mat, N_v.mat, Input_data.mat, Output_data.mat的文件夹路径
            dataset: 数据集, 用于取出输入数据data, 类型为Tensor, 当data为None时有效
            num_example: 从dataset中取出的样本数, 当data为None时有效
            neuron_type: 神经元类型
            Timestep: 输入数据data的时间步数, 当输入数据格式为 [N, *] 或 [*] 时有效
            data: 输入数据data, 类型为Tensor, 为None时必须指定dataset
        
        用法：
            save_N_sv(net, "./N_sv/", neuron_type=neuron.IFNode, data=data)
        或  save_N_sv(net, "./N_sv/", neuron_type=neuron.IFNode, data=data, Timestep=Timestep)
        或  save_N_sv(net, "./N_sv/", dataset=dataset, num_example=2, neuron_type=neuron.IFNode)
        或  save_N_sv(net, "./N_sv/", dataset=dataset, num_example=2, neuron_type=neuron.IFNode, Timestep=Timestep)
    '''
    if not os.path.exists(N_sv_path):
        os.makedirs(N_sv_path)

    assert not(dataset==None and data==None), "dataset and data can't be None at the same time." 
    if data==None:
        data = torch.cat([dataset[i][0].unsqueeze(0) for i in range(num_example)], 0)
    assert len(data.shape)==4 or len(data.shape)==5, "data.shape must be [T, N, C, H, W] or [N, C, H, W]."

    if len(data.shape)==4:
        if Timestep is not None:
            data = data.repeat(Timestep, 1, 1, 1, 1)
    mdic_inputdata = {}
    mdic_inputdata['input_data'] = data.cpu().numpy()
    savemat(N_sv_path + "Input_data.mat", mdic_inputdata)
    print(f'data.shape = {data.shape}, data is saved.')

    if torch.cuda.is_available():
        net = net.cuda()
        data = data.cuda()
    else:
        net = net.cpu()
        data = data.cpu()
    net.eval()
    functional.set_step_mode(net, step_mode='m')
    for m in net.modules():
        if isinstance(m, neuron_type):
            m.store_v_seq = True

    mdic_v = {}
    mdic_s = {}
    spike_seq_monitor = monitor.OutputMonitor(net, neuron_type)
    v_seq_monitor = monitor.AttributeMonitor('v_seq', pre_forward=False, net=net, instance=neuron_type)
    functional.reset_net(net)
    with torch.no_grad():
        output = net(data)
    functional.reset_net(net)
    mdic_outputdata = {}
    mdic_outputdata['output_data'] = output.cpu().numpy()
    savemat(N_sv_path + "Output_data.mat", mdic_outputdata)
    print(f'output.shape = {output.shape}, output is saved.\n')

    for i in range(len(spike_seq_monitor.records)):
        print(f'{spike_seq_monitor.monitored_layers[i]}.shape = {spike_seq_monitor.records[i].shape}, s and v are saved.')       # [T, N, *]
        mdic_s[spike_seq_monitor.monitored_layers[i].replace('.', '_') + '_s'] = spike_seq_monitor.records[i].detach().cpu().numpy()
        mdic_v[v_seq_monitor.monitored_layers[i].replace('.', '_') + '_v'] = v_seq_monitor.records[i].detach().cpu().numpy()
    spike_seq_monitor.clear_recorded_data()
    spike_seq_monitor.remove_hooks()
    v_seq_monitor.clear_recorded_data()
    v_seq_monitor.remove_hooks()
    savemat(N_sv_path + "N_s.mat", mdic_s)
    savemat(N_sv_path + "N_v.mat", mdic_v)