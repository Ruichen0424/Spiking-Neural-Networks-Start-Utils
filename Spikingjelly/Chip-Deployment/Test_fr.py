import torch
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from spikingjelly.activation_based import neuron, functional, monitor

def list_sum(A:list, B:list):
    if len(A) == len(B):
        C = []
        for i in range(len(A)):
            C.append(A[i] + B[i])
    else:
         C = B
    return C

def cal_firing_rate(s_seq: torch.Tensor):   # s_seq.shape = [T, N, *]
    return s_seq.flatten(2).mean(2).sum(1)

def cal_firing_var(s_seq: torch.Tensor):   # s_seq.shape = [T, N, *]
    return s_seq.flatten(2).var(2).sum(1)

def plot_fr(all_N_name, allN_fr, tag='sn', ylim=None, xlable='Layer', ylable='Firing rate', title='Firing rate with different layers',
            lable_size=12, title_size=14, ticks_size=10, small_ticks_size=1.5, fig_size=(4, 2), dpi=100, grid=True):
    plot_data = [allN_fr[i] for i in range(len(allN_fr)) if tag in all_N_name[i]]
    plot_data = torch.stack(plot_data).numpy()

    plt.figure(figsize=fig_size, dpi=dpi)
    plt.tick_params('both', which='major', width=small_ticks_size, direction='in')
    x = np.linspace(1, plot_data.shape[0], plot_data.shape[0])
    plt.plot(x, plot_data)

    plt.xlim((1, plot_data.shape[0]))
    if ylim is not None:
        plt.ylim(ylim)
    plt.xlabel(xlable, fontsize=lable_size)
    plt.ylabel(ylable, fontsize=lable_size)
    plt.title(title, fontsize=title_size)
    plt.xticks(fontsize=ticks_size)
    plt.yticks(fontsize=ticks_size)
    plt.gca().spines['top'].set_linewidth(small_ticks_size)
    plt.gca().spines['bottom'].set_linewidth(small_ticks_size)
    plt.gca().spines['left'].set_linewidth(small_ticks_size)
    plt.gca().spines['right'].set_linewidth(small_ticks_size)
    plt.grid(grid)
    plt.show()

    return plot_data

def test_fr(net, dataloader, neuron_type=neuron.IFNode, Timestep=None, Output=True, Mean=True, Test_acc=True):
    '''
        获得网络net每一层类型为neuron_type的神经元在dataloader上的平均输入/输出的均值/方差。

        参数：
            net: 网络
            dataloader: 用于测试的数据集, 输出格式为 [N, T, *] 或 [N, *]:
                        当为 [N, *] 时, 若提供Timestep, 则将数据处理为 [T, N, *]; 若Timestep为None, 则不对数据进行处理，保持 [N, *]
            neuron_type: 神经元类型
            Timestep: 表示数据的时间步长, 当dataloader输出为 [N, *] 时, 将数据处理为 [T, N, *]。 若为None时不处理
            Output: 为True时统计输出量, 为Flase时统计输入量
            Mean: 为True时统计均值, 为False时统计方差
            Test_acc: 为True时统计准确率, 为False时不统计

        返回值：
            allN_allT_fr: tensor, 每一层神经元在每一时间步的输入/输出的均值/方差, allN_allT_fr.shape = [N_l, T]
                        N_l 为net中类型为neuron_type的神经元的总层数
            allN_fr: tensor, 每一层神经元在所有时间步的平均输入/输出的均值/方差, allN_fr.shape = [N_l]
            all_N_name: list, 每一层神经元的网络名称
            acc: float, 网络net在dataloader上的准确率, 若Test_acc为Flase时返回0.

        用法：
            allN_allT_fr, allN_fr, all_N_name, acc = test_fr_output(net, dataloader, neuron.IFNode)
        或  allN_allT_fr, allN_fr, all_N_name, acc = test_fr_output(net, dataloader, neuron.IFNode, Timestep)
    '''
    net.eval()
    functional.set_step_mode(net, step_mode='m')
    functional.reset_net(net)
    if torch.cuda.is_available():
        net = net.cuda()

    if Output:
        if Mean:
            fr_monitor = monitor.OutputMonitor(net, neuron_type, cal_firing_rate)
        else:
            fr_monitor = monitor.OutputMonitor(net, neuron_type, cal_firing_var)
    else:
        if Mean:
            fr_monitor = monitor.InputMonitor(net, neuron_type, cal_firing_rate)
        else:
            fr_monitor = monitor.InputMonitor(net, neuron_type, cal_firing_var)

    allN_allT_fr = []
    allN_fr = []
    num_sample = 0
    acc = 0

    with torch.no_grad():
        for img, label in tqdm(dataloader):
            if torch.cuda.is_available():
                img = img.cuda()
                label = label.cuda()
            if len(img.shape) == 4:
                if Timestep is not None:
                    img = img.repeat(Timestep, 1, 1, 1, 1)      # img.shape = [T, N, C, H, W]

            out_fr = net(img)
            if Test_acc:
                acc += (out_fr.argmax(1) == label).float().sum().item()
            num_sample += label.numel()
            allN_allT_fr = list_sum(allN_allT_fr, fr_monitor.records)
            fr_monitor.clear_recorded_data()
            functional.reset_net(net)
            
    all_N_name = fr_monitor.monitored_layers
    fr_monitor.remove_hooks()
    allN_allT_fr = torch.cat([allN_allT_fr[i].unsqueeze(0)/num_sample for i in range(len(allN_allT_fr))])
    allN_fr = allN_allT_fr.mean(1)
    acc /= num_sample
    
    return allN_allT_fr.cpu(), allN_fr.cpu(), all_N_name, acc
