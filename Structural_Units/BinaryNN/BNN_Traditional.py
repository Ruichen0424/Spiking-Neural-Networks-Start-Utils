import torch
import torch.nn as nn
import torch.nn.functional as F

def SigmoidFunc(x, alpha=3.):   # 带参数的Sigmoid函数
    return torch.sigmoid(alpha * x)

def SigmoidGFunc(x, alpha=3.):   # 带参数Sigmoid函数的导数
    return alpha * SigmoidFunc(x, alpha) * (1 - SigmoidFunc(x, alpha))

def BinaryForFunc(x):   # 权值量化前向函数
    return torch.sign(x)

def BinaryWBackFunc(x, alpha=3.):   # 权值量化反向梯度函数
    return 2 * SigmoidGFunc(x, alpha)
    
class BinaryWeight(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return BinaryForFunc(input)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output * BinaryWBackFunc(input, alpha=3.)
    
class BinaryLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False):
        super(BinaryLinear, self).__init__(in_features, out_features, bias)
        
    def forward(self, x):

        w = self.weight
        bw = BinaryWeight.apply(w)
        # scaling_factor = torch.mean(abs(w),dim=1,keepdim=True)
        # scaling_factor = scaling_factor.detach()
        # bw = scaling_factor * BinaryWeight.apply(w)
        
        return F.linear(x, bw, self.bias)
    
class BinaryConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False):
        super(BinaryConv2d, self).__init__(in_channels, out_channels, 
            kernel_size, stride, padding, dilation, groups, bias)
        
    def forward(self, x):
        
        w = self.weight
        bw = BinaryWeight.apply(w)
        # scaling_factor = torch.mean(torch.mean(torch.mean(abs(w),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
        # scaling_factor = scaling_factor.detach()
        # bw = scaling_factor * BinaryWeight.apply(w)
    
        return F.conv2d(x, bw, self.bias, self.stride,
                    self.padding, self.dilation, self.groups)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    print('测试：')
    x = torch.linspace(-5, 5, 501)
    plt.plot(x, SigmoidFunc(x), label = f'SigmoidFunc')
    plt.plot(x, SigmoidGFunc(x), label = f'SigmoidGFunc')
    plt.plot(x, BinaryForFunc(x), label = f'BinaryForFunc')
    plt.plot(x, BinaryWBackFunc(x), label = f'BinaryWBackFunc')
    plt.legend()
    plt.show()