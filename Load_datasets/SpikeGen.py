import torch


class PoissonEncoder2D(torch.nn.Module):
    '''
        对图片进行泊松编码预处理，将图片转换为脉冲编码的形式
        [C, H, W] -> [T, C, H, W]

        用法：
            transform = transforms.Compose([
                transforms.ToTensor(),
                PoissonEncoder(T=Timestep),          # Timestep为编码时间步长
                ])
    '''
    def __init__(self, T=4):
        super().__init__()
        self.T = T

    def forward(self, img):
        img = img.repeat(self.T, 1, 1, 1)
        coding_img = torch.rand_like(img).le(img).to(img)
        return coding_img
    
class PoissonEncoder1D(torch.nn.Module):
    '''
        对图片进行泊松编码预处理，将图片转换为脉冲编码的形式
        [C, H] -> [T, C, H]

        用法：
            transform = transforms.Compose([
                transforms.ToTensor(),
                PoissonEncoder(T=Timestep),          # Timestep为编码时间步长
                ])
    '''
    def __init__(self, T=4):
        super().__init__()
        self.T = T

    def forward(self, img):
        img = img.repeat(self.T, 1, 1)
        coding_img = torch.rand_like(img).le(img).to(img)
        return coding_img
    
class PoissonEncoder0D(torch.nn.Module):
    '''
        对图片进行泊松编码预处理，将图片转换为脉冲编码的形式
        [C] -> [T, C]

        用法：
            transform = transforms.Compose([
                transforms.ToTensor(),
                PoissonEncoder(T=Timestep),          # Timestep为编码时间步长
                ])
    '''
    def __init__(self, T=4):
        super().__init__()
        self.T = T

    def forward(self, img):
        img = img.repeat(self.T, 1)
        coding_img = torch.rand_like(img).le(img).to(img)
        return coding_img