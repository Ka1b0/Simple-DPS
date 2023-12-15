import random
import torch
import kornia
import torch.nn.functional as F
from functools import partial
from utils import set_seed


__OPERATOR__ = {}

def register_operator(name):
    def wrapper(cls):
        __OPERATOR__[name] = cls
        return cls
    return wrapper


def get_operator(config):
    name = config['operator']['name']
    return __OPERATOR__[name](config)

def get_noise_func(config):
    if config['noise']['name'] == 'gaussian':
        noise_func = partial(gaussian_noise, sigma=config['noise']['sigma'])
    else: noise_func = partial(possion_noise, rate=config['noise']['rate'])
    return noise_func

class Operator(torch.nn.Module):

    def forward(self, x):
        pass

    def preprocess(self, x, seed=0):
        pass

    def transpose(self, x):
        return x
    
    def ortho_project(self, x):
        return x - self.transpose(self(x))

    def project(self, x, y):
        if self.linear:
            return self.ortho_project(x) + self.forward(y)
        else: 
            return x + y - self.forward(x) 


@register_operator(name='super-resolution')
class SuperResolution(Operator):

    def __init__(self, config):
        super().__init__()
        self.linear = True
        self.scale_factor = config['operator']['scale_factor']
    
    def forward(self, x):
        return F.interpolate(x, scale_factor=1/self.scale_factor, mode='bilinear', antialias=True)
    
    def transpose(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', antialias=True)
    
    def project(self, x, y):
        return x - self.transpose(self.forward(x)) + self.transpose(y)

@register_operator(name='gaussian-blur')
class GaussianBlur(Operator):

    def __init__(self, config):
        super().__init__()
        self.linear = True
        self.kernel_size = config['operator']['gaussian_kernel']
        self.var = config['operator']['gaussian_var']
    
    def forward(self, x):
        return kornia.filters.gaussian_blur2d(x, kernel_size=self.kernel_size, sigma=(self.var, self.var))


@register_operator(name='box-inpaint')
class Inpainting(Operator):
    def __init__(self, config):
        super().__init__()
        self.linear = True
        self.width = config['operator']['box_width']

    def preprocess(self, x, seed):
        set_seed(seed)
        n, c, h, w = x.shape
        mask = torch.ones_like(x)
        x = random.randint(0, w - self.width)
        y = random.randint(0, h - self.width)
        mask[:,:,y:y + self.width,x:x + self.width] = 0
        self.mask = mask

    def forward(self, x):
        return x * self.mask


@register_operator(name='random-inpaint')
class Inpainting(Operator):
    def __init__(self, config):
        super().__init__()
        self.linear = True
        self.rate = config['operator']['random_rate']

    def preprocess(self, x, seed):
        set_seed(seed)
        n, c, h, w = x.shape
        v = torch.rand_like(x).sum(dim=1, keepdim=True)
        threshold = torch.kthvalue(v.reshape(n, -1), int(h * w * self.rate), dim=1)[0]
        mask = v > threshold.reshape(n, 1, 1, 1)
        self.mask = mask

    def forward(self, x):
        return x * self.mask

@register_operator(name='motion-blur')
class MotionBlur(Operator):

    def __init__(self, config):
        super().__init__()
        self.linear = True
        self.kernel_size = config['operator']['motion_kernel']
        self.intensity = config['operator']['motion_direction']
    
    def forward(self, x):
        return kornia.filters.motion_blur(x, kernel_size=self.kernel_size, angle=0, direction=self.intensity)


@register_operator(name='gamma')
class Gamma(Operator):

    def __init__(self, config):
        super().__init__()
        self.linear = False
        self.gamma = config['operator']['gamma']
        self.gain = config['operator']['gain']
    
    def forward(self, x):
        x = (x + 1) / 2
        x = kornia.enhance.adjust_gamma(x, self.gamma, self.gain)
        return x * 2 - 1


@register_operator(name='sobel')
class Sobel(Operator):

    def __init__(self, config):
        super().__init__()
        self.linear = False
    
    def forward(self, x):
        x = (x + 1) / 2
        x = kornia.filters.sobel(x) ** 0.25
        return x * 2 - 1

def gaussian_noise(x, sigma):
    return x + torch.randn_like(x) * sigma

def possion_noise(x, rate):
    rate *= 255.0
    x = ((x + 1.0) / 2.0).clamp(0, 1)
    x = torch.poisson(x * rate) / rate
    return (x * 2.0 - 1.0).clamp(-1, 1)



