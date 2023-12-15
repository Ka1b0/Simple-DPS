import torch

__CONDITIONER__ = {}

def register_conditioner(name):
    def wrapper(cls):
        __CONDITIONER__[name] = cls
        return cls
    return wrapper


def get_conditioner(config, operator, noise_func):
    name = config['conditioner']['name']
    return __CONDITIONER__[name](config, operator, noise_func)


class Conditioner(torch.nn.Module):
    def __init__(self, config, operator, noise_func):
        super().__init__()
        self.config = config
        self.operator = operator
        self.noise_func = noise_func
    
    def grad_norm(self, x_t, x_0_hat, y_n):
        difference = y_n - self.operator.forward(x_0_hat)
        norm = torch.linalg.norm(difference)
        if self.config['noise']['name'] == 'poisson':
            norm = (norm / measurement.abs()).mean()
        grad = torch.autograd.grad(outputs=norm, inputs=x_t)[0]
        return grad, norm


@register_conditioner(name='proj')
class Projection(Conditioner):
    def forward(self, x_0_hat, x_tp, x_t, y_n, y_tn):
        return self.operator.project(x_tp, y_tn)


@register_conditioner(name='mcg')
class ManifoldConstraintGradient(Conditioner):
    def __init__(self, config, operator, noiser):
        super().__init__(config, operator, noiser)
        self.scale = config['conditioner']['scale']
        
    def forward(self, x_0_hat, x_tp, x_t, y_n, y_tn):
        grad, norm = self.grad_norm(x_t, x_0_hat, y_n)
        x_tp -= grad * self.scale
        self.operator.project(x_tp, y_tn)
        return x_tp, norm


@register_conditioner(name='ps')
class PosteriorSampling(Conditioner):
    def __init__(self, config, operator, noiser):
        super().__init__(config, operator, noiser)
        self.scale = config['conditioner']['scale']

    def forward(self, x_0_hat, x_tp, x_t, y_n, y_tn):
        grad, norm = self.grad_norm(x_t, x_0_hat, y_n)
        x_tp -= grad * self.scale
        return x_tp, norm
    