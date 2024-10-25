from utils import count_parameters_in_MB
from model import ista_model 
from collections import namedtuple

def main(Genotype, Choice, num_class):
    if Choice == 224:
        model = ista_model.NetworkImageNet(36, num_class, 8, True, Genotype)
    elif Choice == 32:
        model = ista_model.NetworkCIFAR(36, num_class, 20, True, Genotype)
    param_size = count_parameters_in_MB(model)
    print('param size = {:.4f}MB'.format(param_size))

if __name__ == '__main__':
    Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')
    Genotype = Genotype(normal=[('max_pool_3x3', (0)), ('skip_connect', (1)), ('avg_pool_3x3', (0)), ('skip_connect', (2)), ('avg_pool_3x3', (3)), ('avg_pool_3x3', (1)), ('skip_connect', (1)), ('avg_pool_3x3', (4))], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', (0)), ('skip_connect', (1)), ('sep_conv_3x3', (0)), ('skip_connect', (1)), ('dil_conv_3x3', (1)), ('avg_pool_3x3', (3)), ('sep_conv_5x5', (1)), ('skip_connect', (0))], reduce_concat=range(2, 6))
    Choice = 224
    num_class = 102
    main(Genotype, Choice, num_class)