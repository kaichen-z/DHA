import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from utils import SK_LASSO
from model.darts_genotypes import *
from model.darts_operation import *
from utils import count_parameters_in_MB

class Supernet_Itsa(nn.Module):
    def __init__(self, train_config, dataset_config, arch_config, device_ids=None):
        super().__init__()
        self.grad_clip = train_config['w_grad_clip']
        self.report_freq = train_config['print_freq']
        self.steps = arch_config['steps']
        self.sparseness = arch_config['sparseness']
        self.num_ops = len(PRIMITIVES_ISTA)
        self.proj_dims = arch_config['proj_dims']
        if dataset_config['name'] in ['cifar10','cifar100']:
            self.model = SearchCNN(C=arch_config['init_channels'],num_classes=dataset_config['num_class'], \
                layers=arch_config['layers'], proj_dims=arch_config['proj_dims']).cuda()
        elif dataset_config['name'] in ['sport8','mit67','flowers102','imagenet']:
            self.model = SearchCNN_IMAGE(C=arch_config['init_channels'],num_classes=dataset_config['num_class'], \
                layers=arch_config['layers'], proj_dims=arch_config['proj_dims']).cuda()
        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        self.device_ids = device_ids
        print("Param size = {}MB".format(count_parameters_in_MB(self.model)))

    def weights(self):
        weights = []
        for k, p in self.model.named_parameters():
            if 'alpha' not in k:
                weights.append(p)
        return weights
    
    def alphas(self): #Used for updating parameters.
        return self.model.arch_parameters()

    def initialization(self):
        self.base_A_normals = []
        self.base_A_reduces = []
        for i in range(self.steps):
            self.base_A_normals.append(torch.from_numpy(np.random.rand(self.proj_dims, (i+2)*self.num_ops)))
            self.base_A_reduces.append(torch.from_numpy(np.random.rand(self.proj_dims, (i+2)*self.num_ops)))
        self.alpha_normal = self.model.alphas_normal_.detach().cpu().numpy() #(steps, proj_dim)
        self.alpha_reduce = self.model.alphas_reduce_.detach().cpu().numpy() #(steps, proj_dim)
        self.x_normals = self.do_recovery(self.base_A_normals, self.alpha_normal) #(step, (i+2)*num_ops)
        self.x_reduces = self.do_recovery(self.base_A_reduces, self.alpha_reduce) #(step, (i+2)*num_ops)
        self.show_selected(0, self.x_normals, self.x_reduces) # Projecting it to Genotype

    def do_recovery(self, As, alpha, x_last=None, freeze_flag=None): 
        # Fixed
        xs = []
        for i in range(self.steps):
            if freeze_flag is not None and freeze_flag[i]:
                xs.append(x_last[i])
                continue
            b = alpha[i]
            x = SK_LASSO(As[i].cpu().numpy().copy(), b)
            xs.append(x)
        return xs
    
    def show_selected(self, epoch, x_normals, x_reduces):
        #print("normal cell:")
        gene_normal = []
        for i, x in enumerate(x_normals):
            index, _ = picking_optimal(x, self.num_ops, self.sparseness)
            id1, id2 = index
            #id1, id2 = np.abs(x).argsort()[-2:]
            '''
            print("Step {}: node{} op{}, node{} op{}".format(
                i + 1, id1 // self.num_ops,
                       id1 % self.num_ops,
                       id2 // self.num_ops,
                       id2 % self.num_ops))
            '''
            gene_normal.append((PRIMITIVES_ISTA[id1 % self.num_ops], id1 // self.num_ops))
            gene_normal.append((PRIMITIVES_ISTA[id2 % self.num_ops], id2 // self.num_ops))
        #print("reduction cell:")
        gene_reduce = []
        for i, x in enumerate(x_reduces):
            index, _ = picking_optimal(x, self.num_ops, self.sparseness)
            id1, id2 = index
            #id1, id2 = np.abs(x).argsort()[-2:]
            '''
            print("Step {}: node{} op{}, node{} op{}".format(
                i + 1, id1 // self.num_ops,
                       id1 % self.num_ops,
                       id2 // self.num_ops,
                       id2 % self.num_ops))
            '''
            gene_reduce.append((PRIMITIVES_ISTA[id1 % self.num_ops], id1 // self.num_ops))
            gene_reduce.append((PRIMITIVES_ISTA[id2 % self.num_ops], id2 // self.num_ops))
        concat = range(2, 2 + len(x_normals))
        self.Genotype = Genotype(
            normal = gene_normal, normal_concat = concat,
            reduce = gene_reduce, reduce_concat = concat)
        print(self.Genotype)

    def sample_and_proj(self, base_As, xs):
        # base_As (num_dim, (i+2)*num_ops)
        # xs ((i+2)*num_ops)
        As= []
        biases = []
        for i in range(self.steps):
            A = base_As[i].numpy().copy()
            E = A.T.dot(A) - np.eye(A.shape[1])
            x = xs[i].copy()
            #zero_idx = np.abs(x).argsort()[:-self.sparseness]
            _, zero_idx = picking_optimal(x, self.num_ops, self.sparseness)
            x[zero_idx] = 0.
            A[:, zero_idx] = 0.
            As.append(torch.from_numpy(A).float())
            E[:, zero_idx] = 0.
            bias = E.T.dot(x).reshape(-1, self.num_ops)
            biases.append(torch.from_numpy(bias).float())
        biases = torch.cat(biases)
        return As, biases

    def genotype(self):
        return self.Genotype

    def pretrain(self):
        #print("Doing Search ...")
        self.A_normals, self.normal_biases = self.sample_and_proj(self.base_A_normals, self.x_normals) 
        # A_normals is the top_2 based_A_normals, normal_biases is the updated z (step, (i+2)*num_ops)
        self.A_reduces, self.reduce_biases = self.sample_and_proj(self.base_A_reduces, self.x_reduces)
        self.model.init_proj_mat(self.A_normals, self.A_reduces)
        self.model.init_bias(self.normal_biases, self.reduce_biases)

    def postrain(self, i):
        alpha_normal, alpha_reduce = self.model.arch_parameters()
        alpha_normal = alpha_normal.detach().cpu().numpy() 
        alpha_reduce = alpha_reduce.detach().cpu().numpy()  
        self.alpha_normal = alpha_normal
        self.alpha_reduce = alpha_reduce
        #print("Doing Recovery ...")
        self.x_normals = self.do_recovery(self.base_A_normals, self.alpha_normal)
        self.x_reduces = self.do_recovery(self.base_A_reduces, self.alpha_reduce)
        self.show_selected(i+1, self.x_normals, self.x_reduces)
        
    def forward(self, x):
        if len(self.device_ids) == 1:
            return self.model(x)
        xs = nn.parallel.scatter(x, self.device_ids)
        # replicate modules
        replicas = nn.parallel.replicate(self.model, self.device_ids)
        outputs = nn.parallel.parallel_apply(replicas,
                                             list(zip(xs)),
                                             devices=self.device_ids)
        return nn.parallel.gather(outputs, self.device_ids[0])
        '''
        outputs = self.model(x)
        return outputs
        '''
        
class MixedOp(nn.Module):
    def __init__(self, C, stride):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES_ISTA:
            op = OPS[primitive](C, stride, False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self._ops.append(op)

    def forward(self, x, weights):
        if weights.sum() == 0:
            return 0
        return sum(w * op(x) for w, op in zip(weights, self._ops) if w != 0)

class Cell(nn.Module):
    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()
        self.reduction = reduction
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = StdConv(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = StdConv(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._multiplier = multiplier
        self._ops = nn.ModuleList()
        for i in range(self._steps):
            for j in range(2+i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride)
                self._ops.append(op)

    def forward(self, s0, s1, weights):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            s = sum(self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)
        return torch.cat(states[-self._multiplier:], dim=1)

class SearchCNN(nn.Module):
    def __init__(self, C, num_classes, layers, \
        proj_dims=2, steps=4, multiplier=4, stem_multiplier=3):
        super(SearchCNN, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._steps = steps
        self._multiplier = multiplier
        self.num_edges = sum(1 for i in range(self._steps) for n in range(2 + i))
        self.proj_dims = proj_dims
        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr))
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers//3, 2*layers//3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells.append(cell)
            C_prev_prev, C_prev = C_prev, multiplier * C_curr
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)
        self._initialize_alphas()
    def new(self):
        model_new = SearchCNN(self._C, self._num_classes, self._layers).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new
    def forward(self, input):
        s0 = s1 = self.stem(input)
        self.proj_alphas(self.A_normals, self.A_reduces) 
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = self.alphas_reduce
            else:
                weights = self.alphas_normal
            s0, s1 = s1, cell(s0, s1, weights)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0),-1))
        return logits
    def _loss(self, input, target):
        logits = self(input)
        return F.cross_entropy(logits, target)
    def _initialize_alphas(self):
        self.alphas_normal_ = nn.Parameter(1e-3*torch.randn(self._steps, self.proj_dims))
        self.alphas_reduce_ = nn.Parameter(1e-3*torch.randn(self._steps, self.proj_dims))
        self._arch_parameters = [
            self.alphas_normal_,
            self.alphas_reduce_,]
    def init_proj_mat(self, A_normals, A_reduces):
        self.A_normals = A_normals
        self.A_reduces = A_reduces
    def init_bias(self, normal_bias, reduce_bias):
        self.normal_bias = normal_bias
        self.reduce_bias = reduce_bias
    def proj_alphas(self, A_normals, A_reduces):
        assert len(A_normals) == len(A_reduces) == self._steps
        alphas_normal = []
        alphas_reduce = []
        alphas_normal_ = self.alphas_normal_ #F.softmax(self.alphas_normal_, dim=-1)
        alphas_reduce_ = self.alphas_reduce_ #F.softmax(self.alphas_reduce_, dim=-1)
        for i in range(self._steps):
            A_normal = A_normals[i].to(alphas_normal_.device).requires_grad_(False)
            t_alpha = alphas_normal_[[i]].mm(A_normal).reshape(-1, len(PRIMITIVES_ISTA))
            alphas_normal.append(t_alpha)
            A_reduce = A_reduces[i].to(alphas_reduce_.device).requires_grad_(False)
            t_alpha = alphas_reduce_[[i]].mm(A_reduce).reshape(-1, len(PRIMITIVES_ISTA))
            alphas_reduce.append(t_alpha)
        self.alphas_normal = torch.cat(alphas_normal) - self.normal_bias.to(alphas_normal_.device)
        # alpha ((i+2)*num_ops)
        self.alphas_reduce = torch.cat(alphas_reduce) - self.reduce_bias.to(alphas_reduce_.device)
    def arch_parameters(self):
        return self._arch_parameters
    def genotype(self):
        def _parse(weights):
            gene = []
            n = 2
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                edges = sorted(range(i + 2),
                    key=lambda x: -max(W[x][k] for k in range(len(W[x]))
                                        if k != PRIMITIVES_ISTA.index('none')))[:2]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k != PRIMITIVES_ISTA.index('none'):
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append((PRIMITIVES_ISTA[k_best], j))
                start = end
                n += 1
            return gene
        gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
        gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())
        concat = range(2+self._steps-self._multiplier, self._steps+2)
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concat)
        return genotype

class SearchCNN_IMAGE(nn.Module):
    def __init__(self, C, num_classes, layers, proj_dims=2, steps=4, multiplier=4, stem_multiplier=3):
        super(SearchCNN_IMAGE, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._steps = steps
        self._multiplier = multiplier
        self.num_edges = sum(1 for i in range(self._steps) for n in range(2 + i))
        self.proj_dims = proj_dims
        self.stem0 = nn.Sequential(
            nn.Conv2d(3, C // 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(C // 2, C, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C),) # Different
        self.stem1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C),) # Different
        C_prev_prev, C_prev, C_curr = C, C, C
        self.cells = nn.ModuleList()
        reduction_prev = True
        for i in range(layers):
            if i in [layers//3, 2*layers//3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells.append(cell)
            C_prev_prev, C_prev = C_prev, multiplier * C_curr
        self.global_pooling = nn.AvgPool2d(7)
        self.classifier = nn.Linear(C_prev, num_classes)
        self._initialize_alphas()
    def new(self):
        model_new = SearchCNN(self._C, self._num_classes, self._layers).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new
    def forward(self, input):
        s0 = self.stem0(input)  # Different
        s1 = self.stem1(s0) # Different
        self.proj_alphas(self.A_normals, self.A_reduces) 
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = self.alphas_reduce
            else:
                weights = self.alphas_normal
            s0, s1 = s1, cell(s0, s1, weights)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0),-1))
        return logits
    def _loss(self, input, target):
        logits = self(input)
        return F.cross_entropy(logits, target)
    def _initialize_alphas(self):
        self.alphas_normal_ = nn.Parameter(1e-3*torch.randn(self._steps, self.proj_dims))
        self.alphas_reduce_ = nn.Parameter(1e-3*torch.randn(self._steps, self.proj_dims))
        self._arch_parameters = [
            self.alphas_normal_,
            self.alphas_reduce_,]
    def init_proj_mat(self, A_normals, A_reduces):
        self.A_normals = A_normals
        self.A_reduces = A_reduces
    def init_bias(self, normal_bias, reduce_bias):
        self.normal_bias = normal_bias
        self.reduce_bias = reduce_bias
    def proj_alphas(self, A_normals, A_reduces):
        assert len(A_normals) == len(A_reduces) == self._steps
        alphas_normal = []
        alphas_reduce = []
        alphas_normal_ = self.alphas_normal_ #F.softmax(self.alphas_normal_, dim=-1)
        alphas_reduce_ = self.alphas_reduce_ #F.softmax(self.alphas_reduce_, dim=-1)
        for i in range(self._steps):
            A_normal = A_normals[i].to(alphas_normal_.device).requires_grad_(False)
            t_alpha = alphas_normal_[[i]].mm(A_normal).reshape(-1, len(PRIMITIVES_ISTA))
            alphas_normal.append(t_alpha)
            A_reduce = A_reduces[i].to(alphas_reduce_.device).requires_grad_(False)
            t_alpha = alphas_reduce_[[i]].mm(A_reduce).reshape(-1, len(PRIMITIVES_ISTA))
            alphas_reduce.append(t_alpha)
        self.alphas_normal = torch.cat(alphas_normal) - self.normal_bias.to(alphas_normal_.device)
        # alpha ((i+2)*num_ops)
        self.alphas_reduce = torch.cat(alphas_reduce) - self.reduce_bias.to(alphas_reduce_.device)
    def arch_parameters(self):
        return self._arch_parameters
    def genotype(self):
        def _parse(weights):
            gene = []
            n = 2
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                edges = sorted(range(i + 2),
                    key=lambda x: -max(W[x][k] for k in range(len(W[x]))
                                        if k != PRIMITIVES_ISTA.index('none')))[:2]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k != PRIMITIVES_ISTA.index('none'):
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append((PRIMITIVES_ISTA[k_best], j))
                start = end
                n += 1
            return gene
        gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
        gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())
        concat = range(2+self._steps-self._multiplier, self._steps+2)
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concat)
        return genotype

def picking_optimal(edges, num_ops, k):
    # num_ops = (steps+2)*num_ops
    edges = torch.tensor(edges)
    edges_matrix = edges.view(-1, num_ops)
    edge_max, primitive_indices = torch.topk(edges_matrix[:, :-1], 1) 
    topk_edge_values, topk_edge_indices = torch.topk(edge_max.view(-1), k)
    indices = (topk_edge_indices*num_ops).view(-1) + (primitive_indices[topk_edge_indices]).view(-1)
    mask = torch.tensor([i for i in range(len(edges))])
    indices_sort, _ = torch.sort(indices)
    j = 0
    for i in indices_sort:
        i = i - j
        mask = torch.cat([mask[0:i], mask[i+1:]])
        j += 1
    return indices, mask