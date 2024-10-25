import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from utils import SK_LASSO, count_parameters_in_MB, drop_path
from model import ista_model 
from model.darts_genotypes import *
from model.darts_operation import *
from torch.nn.parallel._functions import Broadcast

class Supernet_Itsa(nn.Module):
    def __init__(self, train_config, dataset_config, arch_config, device_ids=None):
        super().__init__()
        self.grad_clip = train_config['w_grad_clip']
        self.report_freq = train_config['print_freq']
        self.arch_config = arch_config
        self.steps = arch_config['steps']
        self.sparseness = arch_config['sparseness']
        self.num_ops = len(PRIMITIVES_ISTA)
        self.proj_dims = arch_config['proj_dims']
        if dataset_config["name"] in ["cifar10","cifar100"]:
            self.model = SearchCNN(C=arch_config['init_channels'],num_classes=dataset_config['num_class'], \
                layers=arch_config['layers'], proj_dims=arch_config['proj_dims'], auxiliary=arch_config['auxiliary'])
        elif dataset_config["name"] in ["sport8","mit67","flowers102","imagenet"]:
            self.model = SearchCNN_IMAGE(C=arch_config['init_channels'],num_classes=dataset_config['num_class'], \
                layers=arch_config['layers'], proj_dims=arch_config['proj_dims'], auxiliary=arch_config['auxiliary'])
        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        self.device_ids = device_ids

    def weights(self):
        weights = []
        for k, p in self.model.named_parameters():
            if 'alpha' not in k:
                weights.append(p)
        return weights
    
    def named_weights(self):
        named_weights = []
        for element in self.model.named_parameters():
            if 'alpha' not in element[0]:
                named_weights.append(element)
        return named_weights

    def alphas(self):
        return self.model.arch_parameters()

    def initialization(self):
        self.base_A_normals = []
        self.base_A_reduces = []
        for i in range(self.steps):
            self.base_A_normals.append(torch.from_numpy(np.random.rand(self.proj_dims, (i+2)*self.num_ops)))
            self.base_A_reduces.append(torch.from_numpy(np.random.rand(self.proj_dims, (i+2)*self.num_ops)))
            # Base_A_normals: [steps, proj_dims, (steps+2)*num_ops]
        self.alpha_normal = torch.stack(self.model.alphas_normal_).detach().cpu().numpy()
        self.alpha_reduce = torch.stack(self.model.alphas_reduce_).detach().cpu().numpy()
        # alpha_normal: [steps, proj_dims] 
        self.x_normals_new = self.do_recovery(self.base_A_normals, self.alpha_normal)
        self.x_reduces_new = self.do_recovery(self.base_A_reduces, self.alpha_reduce)
        # x_normals_new: [steps, (steps+2)*num_ops]
        self.x_normals_last = self.x_normals_new.copy()
        self.x_reduces_last = self.x_reduces_new.copy()
        self.normal_freeze_flag, self.reduce_freeze_flag, _ = self.show_selected( # The selected structure is projected into Genotype
            0, self.x_normals_last, self.x_reduces_last, self.x_normals_new, self.x_reduces_new)

    def do_recovery(self, As, alpha, x_last=None, freeze_flag=None):
        xs = []
        for i in range(self.steps):
            if freeze_flag is not None and freeze_flag[i]:
                xs.append(x_last[i])
                continue
            #lasso = LASSO(As[i].cpu().numpy().copy())
            b = alpha[i] 
            #x = lasso.solve(b)
            x = SK_LASSO(As[i].cpu().numpy().copy(), b)
            xs.append(x)
        return xs

    def show_selected(self, epoch, x_normals_last, x_reduces_last,
                                   x_normals_new, x_reduces_new):
        self.normal_freeze_flag = []
        self.reduce_freeze_flag = []
        self.sum_dist = 0
        print("x_normals distance:")
        for i, (x_n_b, x_n_a) in enumerate(zip(x_normals_last, x_normals_new)):
            dist = np.linalg.norm(x_n_a - x_n_b, 2)
            self.normal_freeze_flag.append(False if epoch == 0 else dist <= self.arch_config['dist_limit'])
            self.sum_dist += dist
            print("Step {}: L2 dist is {}. {}".format(i+1, dist,
                            "freeze!!!" if self.normal_freeze_flag[-1] else "active"))
        print("x_reduces distance:")
        for i, (x_r_b, x_r_a) in enumerate(zip(x_reduces_last, x_reduces_new)):
            dist = np.linalg.norm(x_r_a - x_r_b, 2)
            self.reduce_freeze_flag.append(False if epoch == 0 else dist <= self.arch_config['dist_limit'])
            self.sum_dist += dist
            print("Step {}: L2 dist is {}. {}".format(i+1, dist,
                            "freeze!!!" if self.reduce_freeze_flag[-1] else "active"))
        #print("normal cell:")
        gene_normal = []
        for i, x in enumerate(x_normals_new):
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
        for i, x in enumerate(x_reduces_new):
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
        concat = range(2, 2 + len(x_normals_new))
        self.Genotype = Genotype(
            normal = gene_normal, normal_concat = concat,
            reduce = gene_reduce, reduce_concat = concat)
        model_cifar = ista_model.NetworkCIFAR(36, 10, 20, True, self.Genotype)
        self.param_size = count_parameters_in_MB(model_cifar)
        print('param size = {:.4f}MB'.format(self.param_size))
        return self.normal_freeze_flag, self.reduce_freeze_flag, self.param_size

    def sample_and_proj(self, base_As, xs):
        # Base_As: [steps, proj_dims, (steps+2)*num_ops]
        # xs: [steps, (steps+2)*num_ops]
        As= []
        biases = []
        for i in range(self.steps):
            A = base_As[i].numpy().copy()
            E = A.T.dot(A) - np.eye(A.shape[1])
            x = xs[i].copy()
            #zero_idx = np.abs(x).argsort()[:-self.sparseness] 
            _, zero_idx = picking_optimal(x, self.num_ops, self.sparseness)
            # The choice of optimal 2 options selected here.
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
    
    def pretrain(self, all_freeze, drop_path_prob, i, epochs):
        #print("Doing Search ...")
        self.A_normals, self.normal_biases = self.sample_and_proj(self.base_A_normals, self.x_normals_last)
        # Process A_normal to Best 2 options.
        self.A_reduces, self.reduce_biases = self.sample_and_proj(self.base_A_reduces, self.x_reduces_last)
        if not all_freeze:
            self.model.drop_path_prob = 0
            if False not in self.normal_freeze_flag and False not in self.reduce_freeze_flag:
                self.Stop_flag =  True
            else:
                self.Stop_flag = False # This flag is used to determine whether we should stop the process.
            self.model.init_proj_mat(self.A_normals, self.A_reduces) 
            # Replacing the model.A_normals with A_normals
            self.model.freeze_alpha(self.normal_freeze_flag, self.reduce_freeze_flag) 
            # Freezing certain architecture parameters.
            self.model.init_bias(self.normal_biases, self.reduce_biases)
        elif all_freeze:
            self.model.drop_path_prob = drop_path_prob*i/epochs
            self.model.alphas_detach()
    
    def postrain(self, all_freeze, steps, i):
        if not all_freeze:
            alphas = self.model.arch_parameters()
            self.alpha_normal = torch.stack(alphas[:steps]).detach().cpu().numpy()
            self.alpha_reduce = torch.stack(alphas[steps:]).detach().cpu().numpy()
            #print("Doing Recovery ...")
            if not self.Stop_flag: 
                self.x_normals_new = self.do_recovery(self.base_A_normals, self.alpha_normal,
                        self.x_normals_last, self.normal_freeze_flag)
                self.x_reduces_new = self.do_recovery(self.base_A_reduces, self.alpha_reduce,
                        self.x_reduces_last, self.reduce_freeze_flag)
                self.normal_freeze_flag, self.reduce_freeze_flag, self.param_size = self.show_selected(
                    i+1, self.x_normals_last, self.x_reduces_last, self.x_normals_new, self.x_reduces_new)
                if self.param_size >= self.arch_config['param_limit']: # large model may cause out of memory !!!
                    print('-------------> rejected !!!')
                    self.size_flag = False
                else: 
                    print('-------------> accepted !!!')
                    self.size_flag = True
                if self.size_flag:
                    self.x_normals_last = self.x_normals_new
                    self.x_reduces_last = self.x_reduces_new
    
    def forward(self, x, all_freeze):
        self.model.all_freeze = all_freeze
        self.model.check_proj_alphas()
        if len(self.device_ids) == 1:
            return self.model(x)
        xs = nn.parallel.scatter(x, self.device_ids)
        # replicate modules
        replicas = nn.parallel.replicate(self.model, self.device_ids)
        outputs = nn.parallel.parallel_apply(replicas,
                                             list(zip(xs)),
                                             devices=self.device_ids)
        return nn.parallel.gather(outputs, self.device_ids[0])

class MixedOp(nn.Module):
    def __init__(self, C, stride):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES_ISTA:
            op = OPS[primitive](C, stride, True)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=True))
            if 'skip' in primitive and isinstance(op, Identity):
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=True))
            self._ops.append(op)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
                m.weight.requires_grad = False
                m.bias.requires_grad = False
    def forward(self, x, weights, drop_prob):
        if weights.sum() == 0:
            return 0
        feats = []
        weights = weights.to(x.get_device())
        # Ensuring that x and weights in the same device.
        for w, op in zip(weights, self._ops):
            if w == 0:
                continue
            feat = w * op(x)
            if self.training and drop_prob > 0:
                feat = drop_path(feat, drop_prob)
            feats.append(feat)
        return sum(feats)

class Cell(nn.Module):
    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()
        self.reduction = reduction
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=True)
        else:
            self.preprocess0 = StdConv(C_prev_prev, C, 1, 1, 0, affine=True)
        self.preprocess1 = StdConv(C_prev, C, 1, 1, 0, affine=True)
        self._steps = steps
        self._multiplier = multiplier
        self._ops = nn.ModuleList()
        for i in range(self._steps):#2+3+4+5
            for j in range(2+i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride)
                self._ops.append(op)
    def forward(self, s0, s1, weights, drop_prob):
        # Weights: [2, 8], [3, 8], [4, 8], [5, 8]
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            s = sum(self._ops[offset+j](h, weights[offset+j], drop_prob) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)
        return torch.cat(states[-self._multiplier:], dim=1)

class SearchCNN(nn.Module):
    def __init__(self, C, num_classes, layers,
                 proj_dims=2, steps=4, multiplier=4, stem_multiplier=3, auxiliary=False):
        super(SearchCNN, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._steps = steps
        self._multiplier = multiplier
        self.num_edges = sum(1 for i in range(self._steps) for n in range(2 + i))
        self.proj_dims = proj_dims
        self.auxiliary = auxiliary
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
            if i == 2*layers//3:
                C_to_auxiliary = C_prev
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)
        if self.auxiliary:
            self.auxiliary_head = AuxiliaryHeadCIFAR(C_to_auxiliary, num_classes)
        self._initialize_alphas()
        self.all_freeze = False
    def new(self):
        model_new = SearchCNN(self._C, self._num_classes, self._layers).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new
    def check_proj_alphas(self):
        if not self.all_freeze:
            self.proj_alphas(self.A_normals, self.A_reduces)
    def forward(self, input):
        logits_aux = None
        s0 = s1 = self.stem(input)
        #if not self.all_freeze:
        #    self.proj_alphas(self.A_normals, self.A_reduces)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = self.alphas_reduce
            else:
                weights = self.alphas_normal
            s0, s1 = s1, cell(s0, s1, weights, self.drop_path_prob)
            if i== 2 * self._layers//3:
                if self.auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0),-1))
        return logits, logits_aux
    def _loss(self, input, target):
        logits = self(input)
        return F.cross_entropy(logits, target)
    def _initialize_alphas(self):
        self.alphas_normal_ = []
        self.alphas_reduce_ = []
        for i in range(self._steps):
            self.alphas_normal_.append(nn.Parameter(1e-3 * torch.randn(self.proj_dims, device='cuda')))
            self.alphas_reduce_.append(nn.Parameter(1e-3 * torch.randn(self.proj_dims, device='cuda')))
        # Initializing the vairable for alpha
        self._arch_parameters = self.alphas_normal_ + self.alphas_reduce_
    def freeze_alpha(self, normal_freeze_alpha, reduce_freeze_alpha):
        offset = 0
        for i, (flag, alpha) in enumerate(zip(normal_freeze_alpha, self.alphas_normal_)):
            if flag and alpha.requires_grad:
                alpha.requires_grad = False
                for cell in self.cells:
                    if cell.reduction:
                        continue
                    for j in range(offset, offset+i+2):
                        op = cell._ops[j]
                        for m in op.modules():
                            if isinstance(m, nn.BatchNorm2d):
                                m.weight.requires_grad = True
                                m.bias.requires_grad = True
            offset += i + 2  # 0, 2, 5, 9,
        offset = 0
        for i, (flag, alpha) in enumerate(zip(reduce_freeze_alpha, self.alphas_reduce_)):
            if flag and alpha.requires_grad:
                alpha.requires_grad = False
                for cell in self.cells:
                    if not cell.reduction:
                        continue
                    for j in range(offset, offset+i+2):
                        op = cell._ops[j]
                        for m in op.modules():
                            if isinstance(m, nn.BatchNorm2d):
                                m.weight.requires_grad = True
                                m.bias.requires_grad = True
            offset += i + 2
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
        alphas_normal_ =  torch.stack(self.alphas_normal_) 
        alphas_reduce_ =  torch.stack(self.alphas_reduce_) 
        for i in range(self._steps):
            A_normal = A_normals[i].to(alphas_normal_.device).requires_grad_(False)
            t_alpha = alphas_normal_[[i]].mm(A_normal).reshape(-1, len(PRIMITIVES_ISTA))
            alphas_normal.append(t_alpha)
            A_reduce = A_reduces[i].to(alphas_reduce_.device).requires_grad_(False)
            t_alpha = alphas_reduce_[[i]].mm(A_reduce).reshape(-1, len(PRIMITIVES_ISTA))
            alphas_reduce.append(t_alpha)
        self.alphas_normal = torch.cat(alphas_normal) - self.normal_bias.to(
             alphas_normal_.device)
        self.alphas_reduce = torch.cat(alphas_reduce) - self.reduce_bias.to(
             alphas_reduce_.device)
    def alphas_detach(self):
        self.alphas_normal = self.alphas_normal.detach()
        self.alphas_reduce = self.alphas_reduce.detach()
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

class AuxiliaryHeadCIFAR(nn.Module):
    def __init__(self, C, num_classes):
        """assuming input size 8x8"""
        super(AuxiliaryHeadCIFAR, self).__init__()
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False), # image size = 2 x 2
            nn.Conv2d(C, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True))
        self.classifier = nn.Linear(768, num_classes)
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0),-1))
        return x

class SearchCNN_IMAGE(nn.Module):
    def __init__(self, C, num_classes, layers,
                 proj_dims=2, steps=4, multiplier=4, stem_multiplier=3, auxiliary=False):
        # Layers: 20
        super(SearchCNN_IMAGE, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._steps = steps
        self._multiplier = multiplier
        self.num_edges = sum(1 for i in range(self._steps) for n in range(2 + i))
        self.proj_dims = proj_dims
        self.auxiliary = auxiliary
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
        C_prev_prev, C_prev, C_curr = C, C, C # Different
        self.cells = nn.ModuleList()
        reduction_prev = True # Different
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
            if i == 2*layers//3:
                C_to_auxiliary = C_prev
        #self.global_pooling = nn.AdaptiveAvgPool2d(7) 
        self.global_pooling = nn.AvgPool2d(7) # Different
        self.classifier = nn.Linear(C_prev, num_classes)
        if self.auxiliary:
            self.auxiliary_head = AuxiliaryHeadImageNet(C_to_auxiliary, num_classes) # Different
        self._initialize_alphas()
        self.all_freeze = False
    def new(self):
        model_new = SearchCNN_IMAGE(self._C, self._num_classes, self._layers).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new
    def check_proj_alphas(self):
        if not self.all_freeze:
            self.proj_alphas(self.A_normals, self.A_reduces)
    def forward(self, input):
        logits_aux = None
        s0 = self.stem0(input)  # Different
        s1 = self.stem1(s0) # Different
        #if not self.all_freeze:
        #    self.proj_alphas(self.A_normals, self.A_reduces)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = self.alphas_reduce
            else:
                weights = self.alphas_normal
            s0, s1 = s1, cell(s0, s1, weights, self.drop_path_prob)
            if i== 2 * self._layers//3:
                if self.auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0),-1))
        return logits, logits_aux
    def _loss(self, input, target):
        logits = self(input)
        return F.cross_entropy(logits, target)
    def _initialize_alphas(self):
        self.alphas_normal_ = []
        self.alphas_reduce_ = []
        for i in range(self._steps):
            self.alphas_normal_.append(nn.Parameter(1e-3 * torch.randn(self.proj_dims, device='cuda')))
            self.alphas_reduce_.append(nn.Parameter(1e-3 * torch.randn(self.proj_dims, device='cuda')))
        # Initializing the vairable for alpha
        self._arch_parameters = self.alphas_normal_ + self.alphas_reduce_
    def freeze_alpha(self, normal_freeze_alpha, reduce_freeze_alpha):
        offset = 0
        for i, (flag, alpha) in enumerate(zip(normal_freeze_alpha, self.alphas_normal_)):
            if flag and alpha.requires_grad:
                alpha.requires_grad = False
                for cell in self.cells:
                    if cell.reduction:
                        continue
                    for j in range(offset, offset+i+2):
                        op = cell._ops[j]
                        for m in op.modules():
                            if isinstance(m, nn.BatchNorm2d):
                                m.weight.requires_grad = True
                                m.bias.requires_grad = True
            offset += i + 2  # 0, 2, 5, 9,
        offset = 0
        for i, (flag, alpha) in enumerate(zip(reduce_freeze_alpha, self.alphas_reduce_)):
            if flag and alpha.requires_grad:
                alpha.requires_grad = False
                for cell in self.cells:
                    if not cell.reduction:
                        continue
                    for j in range(offset, offset+i+2):
                        op = cell._ops[j]
                        for m in op.modules():
                            if isinstance(m, nn.BatchNorm2d):
                                m.weight.requires_grad = True
                                m.bias.requires_grad = True
            offset += i + 2
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
        alphas_normal_ =  torch.stack(self.alphas_normal_) 
        alphas_reduce_ =  torch.stack(self.alphas_reduce_) 
        for i in range(self._steps):
            A_normal = A_normals[i].to(alphas_normal_.device).requires_grad_(False)
            t_alpha = alphas_normal_[[i]].mm(A_normal).reshape(-1, len(PRIMITIVES_ISTA))
            alphas_normal.append(t_alpha)
            A_reduce = A_reduces[i].to(alphas_reduce_.device).requires_grad_(False)
            t_alpha = alphas_reduce_[[i]].mm(A_reduce).reshape(-1, len(PRIMITIVES_ISTA))
            alphas_reduce.append(t_alpha)
        self.alphas_normal = torch.cat(alphas_normal) - self.normal_bias.to(
             alphas_normal_.device)
        self.alphas_reduce = torch.cat(alphas_reduce) - self.reduce_bias.to(
             alphas_reduce_.device)
    def alphas_detach(self):
        self.alphas_normal = self.alphas_normal.detach()
        self.alphas_reduce = self.alphas_reduce.detach()
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

class AuxiliaryHeadImageNet(nn.Module):
  def __init__(self, C, num_classes):
    """assuming input size 14x14"""
    super(AuxiliaryHeadImageNet, self).__init__()
    self.features = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.AvgPool2d(5, stride=2, padding=0, count_include_pad=False),
      nn.Conv2d(C, 128, 1, bias=False),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True),
      nn.Conv2d(128, 768, 2, bias=False),
      nn.BatchNorm2d(768),
      nn.ReLU(inplace=True))
    self.classifier = nn.Linear(768, num_classes)
  def forward(self, x):
    x = self.features(x)
    x = self.classifier(x.view(x.size(0),-1))
    return x


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