import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict 

class HPO_INIT:
    # This HPO controller including both the learnign rate and weight_decay.
    def __init__(self, config, device):
        self.config = config
        self.learning =  nn.DataParallel(torch.tensor((config['learning_up']+config['learning_low'])/2, requires_grad=True, device=device))
        self.weight_decay =  nn.DataParallel(torch.tensor((config['weight_decay_up']+config['weight_decay_low'])/2, requires_grad=True, device=device))
        self.state = defaultdict(dict)
    '''
    def limited(self, freeze=False):
        if not freeze:
            self.weight_decay.module.data = self.weight_decay.module.data.clamp(self.config['weight_decay_low'], self.config['weight_decay_up'])
            self.learning.module.data = self.learning.module.data.clamp((self.config['learning_up']+self.config['learning_low'])/2, (self.config['learning_up']+self.config['learning_low'])/2)
        elif freeze:
            self.weight_decay.module.data = self.weight_decay.module.data.clamp(self.config['weight_decay_low'], self.config['weight_decay_up'])
            self.learning.module.data = self.learning.module.data.clamp(self.config['learning_low'], self.config['learning_up'])
    '''
    def limited(self, freeze=False):
        self.weight_decay.module.data = self.weight_decay.module.data.clamp(self.config['weight_decay_low'], self.config['weight_decay_up'])
        self.learning.module.data = self.learning.module.data.clamp(self.config['learning_low'], self.config['learning_up'])

    def restart(self,):
        self.weight_decay.module.data = torch.tensor((self.config['weight_decay_up']+self.config['weight_decay_low'])/2).cuda()
        self.learning.module.data = torch.tensor((self.config['learning_up']+self.config['learning_low'])/2).cuda()

    def zero_grad(self, params):
        for p in params:
            try:
                p.grad.detach_()
                p.grad.zero_()
            except:
                pass 
    def SGD_STEP(self, name_params, momentum, dampening=0, nesterov=False):
        for n, p in name_params:
            if p.grad is None:
                continue
            d_p = p.grad.data # Here the gradient changed. 
            if self.weight_decay != 0:
                d_p += p.data*self.weight_decay.module
            if momentum != 0:
                if 'momentum_buffer' not in self.state[p]:
                    buf = self.state[p]['momentum_buffer'] = torch.clone(d_p).detach()
                else:
                    buf = self.state[p]['momentum_buffer']
                    buf.mul_(momentum).add_(1 - dampening, d_p)
                if nesterov:
                    d_p = d_p.add(momentum, buf)
                else:
                    d_p = buf
            p.detach_()
            p.copy_(p.data + d_p*(-self.learning.module))    
    def reset_model(self, parameters):
        for variable in parameters:
            variable.detach_()
            variable.requires_grad = True
            try:
                self.state[variable]['momentum_buffer'].detach_()
            except:
                error = 1

class DA_INIT:
    def __init__(self, config, device):
        self.config = config
        policy_alpha = np.ones((self.config["policy"], self.config["policy"]))
        self.policy_alpha = torch.tensor(policy_alpha/policy_alpha.sum(), requires_grad=True, device=device)
        self.policy_alpha = nn.DataParallel(self.policy_alpha)
        self.weights_matrix()
        # Rho is used in the Feng Loss Function.
        try:
            self.rho = config['rho']
        except:
            self.rho = 1.25
        # The probability is represented as exp(xi)/exp(xi).sum().
    def weights_matrix(self):
        probability_matrix = torch.exp(self.policy_alpha.module.detach())/(torch.exp(self.policy_alpha.module.detach()).sum())
        try:
            unif = torch.distributions.Uniform(0,1).sample((self.policy_alpha.module.size(0),self.policy_alpha.module.size(1))).type(self.policy_alpha.module.type())
        except:
            unif = torch.distributions.Uniform(0,1).sample((self.policy_alpha.size(0),self.policy_alpha.size(1))).type(self.policy_alpha.type())
        g = -torch.log(-torch.log(unif))
        h = (g + torch.log(probability_matrix))/self.config['temperature']
        self.matrix = h.exp()/h.exp().sum() 
        return self.matrix.detach().cpu().numpy()
    def entropy_alpha(self):
        probability_matrix = torch.exp(self.policy_alpha.module.detach())/(torch.exp(self.policy_alpha.module.detach()).sum())
        entropy = (-np.log(probability_matrix.cpu().numpy())*probability_matrix.cpu().numpy()).sum()
        return entropy
    def prob_loss(self, loss, train_op):
        probability_matrix = torch.exp(self.policy_alpha.module)/(torch.exp(self.policy_alpha.module).sum())
        action1 = torch.argmax(train_op[:, :, 0], dim=1)
        action2 = torch.argmax(train_op[:, :, 1], dim=1)
        action = torch.stack((action1, action2),dim=1)
        prob_list = torch.stack([probability_matrix[i[0], i[1]] for i in action])
        loss_list = loss*prob_list.float()
        return loss_list.mean()
    def feng_loss(self, loss_tf, loss_ori):
        result_loss = torch.abs(1.0-torch.exp(loss_tf-self.rho*loss_ori))
        return result_loss