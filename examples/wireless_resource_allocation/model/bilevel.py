import torch
from .common import MLP, weighted_mse_loss, weighted_sumrate_loss
from .common import weighted_ratio_loss, mse_per_sample, SumRateLoss


class Net(torch.nn.Module):
    def __init__(self,
                 n_inputs,
                 n_outputs,
                 n_tasks,
                 args):
        super(Net, self).__init__()

        # setup network
        hidden = [int(x) for x in args.hidden_layers.split("-")]
        self.net = MLP([n_inputs] + hidden + [n_outputs])
        self.nc_per_task = n_outputs
        self.n_outputs = n_outputs

        # setup optimizer
        self.lr = args.lr
        self.n_iter = args.n_iter
        self.mini_batch_size = args.mini_batch_size

        # setup losses
        self.noise = args.noise
        self.loss_wmse = weighted_mse_loss
        self.loss_dual = weighted_ratio_loss

        # allocate buffer
        self.M = []  
        self.age = 0
        self.memories = args.n_memories

        self.set_index = []

    def forward(self, x, t):
        output = self.net(x)
        return output

    def get_batch(self, x, y):
        if self.M:
            # combine buffer with current samples
            set_x, set_y = self.M
            set_x = torch.cat([set_x, x], 0)
            set_y = torch.cat([set_y, y], 0)
        else:
            set_x, set_y = x, y
        return set_x, set_y

    def observe(self, x, t, y, loss_type='MSE', x_te=None, x_tr=None, scale=1.0):
        self.train()
        set_x, set_y = self.get_batch(x, y)
        grads_squared = [0.0 for _ in self.net.parameters()]
        self.age += 1

        # Parameters
        stepsize = 1e-1
        gamma = 1e0
        inn_iter = 20
        
        for epoch in range(self.n_iter):
            permutation1 = torch.randperm(set_x.size()[0])
            permutation2 = torch.randperm(set_x.size()[0])
            weight_scale = set_x.size()[0] / self.mini_batch_size
            for i in range(0, x.size()[0], self.mini_batch_size):

                self.zero_grad()
                indices1 = permutation1[i:i + self.mini_batch_size]
                batch_x1 = set_x[indices1]
                batch_y1 = set_y[indices1]

                indices2 = permutation2[i:i + self.mini_batch_size]
                batch_x2 = set_x[indices2]
                batch_y2 = set_y[indices2]

                # Compute lambda_star
                nt = self.mini_batch_size 
                lambda_star = torch.rand(nt)
                lambda_star = torch.multiply(lambda_star, lambda_star > 0)
                lambda_star = lambda_star - (1 / nt) * torch.matmul(torch.ones(nt, 1),
                                                                    torch.matmul(torch.ones(1, nt), lambda_star) - 1)
                
                u = self.loss_dual(batch_x1, self.forward(batch_x1, t), batch_y1, self.noise).detach()
                for j in range(inn_iter):
                    grad_gl = -u + gamma * lambda_star
                    lambda_star = lambda_star - stepsize*grad_gl
                    lambda_star = torch.multiply(lambda_star, lambda_star > 0)
                    lambda_star = lambda_star - (1/nt)*torch.matmul(torch.ones(nt, 1),
                                                                    torch.matmul(torch.ones(1, nt), lambda_star) - 1)

                # Grad of f wrt Theta
                grad_f_theta_arr = []
                ff = weighted_mse_loss(self.forward(batch_x2, t), batch_y2, lambda_star.detach())
                for jj, param in enumerate(self.net.parameters()):
                    gg = torch.autograd.grad(ff, param, retain_graph=True)
                    gg = torch.reshape(gg[0], (1, torch.numel(gg[0])))
                    grad_f_theta_arr.append(gg)

                # Grad of f wrt lambda
                grad_f_lambda = mse_per_sample(self, self.forward(batch_x2, t), batch_y2).detach()

                # grad of u wrt Theta lambda
                grad_theta_lambda_arr = []
                u = scale * self.loss_dual(batch_x1, self.forward(batch_x1, t), batch_y1, self.noise)
                for jj, param in enumerate(self.net.parameters()):
                    grad_theta_lambda = torch.empty((1, torch.numel(param)))
                    for j in range(u.size()[0]):
                        gg = torch.autograd.grad(u[j], param, retain_graph=True)
                        gg = torch.reshape(gg[0], (1, torch.numel(gg[0])))
                        grad_theta_lambda = torch.cat((grad_theta_lambda, gg))
                    grad_theta_lambda = grad_theta_lambda[1:]
                    grad_theta_lambda_arr.append(grad_theta_lambda)

                # Compute grad of lambda_star
                with torch.no_grad():
                    grads_squared_new = []
                    cc = (1 / gamma) * (-torch.eye(nt) + (1/nt) * torch.ones(nt, nt))
                    for param, g_g, f2_g, square_g in zip(self.net.parameters(), grad_theta_lambda_arr, grad_f_theta_arr,
                                                          grads_squared):
                        grad_current = torch.matmul(torch.t(torch.matmul(cc, g_g)), grad_f_lambda) \
                                       + torch.t(torch.squeeze(f2_g))
                        if len(param.size()) > 1:
                            grad_current = torch.reshape(grad_current, (param.size()[0], param.size()[1]))
                        square_g = 0.99 * square_g + 0.01 * torch.square(grad_current)
                        param -= self.lr / (torch.sqrt(square_g) + 1e-8) * grad_current
                        grads_squared_new.append(square_g)
                    grads_squared = grads_squared_new

                set_w = torch.exp(scale * self.loss_dual(set_x, self.forward(set_x, t), set_y, self.noise))
                _, indices_first = torch.sort(set_w[0:self.memories], descending=True)
                _, indices_second = torch.sort(set_w[self.memories:], descending=True)
                indices_combine = torch.cat((indices_first[0:round(self.memories * (1.0 - 1.0 / (self.age + 1)))],
                                             indices_second[0:round(self.memories / (self.age + 1))]))
                self.M = (set_x[indices_combine], set_y[indices_combine])
