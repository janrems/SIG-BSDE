import torch
import signatory
from sklearn.linear_model import Ridge
import numpy as np
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler


#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class fbsde():
    def __init__(self, x_0, b, sigma, f, g, T, dim_x, dim_y, dim_d):
        self.x_0 = x_0
        self.b = b
        self.sigma = sigma
        self.f = f
        self.g = g
        self.T = T
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_d = dim_d



class Model(nn.Module):
    def __init__(self, equation, dim_h):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(equation.dim_y + equation.dim_d + 4, dim_h)
        self.linear2 = nn.Linear(dim_h, dim_h)
        self.linear3 = nn.Linear(dim_h, dim_h)
        self.linear4 = nn.Linear(dim_h, equation.dim_y + equation.dim_d)

        nn.init.kaiming_normal_(self.linear1.weight)
        nn.init.kaiming_normal_(self.linear2.weight)
        nn.init.kaiming_normal_(self.linear3.weight)
        nn.init.kaiming_normal_(self.linear4.weight)

        self.equation = equation

    def forward(self, x, r, R, s, S):

        def normalize(x):
            xmax = x.max(dim=0).values
            xmin = x.min(dim=0).values
            return (x-xmin)/(xmax-xmin)

        def standardize(x):
            mean = torch.mean(x,dim=0)
            sd = torch.std(x,dim=0)
            return (x-mean)/sd

        def phi(x,r,R, s, S):
            lower = torch.cat((r,s),dim=-1)
            upper = torch.cat((R,S),dim=-1)
            x = torch.relu(self.linear1(x))
            x = torch.relu(self.linear2(x))
            x = torch.relu(self.linear3(x))
            return torch.maximum(lower, torch.minimum(upper,self.linear4(x))) #[bs,(dy*dd)] -> [bs,dy,dd]



        u = torch.cat((x, r, R, s, S), 1)
        beta = phi(u, r, R,s, S)
        return beta


class ModelConstant(nn.Module):
    def __init__(self, equation, dim_h):
        super(ModelConstant, self).__init__()
        self.linear1 = nn.Linear(equation.dim_y + equation.dim_d, dim_h)
        self.linear2 = nn.Linear(dim_h, dim_h)
        self.linear3 = nn.Linear(dim_h, dim_h)
        self.linear4 = nn.Linear(dim_h, equation.dim_y + equation.dim_d)

        nn.init.kaiming_normal_(self.linear1.weight)
        nn.init.kaiming_normal_(self.linear2.weight)
        nn.init.kaiming_normal_(self.linear3.weight)
        nn.init.kaiming_normal_(self.linear4.weight)

        self.equation = equation

    def forward(self, x, r, R, s, S):

        def normalize(x):
            xmax = x.max(dim=0).values
            xmin = x.min(dim=0).values
            return (x-xmin)/(xmax-xmin)

        def standardize(x):
            mean = torch.mean(x,dim=0)
            sd = torch.std(x,dim=0)
            return (x-mean)/sd

        def phi(x,r,R, s, S):
            lower = torch.cat((r,s),dim=-1)
            upper = torch.cat((R,S),dim=-1)
            x = torch.relu(self.linear1(x))
            x = torch.relu(self.linear2(x))
            x = torch.relu(self.linear3(x))
            return torch.maximum(lower, torch.minimum(upper,self.linear4(x))) #[bs,(dy*dd)] -> [bs,dy,dd]




        beta = phi(x, r, R,s, S)
        return beta


class Train_NN():
    def __init__(self, batch_size, itr, lr, dim_h, equation, lambda1, type):
        self.batch_size = batch_size
        self.itr = itr
        self.equation = equation
        if type == "stochastic":
            self.model = Model(self.equation, dim_h)
        elif type == "constant":
            self.model = ModelConstant(self.equation, dim_h)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)
        #self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda1)
        self.losses = []
        self.losses_true = []
        self.type = type


    def gen_input(self):
        y = torch.randn(self.batch_size, self.equation.dim_y)
        z = torch.randn(self.batch_size, self.equation.dim_d)

        if self.type == "constant":
            R = torch.ones(self.batch_size, self.equation.dim_y)
            r = torch.ones(self.batch_size, self.equation.dim_y)*0

            S = torch.ones(self.batch_size, self.equation.dim_y)
            s = torch.ones(self.batch_size, self.equation.dim_y) * 0

        elif self.type == "stochastic":
            tmp1 = torch.rand(self.batch_size,self.equation.dim_y)
            tmp2 = torch.rand(self.batch_size, self.equation.dim_y)
            r = torch.minimum(tmp1,tmp2)
            R = torch.maximum(tmp1,tmp2)


            tmp1 = torch.rand(self.batch_size, self.equation.dim_y)
            tmp2 = torch.rand(self.batch_size, self.equation.dim_y)
            s = torch.minimum(tmp1, tmp2)
            S = torch.maximum(tmp1, tmp2)

        return r, R, s, S, y, z

    def loss_function(self, y, beta, alpha, z):
        return torch.mean(beta*y + alpha*z)

    def train(self):
        for i in range(self.itr):
            r, R,s, S, y, z = self.gen_input()
            inpt = torch.cat((y, z), 1)
            beta = self.model(inpt, r, R, s, S)
            beta, alpha = beta[:, 0], beta[:, 1]
            loss = self.loss_function(y, beta, alpha, z)
            self.losses.append(float(loss))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            beta_true, alpha_true = self.exact(y, r, R, s, S, z)
            errors = torch.abs(beta_true[:, 0] - beta) + torch.abs(alpha_true[:, 0] - alpha)
            self.losses_true.append(float(torch.mean(errors)))
        return y,z,beta, alpha , r, R, s, S

    def exact(self,y,r,R,s,S,z):
        beta = torch.where(y <= 0, R, r)
        alpha = torch.where(z<=0, S, s)
        return beta,alpha



class ModelFixed(nn.Module):
    def __init__(self, equation, dim_h):
        super(ModelFixed, self).__init__()
        self.linear1 = nn.Linear(equation.dim_y + equation.dim_d, dim_h)
        self.linear2 = nn.Linear(dim_h, dim_h)
        self.linear3 = nn.Linear(dim_h, dim_h)
        self.linear4 = nn.Linear(dim_h, equation.dim_y + equation.dim_d)

        self.equation = equation

    def forward(self, x):

        def normalize(x):
            xmax = x.max(dim=0).values
            xmin = x.min(dim=0).values
            return (x-xmin)/(xmax-xmin)

        def standardize(x):
            mean = torch.mean(x,dim=0)
            sd = torch.std(x,dim=0)
            return (x-mean)/sd

        def phi(x):
            lower = torch.zeros_like(x)
            upper = torch.ones_like(x)*0.5
            x = torch.relu(self.linear1(x))
            x = torch.relu(self.linear2(x))
            x = torch.tanh(self.linear3(x))
            return torch.maximum(lower, torch.minimum(upper,self.linear4(x))) #[bs,(dy*dd)] -> [bs,dy,dd]




        beta = phi(x)
        return beta


class Train_NN_Fixed():
    def __init__(self, batch_size, itr, lr, dim_h, equation, lambda1):
        self.batch_size = batch_size
        self.itr = itr
        self.equation = equation
        self.model = ModelFixed(self.equation, dim_h)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)
        #self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda1)
        self.losses = []


    def gen_input(self):
        y = torch.randn(self.batch_size, self.equation.dim_y)
        z = torch.randn(self.batch_size, self.equation.dim_y)

        return torch.cat((y,z),1)

    def loss_function(self, inpt, out):
        ones = torch.ones(self.batch_size, self.equation.dim_y)
        mones = -torch.ones(self.batch_size, self.equation.dim_y)
        sign = torch.cat((ones,mones),1)
        tmp = inpt*out*sign
        tmp = torch.sum(tmp,1)
        return torch.mean(tmp)

    def train(self):
        for i in range(self.itr):
            inpt = self.gen_input()
            out = self.model(inpt)
            loss = self.loss_function(inpt,out)
            self.losses.append(float(loss))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


        return inpt[:,0:self.equation.dim_y], inpt[:,self.equation.dim_y:], out[:,0:self.equation.dim_y], out[:,self.equation.dim_y:]

    def exact(self,y,z):
        beta = torch.where(y <= 0, 0.5, 0.0)
        alpha = torch.where(z>0, 0.5, 0.0)


        return beta, alpha



class ModelSimple(nn.Module):
    def __init__(self, equation, dim_h):
        super(ModelSimple, self).__init__()
        self.linear1 = nn.Linear(equation.dim_y + equation.dim_d, dim_h)
        self.linear2 = nn.Linear(dim_h, dim_h)
        self.linear3 = nn.Linear(dim_h, dim_h)
        self.linear4 = nn.Linear(dim_h, equation.dim_y + equation.dim_d)

        self.equation = equation

    def forward(self, x, lower, upper):

        def normalize(x):
            xmax = x.max(dim=0).values
            xmin = x.min(dim=0).values
            return (x-xmin)/(xmax-xmin)

        def standardize(x):
            mean = torch.mean(x,dim=0)
            sd = torch.std(x,dim=0)
            return (x-mean)/sd

        def phi(x, lower, upper):
            x = torch.relu(self.linear1(x))
            x = torch.relu(self.linear2(x))
            x = torch.tanh(self.linear3(x))
            return torch.maximum(lower, torch.minimum(upper,self.linear4(x))) #[bs,(dy*dd)] -> [bs,dy,dd]




        beta = phi(x, lower, upper)
        return beta


class Train_NN_Simple():
    def __init__(self, batch_size, itr, lr, dim_h, equation, lambda1):
        self.batch_size = batch_size
        self.itr = itr
        self.equation = equation
        self.model = ModelSimple(self.equation, dim_h)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)
        #self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda1)
        self.losses = []


    def gen_input(self):
        y = torch.randn(self.batch_size, self.equation.dim_y)
        z = torch.randn(self.batch_size, self.equation.dim_y)

        tmp1 = torch.rand(self.batch_size,self.equation.dim_y+self.equation.dim_d)
        tmp2 = torch.rand(self.batch_size, self.equation.dim_y+self.equation.dim_d)
        lower = torch.minimum(tmp1,tmp2)
        upper = torch.maximum(tmp1,tmp2)

        return torch.cat((y,z),1), lower, upper

    def loss_function(self, inpt, out):
        ones = torch.ones(self.batch_size, self.equation.dim_y)
        mones = -torch.ones(self.batch_size, self.equation.dim_y)
        sign = torch.cat((ones,mones),1)
        tmp = inpt*out*sign
        tmp = torch.sum(tmp,1)
        return torch.mean(tmp)

    def train(self):
        for i in range(self.itr):
            inpt,lower,upper = self.gen_input()
            out = self.model(inpt, lower, upper)
            loss = self.loss_function(inpt,out)
            self.losses.append(float(loss))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


        return inpt[:,0:self.equation.dim_y], inpt[:,self.equation.dim_y:], out[:,0:self.equation.dim_y], out[:,self.equation.dim_y:]

    def exact(self,y,z,lower,upper):
        r = lower[:,0]
        s = lower[:,1]
        R = upper[:,0]
        S = upper[:,1]
        beta = torch.where(y <= 0, R, r)
        alpha = torch.where(z>0, S, s)


        return beta, alpha

























class BEM_beta():
    def __init__(self, equation, sample_size, depth, N, model):
        self.equation = equation
        self.sample_size = sample_size
        self.depth = depth
        self.N = N
        self.model = model

    def gen_brownian(self):
        dt = self.equation.T / self.N
        W = torch.randn(self.sample_size, self.equation.dim_d, self.N) * np.sqrt(dt)
        Wt = torch.cumsum(W, dim=-1)
        Wt = torch.roll(Wt, 1, -1)
        Wt[:, :, 0] = torch.zeros(self.sample_size, self.equation.dim_d)
        return Wt

    def gen_forward(self):
        flag = True
        while flag:
            dt = self.equation.T/self.N
            x = self.equation.x_0 + torch.zeros(self.sample_size, self.equation.dim_x, self.N)

            Wt = self.gen_brownian()

            for i in range(self.N-1):
                x[:,:,i+1] = x[:,:,i] + self.equation.b(dt * 1, x[:,:,i]) * dt + torch.matmul(self.equation.sigma(dt * 1, x[:,:,i]),
                                                                                                  (Wt[:,:,i+1]-Wt[:,:,i]).reshape(-1, self.equation.dim_d, 1)).reshape(-1, self.equation.dim_x)
            if torch.isnan(x).any() == False:
                flag = False
        return x, Wt

    def gen_R(self, r):
        #return torch.ones(self.sample_size,self.equation.dim_y, self.N)
        Bt = self.gen_brownian()
        return r + 0.1 + torch.abs(0.2*Bt)

    def gen_r(self):
        return 0.2*self.gen_brownian()

    def sig_cex(self,X, rough, i):
        """
        input:
            rough -- signatory.Path, rough path object of common noise

        return:
            m -- next round conditional dist.
        """
        batch, _ = X.size()

        linear = Ridge(alpha=.1, tol=1e-6)
        label = X[:, 0].detach()
        if i == 0:
            data = torch.cat([torch.zeros(batch, 1),
                              torch.zeros(batch, 1)],
                             dim=1)

            linear.fit(data.numpy(), label.numpy())
            l = torch.tensor(linear.coef_).view(-1, 1)
            # i=1

            return torch.matmul(torch.zeros(batch, 2), l) + linear.intercept_
        if i == 1:
            data = torch.cat([rough.path[0][:, :2, 0],
                              torch.ones(batch, 1) * (i / self.N)], dim=1)
            linear.fit(data.numpy(), label.numpy())
            l = torch.tensor(linear.coef_).view(-1, 1)
            # i=1

            return torch.matmul(data, l) + linear.intercept_
        else:
            data = rough.signature(end=i+1).cpu().detach()

            linear.fit(data.numpy(), label.numpy())
            l = torch.tensor(linear.coef_).view(-1, 1)
            # i=1

            return torch.matmul(data, l) + linear.intercept_

    def numerical(self,x,Wt, r, R):
        self.model.eval()

        Wt = torch.transpose(Wt, 1, -1)
        y = torch.zeros(self.sample_size, self.equation.dim_y, self.N)
        y[:,:,-1] = self.equation.g(x[:,:,-1])
        z = torch.zeros(self.sample_size, self.equation.dim_y*self.equation.dim_d, self.N-1)
        augment = signatory.Augment(1,
                                    layer_sizes=(),
                                    kernel_size=1,
                                    include_time=True)

        rough = signatory.Path(augment(Wt), self.depth, basepoint=False)
        dt = self.equation.T/self.N
        t = torch.linspace(0,1,self.N)
        beta_path = torch.zeros(self.sample_size, self.equation.dim_y, self.N-1)
        for i in range(self.N-2,-1,-1):
            target1 = y[:,:,i+1]*(Wt[:,i+1,:]-Wt[:,i,:])

            if torch.isnan(target1).any():
                print(str(i) + "target1 nan")
            if torch.isinf(target1).any():
                print(str(i) + "target1 inf")

            cex1 = self.sig_cex(target1,rough,i)
            if torch.isnan(cex1).any():
                print(str(i) + "cex1 nan")
            if torch.isinf(cex1).any():
                print(str(i) + "cex1 inf")
            z[:,:,i] = cex1/dt #dt

            #Explicit
            ###############################

            # target2 = y[:,:,i+1] + self.equation.f(t[i], x[:,:,i],y[:,:,i+1],z[:,:,i])*dt
            # if torch.isnan(target2).any():
            #     print(str(i) + "target2 nan")
            # if torch.isinf(target2).any():
            #     print(str(i) + "target2 inf")
            # cex2 = self.sig_cex(target2,rough,i)
            # if torch.isnan(cex2).any():
            #     print(str(i) + "cex2 nan")
            # if torch.isinf(cex2).any():
            #     print(str(i) + "cex2 inf")
            # y[:,:,i] = cex2

            #quasi implicit
            inpt = torch.cat((y[:,:,i+1], z[:, :, i]), 1)
            # lower = torch.zeros(self.sample_size, self.equation.dim_y + self.equation.dim_d)
            # upper = torch.ones(self.sample_size, self.equation.dim_y + self.equation.dim_d)
            out = self.model(inpt)
            # beta_path[:,:,i] = beta
            beta = out[:, 0:1]
            alpha = out[:, 1:2]


            target2 = y[:, :, i + 1] + self.equation.f(t[i], beta, alpha, y[:, :, i + 1], z[:, :, i]) * dt
            if torch.isnan(target2).any():
                print(str(i) + "target2 nan")
            if torch.isinf(target2).any():
                print(str(i) + "target2 inf")
            cex2 = self.sig_cex(target2,rough,i)
            if torch.isnan(cex2).any():
                print(str(i) + "cex2 nan")
            if torch.isinf(cex2).any():
                print(str(i) + "cex2 inf")
            y[:,:,i] = cex2
            ###############################################
            #before
            # y_prev = self.sig_cex(y[:,:,i+1],rough,i)
            # inpt = torch.cat((y_prev, z[:,:,i]), 1)
            # #lower = torch.zeros(self.sample_size, self.equation.dim_y + self.equation.dim_d)
            # #upper = torch.ones(self.sample_size, self.equation.dim_y + self.equation.dim_d)
            # out = self.model(inpt)
            # #beta_path[:,:,i] = beta
            # beta = out[:,0:1]
            # alpha = out[:,1:2]
            # y[:,:,i] = y_prev + self.equation.f(t[i], beta,alpha, y_prev, z[:,:,i])*dt



        return y, z, beta_path

    def numerical_exact_beta(self, x, Wt, r, R):
        Wt = torch.transpose(Wt, 1, -1)
        y = torch.zeros(self.sample_size, self.equation.dim_y, self.N)
        y[:, :, -1] = self.equation.g(x[:, :, -1])
        z = torch.zeros(self.sample_size, self.equation.dim_y * self.equation.dim_d, self.N -1)
        beta_path = torch.zeros(self.sample_size, self.equation.dim_y, self.N - 1)
        augment = signatory.Augment(1,
                                    layer_sizes=(),
                                    kernel_size=1,
                                    include_time=True)

        rough = signatory.Path(augment(Wt), self.depth, basepoint=False)
        dt = self.equation.T / self.N
        t = torch.linspace(0, 1, self.N)
        for i in range(self.N - 2, -1, -1):
            target1 = y[:, :, i + 1] * (Wt[:, i + 1, :] - Wt[:, i, :])

            if torch.isnan(target1).any():
                print(str(i) + "target1 nan")
            if torch.isinf(target1).any():
                print(str(i) + "target1 inf")

            cex1 = self.sig_cex(target1, rough, i)
            if torch.isnan(cex1).any():
                print(str(i) + "cex1 nan")
            if torch.isinf(cex1).any():
                print(str(i) + "cex1 inf")
            z[:, :, i] = cex1 / dt  # np.sqrt(dt)

            # Explicit
            ###############################
            #beta = y[:,:,i+1] < 0
            beta = torch.where(y[:,:,i+1] < 0, 0.5*torch.ones_like(y[:,:,i+1]), torch.zeros_like(y[:,:,i+1]))
            alpha = torch.where(z[:, :, i] > 0, 0.5*torch.ones_like(z[:, :, i]), 0*torch.zeros_like(z[:, :, i]))
            #beta_path[:,:,i] = beta
            beta = beta.float()
            alpha = alpha.float()
            target2 = y[:,:,i+1] + self.equation.f(t[i], beta, alpha, y[:,:,i+1],z[:,:,i])*dt
            if torch.isnan(target2).any():
                print(str(i) + "target2 nan")
            if torch.isinf(target2).any():
                print(str(i) + "target2 inf")
            cex2 = self.sig_cex(target2,rough,i)
            if torch.isnan(cex2).any():
                print(str(i) + "cex2 nan")
            if torch.isinf(cex2).any():
                print(str(i) + "cex2 inf")
            y[:,:,i] = cex2

            # quasi implicit
            ###############################################
            # y_prev = self.sig_cex(y[:, :, i + 1], rough, i)
            # beta = y_prev < 0
            # beta = beta.float()
            #
            # y[:, :, i] = y_prev + self.equation.f(t[i], beta, y_prev, z[:, :, i]) * dt

        return y, z, beta_path


    def numerical_fixed_beta(self, x, Wt, alpha, beta):
        Wt = torch.transpose(Wt, 1, -1)
        y = torch.zeros(self.sample_size, self.equation.dim_y, self.N)
        y[:, :, -1] = self.equation.g(x[:, :, -1])
        z = torch.zeros(self.sample_size, self.equation.dim_y * self.equation.dim_d, self.N -1)
        beta_path = torch.zeros(self.sample_size, self.equation.dim_y, self.N - 1)
        augment = signatory.Augment(1,
                                    layer_sizes=(),
                                    kernel_size=1,
                                    include_time=True)

        rough = signatory.Path(augment(Wt), self.depth, basepoint=False)
        dt = self.equation.T / self.N
        t = torch.linspace(0, 1, self.N)
        for i in range(self.N - 2, -1, -1):
            target1 = y[:, :, i + 1] * (Wt[:, i + 1, :] - Wt[:, i, :])

            if torch.isnan(target1).any():
                print(str(i) + "target1 nan")
            if torch.isinf(target1).any():
                print(str(i) + "target1 inf")

            cex1 = self.sig_cex(target1, rough, i)
            if torch.isnan(cex1).any():
                print(str(i) + "cex1 nan")
            if torch.isinf(cex1).any():
                print(str(i) + "cex1 inf")
            z[:, :, i] = cex1 / dt  # np.sqrt(dt)

            # Explicit
            ###############################
            #beta = y[:,:,i+1] < 0


            target2 = y[:,:,i+1] + self.equation.f(t[i], beta, alpha, y[:,:,i+1],z[:,:,i])*dt
            if torch.isnan(target2).any():
                print(str(i) + "target2 nan")
            if torch.isinf(target2).any():
                print(str(i) + "target2 inf")
            cex2 = self.sig_cex(target2,rough,i)
            if torch.isnan(cex2).any():
                print(str(i) + "cex2 nan")
            if torch.isinf(cex2).any():
                print(str(i) + "cex2 inf")
            y[:,:,i] = cex2

            # quasi implicit
            ###############################################
            # y_prev = self.sig_cex(y[:, :, i + 1], rough, i)
            # beta = y_prev < 0
            # beta = beta.float()
            #
            # y[:, :, i] = y_prev + self.equation.f(t[i], beta, y_prev, z[:, :, i]) * dt

        return y, z, beta_path
