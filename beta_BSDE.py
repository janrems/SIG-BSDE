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



class ModelFixed(nn.Module):
    def __init__(self, equation, dim_h):
        super(ModelFixed, self).__init__()
        self.linear1 = nn.Linear(equation.dim_y, dim_h)
        self.linear2 = nn.Linear(dim_h, dim_h)
        self.linear3 = nn.Linear(dim_h, dim_h)
        self.linear4 = nn.Linear(dim_h, equation.dim_y)

        self.equation = equation

    def forward(self, x, r, R):

        def normalize(x):
            xmax = x.max(dim=0).values
            xmin = x.min(dim=0).values
            return (x-xmin)/(xmax-xmin)

        def standardize(x):
            mean = torch.mean(x,dim=0)
            sd = torch.std(x,dim=0)
            return (x-mean)/sd

        def phi(x,r,R):
            x = torch.relu(self.linear1(x))
            x = torch.relu(self.linear2(x))
            x = torch.tanh(self.linear3(x))
            return torch.maximum(r, torch.minimum(R,self.linear4(x))) #[bs,(dy*dd)] -> [bs,dy,dd]




        beta = phi(x, r, R)
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

        R = torch.ones(self.batch_size, self.equation.dim_y)
        r = torch.ones(self.batch_size, self.equation.dim_y)*0



        return r, R, y

    def loss_function(self, y, beta):
        return torch.mean(beta*y)

    def train(self):
        for i in range(self.itr):
            r, R, y = self.gen_input()
            beta = self.model(y, r, R)
            loss = self.loss_function(y, beta)
            self.losses.append(float(loss))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


        return y, beta, r, R

    def exact(self,y,r,R):
        result = torch.where(y <= 0, R, r)
        return result



class Model(nn.Module):
    def __init__(self, equation, dim_h):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(equation.dim_y + 2, dim_h)
        self.linear2 = nn.Linear(dim_h, dim_h)
        self.linear3 = nn.Linear(dim_h, dim_h)
        self.linear4 = nn.Linear(dim_h, equation.dim_y)

        self.equation = equation

    def forward(self, x, r, R):

        def normalize(x):
            xmax = x.max(dim=0).values
            xmin = x.min(dim=0).values
            return (x-xmin)/(xmax-xmin)

        def standardize(x):
            mean = torch.mean(x,dim=0)
            sd = torch.std(x,dim=0)
            return (x-mean)/sd

        def phi(x,r,R):
            x = torch.relu(self.linear1(x))
            x = torch.relu(self.linear2(x))
            x = torch.tanh(self.linear3(x))
            return torch.maximum(r, torch.minimum(R,self.linear4(x))) #[bs,(dy*dd)] -> [bs,dy,dd]



        u = torch.cat((x, r, R), 1)
        beta = phi(u, r, R)
        return beta


class Train_NN():
    def __init__(self, batch_size, itr, lr, dim_h, equation, lambda1):
        self.batch_size = batch_size
        self.itr = itr
        self.equation = equation
        self.model = Model(self.equation, dim_h)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)
        #self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda1)
        self.losses = []


    def gen_input(self):
        y = torch.randn(self.batch_size, self.equation.dim_y)

        #R = torch.ones(self.batch_size, self.equation.dim_y)
        #r = torch.ones(self.batch_size, self.equation.dim_y)*0

        tmp1 = torch.rand(self.batch_size,self.equation.dim_y)
        tmp2 = torch.rand(self.batch_size, self.equation.dim_y)
        r = torch.minimum(tmp1,tmp2)
        R = torch.maximum(tmp1,tmp2)

        return r, R, y

    def loss_function(self, y, beta):
        return torch.mean(beta*y)

    def train(self):
        for i in range(self.itr):
            r, R, y = self.gen_input()
            beta = self.model(y, r, R)
            loss = self.loss_function(y, beta)
            self.losses.append(float(loss))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


        return y, beta, r, R

    def exact(self,y,r,R):
        result = torch.where(y <= 0, R, r)
        return result




class ModelSimple(nn.Module):
    def __init__(self, equation, dim_h):
        super(ModelSimple, self).__init__()
        self.linear1 = nn.Linear(equation.dim_y, dim_h)
        self.linear2 = nn.Linear(dim_h, dim_h)
        self.linear3 = nn.Linear(dim_h, dim_h)
        self.linear4 = nn.Linear(dim_h, equation.dim_y)

        self.equation = equation

    def forward(self, x, r, R):

        def normalize(x):
            xmax = x.max(dim=0).values
            xmin = x.min(dim=0).values
            return (x-xmin)/(xmax-xmin)

        def standardize(x):
            mean = torch.mean(x,dim=0)
            sd = torch.std(x,dim=0)
            return (x-mean)/sd

        def phi(x,r,R):
            x = torch.relu(self.linear1(x))
            x = torch.relu(self.linear2(x))
            x = torch.tanh(self.linear3(x))
            return torch.maximum(r, torch.minimum(R,self.linear4(x))) #[bs,(dy*dd)] -> [bs,dy,dd]




        beta = phi(x, r, R)
        return beta


class Train_NN_Simple():
    def __init__(self, batch_size, itr, lr, dim_h, equation, lambda1):
        self.batch_size = batch_size
        self.itr = itr
        self.equation = equation
        self.model = Model(self.equation, dim_h)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)
        #self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda1)
        self.losses = []


    def gen_input(self):
        y = torch.randn(self.batch_size, self.equation.dim_y)

        R = torch.ones(self.batch_size, self.equation.dim_y)*0.5
        r = torch.ones(self.batch_size, self.equation.dim_y)*0

        # tmp1 = torch.rand(self.batch_size,self.equation.dim_y)
        # tmp2 = torch.rand(self.batch_size, self.equation.dim_y)
        # r = torch.minimum(tmp1,tmp2)
        # R = torch.maximum(tmp1,tmp2)

        return r, R, y

    def loss_function(self, y, beta):
        return torch.mean(beta*y)

    def train(self):
        for i in range(self.itr):
            r, R, y = self.gen_input()
            beta = self.model(y, r, R)
            loss = self.loss_function(y, beta)
            self.losses.append(float(loss))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


        return y, beta, r, R

    def exact(self,y,r,R):
        result = torch.where(y <= 0, R, r)
        return result



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
        return torch.ones(self.sample_size,self.equation.dim_y, self.N)*0.5
        #Bt = self.gen_brownian()
        return r + 0.1 + torch.abs(0.2*Bt)

    def gen_r(self):
        return torch.ones(self.sample_size, self.equation.dim_y, self.N) * 0.0
        #return torch.abs(0.2*self.gen_brownian())

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
            z[:,:,i] = cex1/np.sqrt(dt) #dt

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
            ###############################################
            y_prev = self.sig_cex(y[:,:,i+1],rough,i)
            beta = self.model(y_prev, r[:, :, i], R[:, :, i])
            beta_path[:,:,i] = beta

            y[:,:,i] = y_prev + self.equation.f(t[i], beta, y_prev, z[:,:,i])*dt



        return y, z, beta_path

    def numerical_exact_beta(self, x, Wt, r, R):
        Wt = torch.transpose(Wt, 1, -1)
        y = torch.zeros(self.sample_size, self.equation.dim_y, self.N)
        y[:, :, -1] = self.equation.g(x[:, :, -1])
        z = torch.zeros(self.sample_size, self.equation.dim_y * self.equation.dim_d, self.N - 1)
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
            beta = torch.where(y[:,:,i+1] <= 0, R[:,:,i+1], r[:,:,i+1])
            beta_path[:,:,i] = beta
            beta = beta.float()
            target2 = y[:,:,i+1] + self.equation.f(t[i], beta,y[:,:,i+1],z[:,:,i])*dt
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
