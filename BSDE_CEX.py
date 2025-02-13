import torch
import signatory
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
import numpy as np


class fbsde():
    def __init__(self, x_0, b, sigma, f, g, T, dim_x, dim_y, dim_d, l, ref_flag):
        self.x_0 = x_0
        self.b = b
        self.sigma = sigma
        self.f = f
        self.g = g
        self.T = T
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_d = dim_d
        self.l = l
        self.ref_flag = ref_flag

class BEM():
    def __init__(self, equation, sample_size, depth, N):
        self.equation = equation
        self.sample_size = sample_size
        self.depth = depth
        self.N = N
        self.y = torch.zeros(self.sample_size, self.equation.dim_y, self.N)
        self.z = torch.zeros(self.sample_size, self.equation.dim_y * self.equation.dim_d, self.N - 1)


    def gen_forward(self):
        flag = True
        cntr = 1
        while flag:
            #print(cntr)
            cntr += 1
            dt = self.equation.T/self.N
            x = self.equation.x_0 + torch.zeros(self.sample_size, self.equation.dim_x, self.N)

            W = torch.randn(self.sample_size, self.equation.dim_d, self.N)*np.sqrt(dt)
            Wt = torch.cumsum(W, dim=-1)
            Wt = torch.roll(Wt, 1, -1)
            Wt[:, :, 0] = torch.zeros(self.sample_size, self.equation.dim_d)

            for i in range(self.N-1):
                x[:,:,i+1] = (x[:,:,i] + self.equation.b(dt * 1, x[:,:,i]) * dt + 
                              torch.matmul(self.equation.sigma(dt * 1, x[:,:,i]),
                                           (Wt[:,:,i+1]-Wt[:,:,i]).reshape(-1, self.equation.dim_d, 1)).reshape(-1, self.equation.dim_x))
            if torch.isnan(x).any() == False:
                flag = False
            if cntr > 100:
                print("break 100")
                #break
        return x, Wt


    def sig_cex(self,X, rough, i):
        """
        input:
            rough -- signatory.Path, rough path object of common noise

        return:
            m -- next round conditional dist.
        """
        batch, _ = X.size()

        if self.depth == 1:
            linear = LinearRegression()
        else:
            linear = Ridge(alpha=0.8, tol=1e-6)

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
                              torch.ones(batch, 1) * (i / N)], dim=1)
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

    def numerical(self,x,Wt):
        Wt = torch.transpose(Wt, 1, -1)
        xt = torch.transpose(x, 1, -1)
        self.y[:,:,-1] = self.equation.g(x)

        augment = signatory.Augment(1,
                                    layer_sizes=(),
                                    kernel_size=1,
                                    include_time=True)

        #rough = signatory.Path(augment(Wt), self.depth, basepoint=False)
        rough = signatory.Path(augment(Wt), self.depth, basepoint=False)
        dt = self.equation.T/self.N
        t = torch.linspace(0,1,self.N)
        for i in range(self.N-2,-1,-1):
            #print(str(i))
            target1 = self.y[:,:,i+1]*(Wt[:,i+1,:]-Wt[:,i,:])

            if torch.isnan(target1).any():
                print(str(i) + "target1 nan")
            if torch.isinf(target1).any():
                print(str(i) + "target1 inf")

            cex1 = self.sig_cex(target1,rough,i)
            if torch.isnan(cex1).any():
                print(str(i) + "cex1 nan")
            if torch.isinf(cex1).any():
                print(str(i) + "cex1 inf")
            self.z[:,:,i] = cex1/dt#np.sqrt(dt)

            #Explicit
            ###############################

            # target2 = self.y[:,:,i+1] + self.equation.f(t[i], x[:,:,i],self.y[:,:,i+1],self.z[:,:,i])*dt
            # # if torch.isnan(target2).any():
            # #     print(str(i) + "target2 nan")
            # # if torch.isinf(target2).any():
            # #     print(str(i) + "target2 inf")
            # cex2 = self.sig_cex(target2,rough,i)
            # # if torch.isnan(cex2).any():
            # #     print(str(i) + "cex2 nan")
            # # if torch.isinf(cex2).any():
            # #     print(str(i) + "cex2 inf")
            # self.y[:,:,i] = cex2

            #quasi implicit
            ###############################################
            #y_prev = self.sig_cex(self.y[:,:,i+1],rough,i)
            #self.y[:,:,i] = y_prev + self.equation.f(t[i], x[:,:,i],y_prev,self.z[:,:,i])*dt

            #implicit
            y_prev = self.sig_cex(self.y[:, :, i + 1], rough, i)
            fp_itr = 10
            y_itr = y_prev + self.sig_cex(self.equation.f(t[i], x[:, :, i], self.y[:,:,i+1], self.z[:, :, i]) * dt, rough,i)
            for k in range(fp_itr):
                y_itr = y_prev + self.equation.f(t[i], x[:, :, i], y_itr, self.z[:, :, i]) * dt
            self.y[:,:,i] = y_itr
            #self.y[:,:,i] = y_prev + self.equation.f(t[i], x[:,:,i],y_prev,self.z[:,:,i])*dt

    def L2(self,true,est,N):
        dt = self.equation.T/N
        diff = torch.mean(torch.sum(torch.linalg.norm((true-est)**2,dim=1)*dt, dim=-1),dim=0)
        l2_true = torch.mean(torch.sum(torch.linalg.norm((true)**2,dim=1)*dt, dim=-1),dim=0)
        return float(torch.sqrt(diff/l2_true))
