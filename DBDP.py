import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#from collections import deque

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")





class fbsde():
    def __init__(self, x_0, b, sigma, gamma, f, g, T, dim_x, dim_y, dim_d, dim_j, l, jump_type, jump_mean, jump_sd, eq_type):
        self.x_0 = x_0
        self.b = b
        self.sigma = sigma
        self.gamma = gamma
        self.f = f
        self.g = g
        self.T = T
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_d = dim_d
        self.dim_j = dim_j
        self.l = l
        self.jump_type = jump_type
        self.jump_mean = jump_mean
        self.jump_sd = jump_sd
        self.eq_type = eq_type



class Model(nn.Module):
    def __init__(self, equation, dim_h):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(equation.dim_x + 1, dim_h)
        self.linear2 = nn.Linear(dim_h, dim_h)
        self.linear3 = nn.Linear(dim_h, dim_h)
        self.linear4 = nn.Linear(dim_h, equation.dim_y + equation.dim_y * equation.dim_d + equation.dim_d * equation.dim_j)


        self.equation = equation

    def forward(self, N, n, x):

        def normalize(x):
            xmax = x.max(dim=0).values
            xmin = x.min(dim=0).values
            return (x-xmin)/(xmax-xmin)

        def standardize(x):
            mean = torch.mean(x,dim=0)
            sd = torch.std(x,dim=0)
            return (x-mean)/sd

        def phi(x):
            x = torch.tanh(self.linear1(x))
            x = torch.tanh(self.linear2(x))
            x = torch.tanh(self.linear3(x))
            return self.linear4(x) #[bs,(dy*dd)] -> [bs,dy,dd]



        delta_t = self.equation.T / N

        x_nor = x
        if n!=0:
            #x_nor = normalize(x)
            x_nor = standardize(x)

        inpt = torch.cat((x_nor, torch.ones(x.size()[0], 1, device=device) * delta_t * n), 1)
        yzu = phi(inpt)
        y = yzu[:,:self.equation.dim_y].clone()
        z = yzu[:,self.equation.dim_y:self.equation.dim_y + self.equation.dim_y * self.equation.dim_d].reshape(-1,self.equation.dim_y, self.equation.dim_d).clone()
        u = yzu[:, self.equation.dim_y + self.equation.dim_y * self.equation.dim_d:].reshape(-1,self.equation.dim_y).clone()
        return y,z,u



class BSDEsolver():
    def __init__(self, equation, dim_h, model,lr,coeff):
        self.model = model
        self.equation = equation
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr*coeff)
        self.dim_h = dim_h


    def loss(self, x, n, y_prev, y, z,u, w, dN, N):
        if n == N-1:
            dist = (y - self.equation.g(x)).norm(2,dim=1)
        else:
            delta_t = self.equation.T / N
            estimate = y - self.equation.f(delta_t*n, x , y, z, u)*delta_t + torch.matmul(z, w).reshape(-1, self.equation.dim_y) + (dN-delta_t)*u
            dist = (y_prev - estimate).norm(2,dim=1)
        return torch.mean(dist)



    def gen_forward(self, batch_size, N,n):
        delta_t = self.equation.T / N
        x = self.equation.x_0 + torch.zeros(batch_size, self.equation.dim_x, device=device, requires_grad=True).reshape(
            -1, self.equation.dim_x)  # [bs,dx]
        if n==0:
            w = torch.randn(batch_size, self.equation.dim_d, 1)*np.sqrt(delta_t)
            rates = torch.ones(batch_size, self.equation.dim_j)*delta_t
            dN = torch.poisson(rates)*self.equation.eq_type
            x_next = x + (self.equation.b(delta_t * 1, x)) * delta_t + torch.matmul(self.equation.sigma(delta_t * 1, x),
                                                                                  w).reshape(-1, self.equation.dim_x) + dN
        else:
            for i in range(n):
                w = torch.randn(batch_size, self.equation.dim_x, 1)*np.sqrt(delta_t)
                rates = torch.ones(batch_size, self.equation.dim_j) * delta_t
                dN = torch.poisson(rates)*self.equation.eq_type
                x = x + (self.equation.b(delta_t * (i+1), x)) * delta_t + torch.matmul(self.equation.sigma(delta_t * (i+1), x),w).reshape(-1, self.equation.dim_x)+dN
            w = torch.randn(batch_size, self.equation.dim_d, 1)*np.sqrt(delta_t)
            rates = torch.ones(batch_size, self.equation.dim_j) * delta_t
            dN = torch.poisson(rates)*self.equation.eq_type
            x_next = x + (self.equation.b(delta_t * (n+1), x)) * delta_t + torch.matmul(self.equation.sigma(delta_t * (n+1), x),w).reshape(-1, self.equation.dim_x) + dN
        return x, w, x_next, dN

    def train(self, batch_size, N, n, itr, path, multiplyer):
        loss_n = []
        if n != N-2:
            mod2 = Model(self.equation, self.dim_h).to(device)
            mod2.load_state_dict(torch.load(path + "state_dict_" + str(n + 1)), strict=False)
            mod2.eval()

        if n == N-2:
            itr_actual = multiplyer*itr
        else:
            itr_actual = itr

        for i in range(itr_actual):

            flag = True
            while flag:
                x, w, x_next, dN = self.gen_forward(batch_size,N, n)
                flag = torch.isnan(x_next).any()




            y,z, u = self.model(N, n, x)

            if n == N-2:
                y_prev = self.equation.g(x)
            else:

                y_prev, z_prev, u_prev = mod2(N,n+1,x_next)


            if 0==1:
                y_prev = torch.maximum(y_prev, self.equation.l(x_next))

            loss = self.loss(x,n,y_prev,y,z,u,w,dN,N)

            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()
            loss_n.append(float(loss))

            #if i%(itr_actual-1) == 0:
                #print("time_"+str(n)+ "iter_"+str(i))
                #for par_group in self.optimizer.param_groups:
                    #print(par_group["lr"])


        return loss_n, y

class BSDEiter():
    def __init__(self, equation, dim_h):
        self.equation = equation
        self.dim_h = dim_h


    def train_whole(self, batch_size, N, path, itr, multiplyer):
        loss_data = []


        for n in range(N-2,-1,-1):
            lr = 0.001
            coeff = 10
            if n==N-1:
                coeff = 1
            print("time "+ str(n))
            mod = Model(self.equation, self.dim_h).to(device)
            bsde_solver = BSDEsolver(self.equation, self.dim_h, mod, lr, coeff)
            if n != N-2:
                #break
                bsde_solver.model.load_state_dict(torch.load(path+"state_dict_" + str(n+1)), strict=False)
                bsde_solver.optimizer.load_state_dict(torch.load(path + "state_dict_opt_" + str(n + 1)))

            loss_n, y = bsde_solver.train(batch_size, N, n, itr, path, multiplyer)
            loss_data.append(loss_n)
            torch.save(bsde_solver.model.state_dict(),path+"state_dict_" + str(n))
            torch.save(bsde_solver.optimizer.state_dict(), path + "state_dict_opt_" + str(n))

        return loss_data














class Result():
    def __init__(self,model, equation):
        self.model = model
        self.equation = equation

    def gen_b_motion(self, batch_size, N):
        delta_t = self.equation.T / N
        W = torch.randn(batch_size, self.equation.dim_d, N, device=device) * np.sqrt(delta_t)

        return W

    def gen_dN(self, batch_size, N):
        delta_t = self.equation.T / N
        rates = torch.ones(batch_size, self.equation.dim_j,N) * delta_t
        dN = torch.poisson(rates)*self.equation.eq_type
        return dN


    def gen_x(self, batch_size, N, W, dN):
        delta_t = self.equation.T / N
        x = self.equation.x_0 + torch.zeros(batch_size, N * self.equation.dim_x, device=device).reshape(-1,self.equation.dim_x, N) #[bs,dx,N]
        for i in range(N-1):
            w = W[:, :, i].reshape(-1, self.equation.dim_d, 1)
            x[:,:,i+1] = x[:,:,i] + self.equation.b(delta_t * i, x[:,:,i]) * delta_t + torch.matmul(self.equation.sigma(delta_t * i, x[:,:,i]),w).reshape(-1, self.equation.dim_x) + dN[:,:,i]
        return x

    def predict(self,N,batch_size,x, path):
        ys = torch.zeros(batch_size, self.equation.dim_y, N)
        zs = torch.zeros(batch_size, self.equation.dim_y, self.equation.dim_d, N)
        us = torch.zeros(batch_size, self.equation.dim_y, N)

        for n in range(N-1):
            self.model.load_state_dict(torch.load(path + "state_dict_" + str(n)), strict=False)
            y,z, u = self.model(N, n, x[:,:,n])
            if 0==1:
                y = torch.maximum(y,self.equation.l(x[:,:,n]))
            ys[:,:,n] = y
            zs[:,:,:,n] = z
            us[:,:,n] = u
        ys[:,:,N-1] = self.equation.g(x[:,:,N-1])
        return ys, zs, us

    def regenerate(self, N, x, W, y, z):
        delta_t = self.equation.T/N
        y_g = y
        for n in range(N-1):
            w = W[:, :, n].reshape(-1, self.equation.dim_d, 1)
            y_g[:,:,n+1] = y[:,:,n] - self.equation.f(delta_t*n, x[:,:,n] ,y[:,:,n], z[:,:,:,n])*delta_t + torch.matmul(z[:,:,:,n], w).reshape(-1, self.equation.dim_y)

        return y_g

    def L2(self,true,est,N):
        dt = self.equation.T/N
        diff = torch.mean(torch.sum(torch.linalg.norm((true-est)**2,dim=1)*dt, dim=-1),dim=0)
        l2_true = torch.mean(torch.sum(torch.linalg.norm((true)**2,dim=1)*dt, dim=-1),dim=0)
        return float(torch.sqrt(diff/l2_true))
