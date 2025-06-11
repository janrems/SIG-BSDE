import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import signatory
from sklearn.linear_model import Ridge
import importlib






from beta_BSDE import fbsde
from beta_BSDE import Train_NN
from beta_BSDE import Train_NN_Fixed
from beta_BSDE import Train_NN_Simple
from beta_BSDE import BEM_beta




path = "state_dicts/"


new_folder_flag = True
new_folder = "beta1D_simple05/"

if new_folder_flag:
    path = new_folder + path
    if not os.path.exists(path):
        os.makedirs(path)
        os.makedirs(new_folder + "Graphs")
        #path = new_folder + path
    graph_path = new_folder + "Graphs/"


dim_x, dim_y, dim_d, dim_h, N, itr, batch_size = 1, 1, 1,11, 500, 500, 1000
x_0, T, multiplyer = torch.zeros(dim_x), 1, 10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




def b(t, x):
    return torch.zeros(batch_size,dim_x)

def sigma(t, x):
    return torch.ones(batch_size, dim_x, dim_d)



def f(t, x, y, z):
    return -x*y

def g(x):
    return -x



def l(x):
    return torch.ones(batch_size, dim_y)*0.7


#####################################################

equation = fbsde(x_0, b, sigma, f, g, T,dim_x, dim_y, dim_d)


lambda1 = lambda epoch: 0.65 ** epoch
NN_train = Train_NN_Simple(batch_size, itr, 0.001, 30, equation, lambda1)
y_tmp, beta_tmp, r_tmp, R_tmp = NN_train.train()




itr_ax = np.linspace(0,len(NN_train.losses),len(NN_train.losses))
plt.plot(itr_ax, NN_train.losses)
plt.show()


beta_true = NN_train.exact(y_tmp, r_tmp, R_tmp)
errors = torch.abs(beta_true-beta_tmp)

plt.hist(errors.detach().numpy(), bins=100)
plt.savefig(graph_path+"hist_NN_Error.png")
plt.show()


#
plt.scatter(beta_true.detach(),beta_tmp.detach())
plt.ylabel(r'$\beta true$')
plt.xlabel(r'$predict$')
plt.savefig(graph_path+"scatter.png")
plt.show()


plt.scatter(y_tmp.detach(),beta_tmp.detach())
plt.ylabel(r'$\beta$')
plt.xlabel(r'$y$')
#plt.savefig(graph_path+"scatteryb.png")
plt.show()



bb = BEM_beta(equation,batch_size,3,N,NN_train.model)

x,Wt = bb.gen_forward()
r = bb.gen_r()
R = bb.gen_R(r)
y,z, beta = bb.numerical(x,Wt,r, R)

y_b, z_b, beta_b = bb.numerical_exact_beta(x,Wt,r,R)







##########################
# Comparison to solution for fixed beta
w_pos = Wt > 0
y_true = 0.5*np.exp(-(T/N))*(-Wt)*(w_pos) + (~w_pos)*(-Wt)
beta_true = y_true<=0


t = torch.linspace(0,T,N)
time = torch.unsqueeze(t,dim=0)
time = torch.unsqueeze(time,dim=0)
time = torch.repeat_interleave(time, repeats=batch_size, dim=0)

# Solutions for beta = 0, 1, 0.5
y_0 = -Wt
y_1 = torch.exp((-T+time)*0.5)*(-Wt)
y_05 = torch.exp((-T+time)*0.25)*(-Wt)


on = 0

t = torch.linspace(0,T,N)



j = np.random.randint(batch_size)
plt.plot(t[on:],y[j,0,on:].detach(), color= "red", label=r'$\rho_t(X)$')
plt.plot(t[on:],y_0[j,0,on:].detach(), color= "blue", label=r'$\rho_t(X,0)$')
plt.plot(t[on:],y_1[j,0,on:].detach(), color= "green", label=r'$\rho_t(X,\frac{1}{2})$')
plt.plot(t[on:],y_05[j,0,on:].detach(), color= "black", label=r'$\rho_t(X,\frac{1}{4})$')
plt.legend()
plt.savefig(graph_path+"y_t.png")
plt.show()



######################################################
# Plot of y, beta, r, R in the stochastic boundary case

j = np.random.randint(batch_size)
plt.plot(t[on:],y[j,0,on:].detach(), color= "red", label=r'$Y_t$')
plt.plot(t[on:],r[j,0,on:].detach(), color= "blue", label=r'$r_t$')
plt.plot(t[on:],R[j,0,on:].detach(), color= "green", label=r'$R_t$')
plt.plot(t[on:-1],beta[j,0,on:].detach(), color= "black", label=r'$\hat{\beta}_t$', linestyle="--")
plt.plot(t[on:-1],beta_b[j,0,on:].detach(), color= "grey", label=r'$\beta_t$', linestyle="--")
plt.legend()
#plt.savefig(graph_path+"all.png")
plt.show()


