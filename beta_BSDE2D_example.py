import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import signatory
from sklearn.linear_model import Ridge
import importlib







from beta_BSDE2D import fbsde
from beta_BSDE2D import Train_NN
from beta_BSDE2D import Train_NN_Fixed
from beta_BSDE2D import Train_NN_Simple
from beta_BSDE2D import BEM_beta




path = "state_dicts/"


new_folder_flag = True
new_folder = "ambiguous_2d_XB/"

if new_folder_flag:
    path = new_folder + path
    if not os.path.exists(path):
        os.makedirs(path)
        os.makedirs(new_folder + "Graphs")
        #path = new_folder + path
    graph_path = new_folder + "Graphs/"


dim_x, dim_y, dim_d, dim_h, N, itr, batch_size = 1, 1, 1,11, 500, 500, 1000
x_0, T, multiplyer = torch.ones(dim_x)*0, 1, 10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



beta = 0.5
alpha = 0.5
gamma = 0.3





def b(t, x):
    return torch.zeros(x.shape[0], x.shape[1])

def sigma(t, x):
    return torch.ones(x.shape[0], x.shape[1], dim_d)



def f(t, beta, alpha, y, z):
    return -beta*y + alpha*z

def g(x):
    return x



def l(x):
    return torch.ones(batch_size, dim_y)*0.7


#####################################################

equation = fbsde(x_0, b, sigma, f, g, T,dim_x, dim_y, dim_d)

#############################
#Deterministic boundaries case

lambda1 = lambda epoch: 0.65 ** epoch
NN_train = Train_NN_Fixed(batch_size, itr, 0.001, 100, equation, lambda1)
y_tmp, z_tmp,beta_tmp,alpha_tmp = NN_train.train()

bb = BEM_beta(equation,batch_size,3,N,NN_train.model)

x,Wt = bb.gen_forward()
r = bb.gen_r()
R = bb.gen_R(r)
y,z, beta = bb.numerical(x,Wt,r, R)

y00,_,_ = bb.numerical_fixed_beta(x,Wt,0,0)
y05,_,_ = bb.numerical_fixed_beta(x,Wt,0.5,0)
y50,_,_ = bb.numerical_fixed_beta(x,Wt,0,0.5)
y55,_,_ = bb.numerical_fixed_beta(x,Wt,0.5,0.5)

y_b, z_b, beta_b = bb.numerical_exact_beta(x,Wt,r,R)


inpt = NN_train.gen_input()


inpt = torch.cat((y[:,:,-2], z[:,:,-1]), 1)
y = inpt[:,0]
z = inpt[:,1]
out = NN_train.model(inpt)
beta = out[:,0]
alpha = out[:,1]

beta_true, alpha_true = NN_train.exact(y,z)

beta_true, alpha_true = NN_train.exact(y[:,:,-2], z[:,:,-1])

errors = torch.abs(beta_true[:,0]-beta) + torch.abs(alpha_true[:,0]-alpha)
errors = torch.abs(beta_true-beta) + torch.abs(alpha_true-alpha)




#############################
#Stochastic boundaries case

lambda1 = lambda epoch: 0.65 ** epoch
NN_train = Train_NN_Simple(batch_size, itr, 0.001, 100, equation, lambda1)
y_tmp, z_tmp,beta_tmp,alpha_tmp = NN_train.train()




itr_ax = np.linspace(0,len(NN_train.losses),len(NN_train.losses))
plt.plot(itr_ax, NN_train.losses)
plt.savefig(graph_path+"losses.png")
plt.show()

inpt,lower,upper = NN_train.gen_input()
y = inpt[:,0]
z = inpt[:,1]
out = NN_train.model(inpt, lower, upper)
beta = out[:,0]
alpha = out[:,1]

beta_true, alpha_true = NN_train.exact(y,z,lower, upper)

errors = torch.abs(beta_true-beta) + torch.abs(alpha_true-alpha)

###############################
#Plots

plt.hist(errors.detach().numpy(), bins=100)
#plt.savefig(graph_path+"hist_NN_Error.png")
plt.show()


#
plt.scatter(beta_true.detach(),beta.detach())
plt.xlabel(r'$\beta true$')
plt.ylabel(r'$predict$')
#plt.savefig(graph_path+"scatter.png")
plt.show()


plt.scatter(y.detach(),beta.detach())
plt.ylabel(r'$\beta$')
plt.xlabel(r'$y$')
#plt.savefig(graph_path+"scatteryb.png")
plt.show()


plt.scatter(z.detach(),alpha.detach())
plt.ylabel(r'$\alpha$')
plt.xlabel(r'$z$')
plt.savefig(graph_path+"scatterza.png")
plt.show()


plt.scatter(y.detach(),beta.detach(), label=r'$(y,\beta)$', color="blue")
plt.scatter(z.detach(),alpha.detach(), label=r'$(z,\alpha)$', color="red")
plt.ylabel(r'$(\beta, \alpha)$')
plt.xlabel(r'$(y, z)$')
plt.legend()
plt.savefig(graph_path+"scatter_both.png")
plt.show()

plt.scatter(alpha_true.detach(),alpha.detach())
plt.xlabel(r'$\alpha true$')
plt.ylabel(r'$predict$')
#plt.savefig(graph_path+"scatter.png")
plt.show()





##########################
# Solutions Y for fixed beta and alpha

w_pos = Wt > 0
y_true = 0.5*np.exp(-(T/N))*(-Wt)*(w_pos) + (~w_pos)*(-Wt)
beta_true = y_true<=0


t = torch.linspace(0,T,N)
time = torch.unsqueeze(t,dim=0)
time = torch.unsqueeze(time,dim=0)
time = torch.repeat_interleave(time, repeats=batch_size, dim=0)

beta_f=0
alpha_f=0
y00 = torch.exp(gamma*Wt + T*alpha_f*gamma + 0.5*time*(alpha_f**2-(alpha_f+gamma)**2) - beta_f*(T-time))

beta_f=0.5
alpha_f=0
y05 = torch.exp(gamma*Wt + T*alpha_f*gamma + 0.5*time*(alpha_f**2-(alpha_f+gamma)**2) - beta_f*(T-time))

beta_f=0
alpha_f=0.5
y50 = torch.exp(gamma*Wt + T*alpha_f*gamma + 0.5*time*(alpha_f**2-(alpha_f+gamma)**2) - beta_f*(T-time))




beta_f=0.5
alpha_f=0.5
y55 = torch.exp(gamma*Wt + T*alpha_f*gamma + 0.5*time*(alpha_f**2-(alpha_f+gamma)**2) - beta_f*(T-time))




t = torch.linspace(0,T,N)
on=0


j = np.random.randint(batch_size)
plt.plot(t[:],y[j,0,:].detach(), color= "red", label=r'$\rho_t(X)$')
plt.plot(t[:],y00[j,0,:].detach(), color= "blue", label=r'$\rho_t(X,0,0)$')
plt.plot(t[:],y50[j,0,:].detach(), color= "green", label=r'$\rho_t(X,\frac{1}{2},0)$')
plt.plot(t[:],y05[j,0,:].detach(), color= "black", label=r'$\rho_t(X,0,\frac{1}{2})$')
plt.plot(t[:],y55[j,0,:].detach(), color= "yellow", label=r'$\rho_t(X,\frac{1}{2},\frac{1}{2})$')
plt.legend()
plt.savefig(graph_path+"y_t.png")
plt.show()


#Realizations of Y, beta, alpha, r and R in stochastic boundary case

j = np.random.randint(batch_size)
plt.plot(t[on:],y[j,0,on:].detach(), color= "red", label=r'$Y_t$')
#plt.plot(t[on:-1],beta[j,0,on:].detach(), color="blue")
#plt.plot(t[on:-1],beta_true[j,0,on:-1].detach(), color="green")
#plt.plot(t[on:],y_true[j,0,on:].detach(), color= "black")
#plt.plot(t[on:],y_b[j,0,on:].detach(), color= "green")
plt.plot(t[on:],r[j,0,on:].detach(), color= "blue", label=r'$r_t$')
plt.plot(t[on:],R[j,0,on:].detach(), color= "green", label=r'$R_t$')
plt.plot(t[on:-1],beta[j,0,on:].detach(), color= "black", label=r'$\hat{\beta}_t$', linestyle="--")
plt.plot(t[on:-1],beta_b[j,0,on:].detach(), color= "grey", label=r'$\beta_t$', linestyle="--")
plt.legend()
plt.savefig(graph_path+"all.png")
plt.show()

