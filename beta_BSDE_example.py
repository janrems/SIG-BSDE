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
    #return torch.ones(batch_size, dim_x)*mu
    #return a*(b_c*torch.ones(batch_size, dim_x) - x)
    return torch.zeros(batch_size,dim_x)

def sigma(t, x):
    return torch.ones(batch_size, dim_x, dim_d)
    #return sig*torch.sqrt(x).reshape(-1, dim_x, dim_d)


def f(t, x, y, z):
    #return (torch.cos(x)*(np.exp((T-t)/2) + 0.5*sig**2) + \
     #       mu*torch.sin(x))*np.exp(0.5*(T-t)) - 0.5*(torch.sin(x)*torch.cos(x)*np.exp(T-t))**2 + 0.5*(y*z)**2
    #return 0.5*beta*z**2
    #return beta*z
    return -x*y

def g(x):
    return -x
    #return torch.exp(beta*x-0.5*T*beta**2)
    #return torch.cos(x)
    #return torch.ones(batch_size,dim_x)


def l(x):
    return torch.ones(batch_size, dim_y)*0.7


###############################################
#bsde_solver = BSDEsolver(equation, dim_h)

#with torch.autograd.set_detect_anomaly(True):
 #   a = bsde_solver.train(batch_size, N, itr)

#####################################################

equation = fbsde(x_0, b, sigma, f, g, T,dim_x, dim_y, dim_d)



lambda1 = lambda epoch: 0.65 ** epoch
NN_train = Train_NN(batch_size, itr, 0.001, 30, equation, lambda1)
y_tmp, beta_tmp, r_tmp, R_tmp = NN_train.train()



lambda1 = lambda epoch: 0.65 ** epoch
NN_train = Train_NN_Fixed(batch_size, itr, 0.001, 30, equation, lambda1)
y_tmp, beta_tmp, r_tmp, R_tmp = NN_train.train()



lambda1 = lambda epoch: 0.65 ** epoch
NN_train = Train_NN_Simple(batch_size, itr, 0.001, 30, equation, lambda1)
y_tmp, beta_tmp, r_tmp, R_tmp = NN_train.train()




itr_ax = np.linspace(0,len(NN_train.losses),len(NN_train.losses))
plt.plot(itr_ax, NN_train.losses)
#plt.savefig(graph_path+"losses.png")
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



#
# plt.scatter(y_tmp.detach(),beta_tmp.detach())
# plt.ylabel(r'$\beta$')
# plt.xlabel(r'$y$')
# plt.savefig(graph_path+"scatter.png")
# plt.show()
#








##########################
w_pos = Wt > 0
y_true = 0.5*np.exp(-(T/N))*(-Wt)*(w_pos) + (~w_pos)*(-Wt)
beta_true = y_true<=0


t = torch.linspace(0,T,N)
time = torch.unsqueeze(t,dim=0)
time = torch.unsqueeze(time,dim=0)
time = torch.repeat_interleave(time, repeats=batch_size, dim=0)

y_0 = -Wt
y_1 = torch.exp((-T+time)*0.5)*(-Wt)
y_05 = torch.exp((-T+time)*0.25)*(-Wt)


on = 0

t = torch.linspace(0,T,N)



j = np.random.randint(batch_size)
plt.plot(t[on:],y[j,0,on:].detach(), color= "red", label=r'$\rho_t(X)$')
#plt.plot(t[on:-1],beta[j,0,on:].detach(), color="blue")
#plt.plot(t[on:-1],beta_true[j,0,on:-1].detach(), color="green")
#plt.plot(t[on:],y_true[j,0,on:].detach(), color= "black")
#plt.plot(t[on:],y_b[j,0,on:].detach(), color= "green")
plt.plot(t[on:],y_0[j,0,on:].detach(), color= "blue", label=r'$\rho_t(X,0)$')
plt.plot(t[on:],y_1[j,0,on:].detach(), color= "green", label=r'$\rho_t(X,\frac{1}{2})$')
plt.plot(t[on:],y_05[j,0,on:].detach(), color= "black", label=r'$\rho_t(X,\frac{1}{4})$')
plt.legend()
plt.savefig(graph_path+"y_t.png")
plt.show()


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
#plt.savefig(graph_path+"all.png")
plt.show()


j = np.random.randint(batch_size)
plt.plot(t[on:],y[j,0,on:].detach()-y_b[j,0,on:].detach(), color= "green")
plt.show()

######################

result = Result(NN_train.model,equation)
result.L2(y_true,y,N)


##################################################################



#yg = result.regenerate(N,x,W,y,z)

j = np.random.randint(batch_size)
plt.plot(t,x[j,0,:].detach())
plt.plot(t,Wt[j,0,:])
plt.show()


#time
time = torch.unsqueeze(t,dim=0)
time = torch.unsqueeze(time,dim=0)
time = torch.repeat_interleave(time, repeats=batch_size, dim=0)
############################


###########################
# Brownian motion
Wt = torch.cumsum(W,dim=-1)
Wt = torch.roll(Wt,1,-1)
Wt[:,:,0] = torch.zeros(batch_size,dim_d)
##########################################

##############################
#ksi = - B_T
#ytrue = torch.exp(0.5*(T-time))*torch.cos(x)
#ytrue = -Wt*torch.exp(time)
#
gamma = np.sqrt(a+2*sig**2)
den = gamma-a+(gamma+a)*torch.exp(gamma*(T-time))
ft = (2*gamma*torch.exp(0.5*(gamma+a)*(T-time))/den)**(2*a*b_c/(sig**2))
gt = 2*(1-torch.exp(gamma*(T-time)))/den
ytrue = torch.exp(x*gt)*ft
#
#ytrue = torch.exp(0.5*(T-time))*torch.cos(x)

#ytrue = -Wt + 0.5*beta*(T-time)
#ytrue = torch.ones(batch_size,1,N)
#
# ytrue = torch.exp(beta*Wt-0.5*time*beta**2) + \
#         np.exp(beta**2*T)*torch.exp(2*beta*Wt-2*time*beta**2) - \
#         torch.exp(2*beta*Wt-time*beta**2)
##############################
#Y thorugh Z
#ztrue = ytrue*gt
ztrue = -torch.exp(0.5*(T-time))*torch.sin(x)
yz = torch.ones(batch_size,dim_y,N)
yz[:,:,-1] = ytrue[:,:,-1]
dt = equation.T/N
for i in range(N-2,-1,-1):
    yz[:,:,i] = yz[:,:,i+1] + f((i+1)*dt,x[:,:,i,], yz[:,:,i], ztrue[:,:,i])*dt - ztrue[:,:,i]*W[:,:,i]



##############################
# #L2 error

result.L2(ytrue,y,N)

result.L2(ytrue,yc,N)

####################################
#SIGNATURE

#ytrue vs y plot comparison

j = np.random.randint(batch_size)
ytruej = ytrue[j,0,:]
#plt.plot(t,y[j,0,:].detach().numpy(), color="red", label="BSDE")
plt.plot(t,yc[j,0,:].detach().numpy(), color="red", label="CEX")
#plt.plot(t,yz[j,0,:],color="black")
#plt.plot(t,y_est[j,0,:].detach().numpy(), color="black", label="SIG")
#plt.plot(t,yg[j,0,:].detach().numpy(), color="black")
plt.plot(t,ytruej, color="blue", label="Analytical")
plt.legend()
plt.savefig(graph_path+str(j))
plt.show()


#ratio metric
j = np.random.randint(batch_size)
ytruej = ytrue[j,0,:]
plt.plot(t,ytruej/y[j,0,:].detach().numpy())
plt.show()

j = np.random.randint(batch_size)
plt.plot(t,z[j,0,0,:].detach().numpy())
plt.show()




k = torch.zeros(batch_size,dim_d,N)
for i in range(N-1):
    k[:,:,i+1] = k[:,:,i] + 2*Wt[:,:,i]*W[:,:,i] + 1/N

j = np.random.randint(batch_size)
plt.plot(t,k[j,0,:], color="red")
plt.plot(t,Wt[j,0,:]**2)
plt.show()


#################################
#loss analysis

#graph a loss at specific time
itr_ax = np.linspace(1,itr*multiplyer,itr*multiplyer)

plt.plot(itr_ax[:],loss[0][:])
plt.show()



itr_ax = np.linspace(1,itr,itr)

plt.plot(itr_ax[:],loss[5][:])
plt.show()


#save loss

loss_np = np.array(loss, )
np.save(path, loss)

plt.plot(t,loss_tensor)
plt.show()


#analize average of last k losses over time

loss_tensor = torch.tensor(loss[2:])

k = 20

last_k = loss_tensor[:,-20:]

loss_k_avg = torch.mean(last_k,dim=1)

plt.plot(t[2:], loss_k_avg)
plt.show()

