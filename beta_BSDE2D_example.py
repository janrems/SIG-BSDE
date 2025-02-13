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
    #return torch.ones(batch_size, dim_x)*mu
    #return a*(b_c*torch.ones(x.shape[0], x.shape[1]) - x)
    return torch.zeros(x.shape[0], x.shape[1])
    #return x*mu

def sigma(t, x):
    #return torch.ones(batch_size, dim_x, dim_d)*sig
    return torch.ones(x.shape[0], x.shape[1], dim_d)
    #return sig*torch.sqrt(x).reshape(-1, x.shape[1], dim_d)
    #return (gamma*x).reshape(-1, dim_x, dim_d)


def f(t, beta, alpha, y, z):
    #return (torch.cos(x)*(np.exp((T-t)/2) + 0.5*sig**2) + \
     #       mu*torch.sin(x))*np.exp(0.5*(T-t)) - 0.5*(torch.sin(x)*torch.cos(x)*np.exp(T-t))**2 + 0.5*(y*z)**2
    #return 0.5*beta*z**2
    #return beta*z
    return -beta*y + alpha*z

def g(x):
    return x
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
#
#
# type = "constant"
# lambda1 = lambda epoch: 0.65 ** epoch
# NN_train = Train_NN(batch_size, itr, 0.001, 100, equation, lambda1, type)
# y_tmp, z_tmp,beta_tmp,alpha_tmp, r_tmp, R_tmp, s_tmp, S_tmp = NN_train.train()
#




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










#################################
plt.plot(itr_ax, NN_train.losses_true)
plt.savefig(graph_path+"losses_true.png")
plt.show()


r, R,s, S, y, z = NN_train.gen_input()
inpt = torch.cat((y, z), 1)
out = NN_train.model(inpt,r, R, s, S)
beta = out[:,0]
alpha = out[:,1]

beta_true, alpha_true = NN_train.exact(y,r,R,s,S,z)
errors = torch.abs(beta_true[:,0]-beta) + torch.abs(alpha_true[:,0]-alpha)

NN_train.loss_function(y, beta_true, alpha_true, z)
NN_train.loss_function(y, beta, alpha, z)



plt.hist(errors.detach().numpy(), bins=100)
plt.savefig(graph_path+"hist_NN_Error.png")
plt.show()


#
plt.scatter(beta_true[:,0].detach(),beta_tmp.detach())
plt.ylabel(r'$\beta true$')
plt.xlabel(r'$predict$')
plt.savefig(graph_path+"scatter.png")
plt.show()


plt.scatter(alpha_true[:,0].detach(),alpha_tmp.detach())
plt.ylabel(r'$\alpha true$')
plt.xlabel(r'$predict$')
plt.savefig(graph_path+"scatter.png")
plt.show()


alpha_wrong = torch.where(torch.abs(alpha-alpha_true[:,0])>0.03, torch.ones(alpha.shape), torch.zeros(alpha.shape))

torch.sum(alpha_wrong)











##########################
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
#plt.plot(t[on:-1],beta[j,0,on:].detach(), color="blue")
#plt.plot(t[on:-1],beta_true[j,0,on:-1].detach(), color="green")
#plt.plot(t[on:],y_true[j,0,on:].detach(), color= "black")
#plt.plot(t[:],y_b[j,0,:].detach(), color= "grey")
plt.plot(t[:],y00[j,0,:].detach(), color= "blue", label=r'$\rho_t(X,0,0)$')
plt.plot(t[:],y50[j,0,:].detach(), color= "green", label=r'$\rho_t(X,\frac{1}{2},0)$')
plt.plot(t[:],y05[j,0,:].detach(), color= "black", label=r'$\rho_t(X,0,\frac{1}{2})$')
plt.plot(t[:],y55[j,0,:].detach(), color= "yellow", label=r'$\rho_t(X,\frac{1}{2},\frac{1}{2})$')
plt.legend()
plt.savefig(graph_path+"y_t.png")
plt.show()

j = np.random.randint(batch_size)
plt.plot(t[:],y[j,0,:].detach(), color= "red", label=r'$\rho_t(X)$')
#plt.plot(t[on:-1],beta[j,0,on:].detach(), color="blue")
#plt.plot(t[on:-1],beta_true[j,0,on:-1].detach(), color="green")
#plt.plot(t[on:],y_true[j,0,on:].detach(), color= "black")
plt.plot(t[:],y_b[j,0,:].detach(), color= "grey")
plt.show()

j = np.random.randint(batch_size)
plt.plot(t[on:-1],z[j,0,on:].detach(), color= "red", label=r'$\rho_t(X)$')
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
plt.savefig(graph_path+"all.png")
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

