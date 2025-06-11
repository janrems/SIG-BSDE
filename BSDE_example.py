import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import signatory
from sklearn.linear_model import Ridge
import importlib







from DBDP import fbsde
from DBDP import BSDEiter
from DBDP import Model
from DBDP import Result

from BSDE_CEX import BEM
from BSDE_CEX import fbsde

path = "state_dicts/"


new_folder_flag = True
new_folder = "maxminB/"

if new_folder_flag:
    path = new_folder + path
    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.exists(new_folder + "Graphs"):
        os.makedirs(new_folder + "Graphs")
        #path = new_folder + path
    graph_path = new_folder + "Graphs/"
ref_flag = False

dim_x, dim_y, dim_d, dim_h, N, itr, batch_size = 1, 1, 1,11, 500, 100, 2**13
x0_value, T, multiplyer = 0.0, 1, 10
x_0 = torch.ones(dim_x)*x0_value

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

beta = 0.1
alpha = 0.0
gamma = 0.2
mu = 0.06
sig = 0.3
a = 1
b_c = 1
r = 0.1
R=0.06
K=100
theta = (mu-r)/sig

bm_min=-0.5
bm_max = 0.5



#FBSDE Coefficients for different examples

def b(t, x):
    #return torch.ones(batch_size, dim_x)*mu
    #return a*(b_c*torch.ones(x.shape[0], x.shape[1]) - x)
    return torch.zeros(x.shape[0], x.shape[1])
    #return x*mu

def sigma(t, x):
    #return torch.ones(batch_size, dim_x, dim_d)
    return torch.ones(x.shape[0], x.shape[1], dim_d)
    #return sig*torch.sqrt(x).reshape(-1, x.shape[1], dim_d)
    #return (gamma*x).reshape(-1, dim_x, dim_d)


def f(t, x, y, z):
    #return (torch.cos(x)*(np.exp((T-t)/2) + 0.5*sig**2) + \
     #       mu*torch.sin(x))*np.exp(0.5*(T-t)) - 0.5*(torch.sin(x)*torch.cos(x)*np.exp(T-t))**2 + 0.5*(y*z)**2
    return 0.5*beta*z**2
    #return -beta*y + alpha*z
    #return -x*y
    #return -(y*r + z*theta) #+(R-r)*torch.minimum(y-z/theta,torch.zeros(y.shape)))
    #return beta*torch.abs(z)
def g(x):
    x = x[:,:,-1]
    #return -x
    return -torch.maximum(torch.ones_like(x)*(bm_min), torch.minimum(torch.ones_like(x)*bm_max, x))
    #return torch.exp(beta*x-0.5*T*beta**2)
    #return torch.cos(x)
    #return torch.ones(x.shape[0], x.shape[1])
    #return torch.maximum(x-K, torch.zeros(x.shape))
    #return x**2
    #return torch.cos(x)

def g_path(x):
    sum = torch.sum(x,dim=-1)/N
    return torch.maximum(sum-K, torch.zeros(sum.shape))

def l(x):
    return torch.ones(batch_size, dim_y)*0.7


#####################################################

equation = fbsde(x_0, b, sigma, f, g, T,dim_x, dim_y, dim_d, l, ref_flag)


#Here we preform multiple runs of the algorithm when needed

###############################
#Multiple iterations for histogram purpose
y0s = []
for j in range(100):
    print("Round: " + str(j))
    bem = BEM(equation, batch_size, 3, N)
    x, Wt = bem.gen_forward()
    bem.numerical(x, Wt)
    y0s.append(float(bem.y[0,0,0]))

y0s = np.array(y0s)
print(np.mean(y0s))
print(np.std(y0s))

bmin = -bm_min

############################################
#MC for cut off Brownian motion in entropic
E0 = ((np.exp(-beta*(-bmin))*0.5*(1+math.erf((-bmin)/(np.sqrt(2*T)))) +
      np.exp(-beta*bm_max)*0.5*(1-math.erf(bm_max/(np.sqrt(2*T))))) +
      0.5*np.exp(0.5*T*beta**2)*(math.erf((bmin-beta*T)/(np.sqrt(2*T))) + math.erf((bm_max+beta*T)/(np.sqrt(2*T)))))



y0MC = np.log(E0)/beta

plt.hist(y0s,bins=10)
plt.vlines(y0MC,0,25, color="red")
plt.savefig(graph_path+"y0_hist_beta01_01.jpg")
plt.show()


bem = BEM(equation, batch_size, 3, N)
x,Wt = bem.gen_forward()
bem.numerical(x,Wt)
########################################################
print(float(bem.y[0,0,0]))


y_0 = []
equation = fbsde(x_0, b, sigma, f, g, T,dim_x, dim_y, dim_d, l, ref_flag)
for j in range(50):
    print("Round: " + str(j))
    bem = BEM(equation, batch_size, 3, N)
    x, Wt = bem.gen_forward()
    bem.numerical(x, Wt)
    y_0.append(float(bem.y[0,0,0]))
y_0 = np.array(y_0)
np.mean(y_0)
np.std(y_0)


############################################
#Deep learning method call for comparison




model = Model(equation, dim_h)
model.eval()
result = Result(model, equation)

flag = True
while flag:
    W = result.gen_b_motion(batch_size, N)
    x = result.gen_x(batch_size, N, W)
    flag = torch.isnan(x).any()

###########################
# Brownian motion
Wt = torch.cumsum(W,dim=-1)
Wt = torch.roll(Wt,1,-1)
Wt[:,:,0] = torch.zeros(batch_size,dim_d)
##########################################

y, z = result.predict(N, batch_size, x, path)




#EXPLICIT SOLUTIONS FOR DIFFERENT EXAMPLES
##################################################################

t = torch.linspace(0,T,N)



#time
t = torch.linspace(0,T,N)
time = torch.unsqueeze(t,dim=0)
time = torch.unsqueeze(time,dim=0)
time = torch.repeat_interleave(time, repeats=batch_size, dim=0)
############################


###########################
# Brownian motion
Wt = torch.cumsum(Wt,dim=-1)
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

ytrue = -Wt + 0.5*beta*(T-time)
ytrue = Wt + 0.5*beta*(T-time)
#ytrue = torch.ones(batch_size,1,N)
#
ytrue_old = torch.exp(beta*Wt-0.5*time*beta**2) + \
        np.exp(beta**2*T)*torch.exp(2*beta*Wt-2*time*beta**2) - \
        torch.exp(2*beta*Wt-time*beta**2)

ytrue = torch.exp(beta*Wt + T*beta**2 - 1.5*(beta**2)*time)

#y+Z
ytrue = torch.exp(beta*Wt + T*beta**2 - 1.5*(beta**2)*time - alpha*(T-time))



#y+Z, gamma
ytrue = torch.exp(gamma*Wt + T*alpha*gamma + 0.5*time*(alpha**2-(alpha+gamma)**2) - beta*(T-time))


normal = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
ytrue = 1/(2*beta**2)+torch.sqrt((T-time)/(2*np.pi))*(torch.abs(Wt)+beta*(T-time)+1/beta)*\
         torch.exp(-(torch.abs(Wt)+beta*(T-time))**2/(2*(T-time))) +\
          ((torch.abs(Wt)+beta*(T-time))**2+(T-time)-1/(2*beta**2))*normal.cdf((torch.abs(Wt)+beta*(T-time))/torch.sqrt(T-time)) +\
        torch.exp(-2*beta*torch.abs(Wt))*(torch.abs(Wt)+(T-time)-1/(2*beta**2))+normal.cdf(-(torch.abs(Wt)-beta*(T-time))/torch.sqrt(T-time))

diff_T = ytrue[:,0,-1] - Wt[:,0,-1]**2
torch.norm(diff_T)

Y0_true = 1/(2*beta**2) + np.sqrt(T/(2*np.pi))*(beta*T+1/beta)*np.exp((beta*T)**2/(2*T)) + \
          ((beta*T)**2+T-1/(2*beta**2))*normal.cdf(torch.tensor([beta*T/np.sqrt(T)])) + \
          (T-1/(2*beta**2))*normal.cdf(torch.tensor([beta*T/np.sqrt(T)]))
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
# #L2 errors

#result.L2(ytrue,bem.y,N)
bem.L2(ytrue,bem.y,N)

bem.L2(ytrue_old,bem.y,N)



result.L2(ytrue,yc,500)

####################################
#Plots

j = np.random.randint(batch_size)
plt.plot(t,bem.y[j,0,:].detach().numpy(), color="red", label="BSDE")
#plt.plot(t,yc[j,0,:].detach().numpy(), color="red", label="CEX")
#plt.plot(t,yz[j,0,:],color="black")
#plt.plot(t,y_est[j,0,:].detach().numpy(), color="black", label="SIG")
#plt.plot(t,yg[j,0,:].detach().numpy(), color="black")
plt.plot(t,ytrue[j,0,:], color="blue", label="Analytical")
#plt.plot(t,ytrue_old[j,0,:], color="green", label="Analytical old")
#plt.plot(t,Wt[j,0,:]**2)
plt.legend()
#plt.savefig(graph_path+str(j))
plt.show()

j = np.random.randint(batch_size)
plt.plot(t[:-1],bem.z[j,0,:].detach().numpy(), color="red", label="BSDE")
plt.show()

plt.hist(Wt[:,0,-1])
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

