import numpy as np
import matplotlib.pyplot as plt
# %matplotlib widget
import scipy.signal
import scipy.linalg
from scipy.integrate import solve_ivp

from __future__ import division
import pyomo.environ as pyo
import numpy as np
import polytope as pt

# Real parameters of the system
m = 5 #mass
b = 0.1 #mass moment of inertia
J = 0.2  #damping corresponding to linear velocity
c = 0.1  #damping corresponding to angular velocity
dT = 0.1 #sampling interval

one_over_m = 1/m
b_eff = (1-b*dT/m)
one_over_J = 1/J
c_eff = (1-c*dT/m)
params_real = np.array([[one_over_m],[b_eff],[one_over_J],[c_eff],[dT]])
print(params_real)

# Parameter adaptation algorithm
def ParameterAdaptation(paramv_prev,paramw_prev,Fv_prev,Fw_prev,Regressorv,Regressorw,errorv,errorw):

    Fv_next = Fv_prev-(Fv_prev@(Regressorv@np.transpose(Regressorv))@Fv_prev)/(1+np.transpose(Regressorv)@Fv_prev@Regressorv)
    paramv_next = paramv_prev+errorv*(Fv_next@Regressorv)

    Fw_next = Fw_prev-(Fw_prev@(Regressorw@np.transpose(Regressorw))@Fw_prev)/(1+np.transpose(Regressorw)@Fw_prev@Regressorw)
    paramw_next = paramw_prev+errorw*(Fw_next@Regressorw)

    return paramv_next.reshape(2,),paramw_next.reshape(2,),Fv_next,Fw_next

# Dynamics of the real system
def RealSystem(CurrState,Controls,params_real):
    one_over_m = params_real[0,0]
    b_eff = params_real[1,0]
    one_over_J = params_real[2,0]
    c_eff = params_real[3,0]
    dT = params_real[4,0]


    NextState = np.zeros((5,1))
    NextState[0,0] = CurrState[0,0]+CurrState[3,0]*np.cos(CurrState[2,0])*dT
    NextState[1,0] = CurrState[1,0]+CurrState[3,0]*np.sin(CurrState[2,0])*dT
    NextState[2,0] = CurrState[2,0]+CurrState[4,0]*dT
    NextState[3,0] = b_eff*CurrState[3,0]+Controls[0,0]*dT*one_over_m
    NextState[4,0] = c_eff*CurrState[4,0]+Controls[1,0]*dT*one_over_J
    return NextState

# Constrained finite time optimal control
def solve_cftoc(P, Q, R, N, x0, FL, FU, TL, TU, bf, Af,params):

    one_over_m = params[0,0]
    b_eff = params[1,0]
    one_over_J = params[2,0]
    c_eff = params[3,0]
    dT = params[4,0]

    model = pyo.ConcreteModel()
    model.N = N
    model.nx = 5
    model.nu = 2
    model.nf = np.size(Af, 0)

    # length of finite optimization problem:
    model.tIDX = pyo.Set( initialize= range(model.N+1), ordered=True )
    model.xIDX = pyo.Set( initialize= range(model.nx), ordered=True )
    model.uIDX = pyo.Set( initialize= range(model.nu), ordered=True )
    model.nfIDX = pyo.Set( initialize= range(model.nf), ordered=True )

    # these are 2d arrays:

    model.Q = Q
    model.P = P
    model.R = R
    model.Af = Af
    model.bf = bf
    # Create state and input variables trajectory:
    model.x = pyo.Var(model.xIDX, model.tIDX)
    model.u = pyo.Var(model.uIDX, model.tIDX)

    #Objective:
    def objective_rule(model):
        costX = 0.0
        costU = 0.0
        costTerminal = 0.0
        for t in model.tIDX:
            for i in model.xIDX:
                for j in model.xIDX:
                    if t < model.N:
                        costX += (model.x[i, t]-model.bf[i,0]) * model.Q[i, j] * (model.x[j, t]-model.bf[i,0])
        for t in model.tIDX:
            for i in model.uIDX:
                for j in model.uIDX:
                    if t < model.N:
                        costU += model.u[i, t] * model.R[i, j] * model.u[j, t]
        for i in model.xIDX:
            for j in model.xIDX:
                costTerminal += (model.x[i, model.N]-model.bf[i,0]) * model.P[i, j] * (model.x[j, model.N]-model.bf[i,0])
        return costX + costU + costTerminal

    model.cost = pyo.Objective(rule = objective_rule, sense = pyo.minimize)

    # Constraints:
    model.init_const1 = pyo.Constraint(expr = model.x[0, 0] == x0[0])
    model.init_const2 = pyo.Constraint(expr = model.x[1, 0] == x0[1])
    model.init_const3 = pyo.Constraint(expr = model.x[2, 0] == x0[2])
    model.init_const4 = pyo.Constraint(expr = model.x[3, 0] == x0[3])
    model.init_const5 = pyo.Constraint(expr = model.x[4, 0] == x0[4])

    model.dynamicsx = pyo.Constraint(model.tIDX, rule=lambda model, t: model.x[0, t+1] == model.x[0,t]+model.x[3,t]*pyo.cos(model.x[2, t])*dT if t < N else pyo.Constraint.Skip)
    model.dynamicsy = pyo.Constraint(model.tIDX, rule=lambda model, t: model.x[1, t+1] == model.x[1,t]+model.x[3,t]*pyo.sin(model.x[2, t])*dT if t < N else pyo.Constraint.Skip)
    model.dynamicsp = pyo.Constraint(model.tIDX, rule=lambda model, t: model.x[2, t+1] == model.x[2,t]+model.x[4,t]*dT if t < N else pyo.Constraint.Skip)
    model.dynamicsv = pyo.Constraint(model.tIDX, rule=lambda model, t: model.x[3, t+1] == b_eff*model.x[3,t]+model.u[0,t]*dT*one_over_m if t < N else pyo.Constraint.Skip)
    model.dynamicsw = pyo.Constraint(model.tIDX, rule=lambda model, t: model.x[4, t+1] == c_eff*model.x[4,t]+model.u[1,t]*dT*one_over_J if t < N else pyo.Constraint.Skip)

    model.control_limit1 = pyo.Constraint(model.tIDX, rule=lambda model, t: model.u[0, t] <= FU
                                   if t < N else pyo.Constraint.Skip)
    model.control_limit2 = pyo.Constraint(model.tIDX, rule=lambda model, t: FL <= model.u[0, t]
                                    if t < N else pyo.Constraint.Skip)
    model.control_limit3 = pyo.Constraint(model.tIDX, rule=lambda model, t: model.u[1, t] <= TU
                                   if t < N else pyo.Constraint.Skip)
    model.control_limit4 = pyo.Constraint(model.tIDX, rule=lambda model, t: TL <= model.u[1, t]
                                    if t < N else pyo.Constraint.Skip)


    def final_const_rule(model, i):
        if model.Af.size == 0:
            k = model.x[:,model.N] == model.bf[i]
        else:
            k = sum(model.Af[i, j] * model.x[j, model.N] for j in model.xIDX) <= model.bf[i]
        return k

    model.final_const = pyo.Constraint(model.nfIDX, rule=final_const_rule)

    solver = pyo.SolverFactory('ipopt')
    results = solver.solve(model)

    if str(results.solver.termination_condition) == "optimal":
        feas = True
    else:
        feas = False

    xOpt = np.asarray([[model.x[i,t]() for i in model.xIDX] for t in model.tIDX]).T
    uOpt = np.asarray([model.u[:,t]() for t in model.tIDX]).T

    JOpt = model.cost()

    return [model, feas, xOpt, uOpt, JOpt]

# Test
import scipy as cp
import scipy.signal
import scipy.linalg
Ts = 0.1       # Ts is the discrete sample-time.

P = 10*np.identity(5)
Q = 1.5*np.identity(5)
R = np.array([[1,0],[0,2]])
N = 500
x0 = np.array([1,1,0*np.pi/2,0,0])
FL = -0.5
FU =  0.5
TL = -0.1
TU =  0.1
Af = []
bf = np.array([[0],[0],[0*np.pi/2],[0],[0]])
# params = np.array([[10],[10]])
[model, feas, xOpt, uOpt, JOpt] = solve_cftoc(P, Q, R, N, x0, FL, FU, TL, TU, bf, Af,params_real)
print('JOpt=', JOpt)
print('xOpt=', xOpt)

fig = plt.figure(figsize=(9, 6))
fig.suptitle('Optimal States from CTFOC results', fontsize=16)
legend_val = ['X Position','Y Position','Heading Angle','Velocity','Angular Velocity'] # need to fill out
for i in range(5):
  plt.plot(xOpt[i], label = legend_val[i])
plt.ylabel('x')
plt.xlabel('Time')
plt.legend()

fig = plt.figure(figsize=(9, 6))
fig.suptitle('Path of Robot from CTFOC results', fontsize=16)
plt.plot(xOpt[0,:],xOpt[1,:])
plt.ylabel('Y')
plt.xlabel('X')

fig = plt.figure(figsize=(9, 6))
fig.suptitle('Optimal Input Values from CTFOC results', fontsize=16)
legend_val = ['Thrust in the direction of the robot','Moment about the Z direction']
for i in range(2):
  plt.plot(uOpt[i], label = legend_val[i])
plt.ylabel('U')
plt.xlabel('Time')

# Receding Horizon MPC
import numpy as np
# Write your code here:

P = 10*np.identity(5) #Terminal cost matrix
Q = 0*np.identity(5) #Path cost matrix
R = np.identity(2) #Input cost matrix

M = 500 #total horizon
N = 30  #mpc horizon

x0 = np.array([1,1,0*np.pi/2,0,0]) #initial state

FL = -1 #lower limit on the thrust
FU =  1 #upper limit on the thrust
TL = -0.1 #lower limit on the moment
TU =  0.1 #upper limit on the moment

Af = [] #not necessary for our case
bf = np.array([[0],[0],[0*np.pi/2],[0],[0]]) #desired final position

nx = 5 #number of states
nu = 2 #number of control inputs

params_v = np.zeros((2,M+1)) #parameters to be estimated corresponding to the linear velocity (mass and damping)
params_w = np.zeros((2,M+1)) #parameters to be estimated corresponding to the angular velocity (inertia and damping)
params   = np.zeros((5,M+1)) #stacking up both the parameters - 5th parameter is the sampling time, doesn't change - actually redundant
params_v[:,0] = np.array([0.09,0.9]).reshape(2,) #initail guess for the parameters
params_w[:,0] = np.array([4.5,0.9]).reshape(2,) #initial guess for the parameters
params[:,0] = np.vstack((params_v[:,0].reshape(2,1),params_w[:,0].reshape(2,1),params_real[4,0])).reshape(5,) #stacked
#matrices used in the parameter adaptation algorithm
sigma_v = 1e+10
sigma_w = 1e+10
F_v = np.zeros((2,2,M+1))
F_w = np.zeros((2,2,M+1))
F_v[:,:,0] = sigma_v*np.identity(2)
F_w[:,:,0] = sigma_w*np.identity(2)



xOpt = np.zeros((nx, M+1))
xOpt[:, 0] = x0.reshape(nx, )

uOpt = np.zeros((nu, M))

x_real = np.zeros((nx, M+1))
x_real[:,0] = x0.reshape(nx, )

xPred = np.zeros((nx, N+1, M))
predErr = np.zeros((nx, M-N+1))

feas = np.zeros((M, ), dtype=bool)
xN = np.zeros((nx,1))

[_,_,xOptM,_,_] = solve_cftoc(P, Q, R, N, x0, FL, FU, TL, TU, bf, Af,params_real) #optimal trajectory with real paramters
#print(xOptM)
#fig = plt.figure(figsize=(9, 6))
# MPC
i = 1
fig = plt.figure(figsize=(9, 6))
fig.suptitle('Robot Trajectories', fontsize=16)
for t in range(M):
    # print(t)
    [model, feas[t], x, u, J] = solve_cftoc(P, Q, R, N, xOpt[:, t], FL, FU, TL, TU, bf, Af,params[:,t].reshape(5,1))
    #print(x)
    if not feas[t]:
        xOpt = []
        uOpt = []
        predErr = []
        break
    if t<M-N+1:
        predErr[0,t] = np.linalg.norm(xOptM[0,t:t+N]-xOpt[0,t])
        predErr[1,t] = np.linalg.norm(xOptM[1,t:t+N]-xOpt[1,t])
        # print(predErr[:,t])
    # Save open loop predictions
    xPred[:, :, t] = x

    # Save closed loop trajectory
    # Note that the second column of x represents the optimal closed loop state
    xOpt[:, t+1] = x[:, 1]
    x_real[:,t+1] = RealSystem(x_real[:,t].reshape(nx,1),u[:,0].reshape(nu,1),params_real).reshape(5,)
    uOpt[:, t] = u[:, 0].reshape(nu, )
    Regressor_v = np.array([[u[0,0]*params_real[4,0]],[x_real[3,t]]])
    Regressor_w = np.array([[u[1,0]*params_real[4,0]],[x_real[4,t]]])
    error_v = x_real[3,t+1]-params_v[:,t].T@Regressor_v
    error_w = x_real[4,t+1]-params_w[:,t].T@Regressor_w
    params_v[:,t+1],params_w[:,t+1],F_v[:,:,t+1],F_w[:,:,t+1] = ParameterAdaptation(params_v[:,t].reshape(2,1),params_w[:,t].reshape(2,1),F_v[:,:,t],F_w[:,:,t],Regressor_v,Regressor_w,error_v,error_w)
    params[:,t+1] = np.vstack((params_v[:,t+1].reshape(2,1),params_w[:,t+1].reshape(2,1),params_real[4,0])).reshape(5,)
    if i==0 or i == 10 or i == 30 or i == 60 or (i%50 == 0 and i > 60):
      plt.plot(x_real[0,:],x_real[1,:], label = 'Iteration: ' + str(i))
    i+=1
    # print(params[:,t]-np.transpose(params_real))
    # Plot Open Loop
    #line1 = plt.plot(x[0,:], x[1,:], 'r--')
plt.ylabel('Y')
plt.xlabel('X')
plt.legend()
plt.show()
# Plot Closed Loop
#line2 = plt.plot(xOpt[0, :], xOpt[1, :], 'bo-')
#plt.legend([line1[0], line2[0]], ['Open-loop', 'Closed-loop']);
#plt.xlabel('x1')
#plt.ylabel('x2')
#plt.axis('equal')

#print(predErr)
#predErr[1].size

# fig = plt.figure(figsize=(9, 6))
# plt.plot(x_real.T)
# plt.ylabel('x')

fig = plt.figure(figsize=(9, 6))
fig.suptitle('Optimal States with Adaptive MPC', fontsize=16)
legend_val = ['X Position','Y Position','Heading Angle','Velocity','Angular Velocity'] # need to fill out
for d in range(5):
  plt.plot(x_real[d], label = legend_val[d])
plt.ylabel('x')
plt.xlabel('Time')
plt.legend()

fig = plt.figure(figsize=(9, 6))
fig.suptitle('Final Robot Trajectory after ' +str(i)+ ' iterations', fontsize=16)
plt.plot(x_real[0,:],x_real[1,:])
plt.ylabel('Y')
plt.xlabel('X')


fig = plt.figure(figsize=(9, 6))
fig.suptitle('Optimal Input Values with Adaptive MPC', fontsize=16)
legend_val = ['Thrust in the direction of the robot','Moment about the Z direction'] # need to fill out
for i in range(2):
  plt.plot(uOpt[i], label = legend_val[i])
plt.ylabel('U')
plt.xlabel('Time')
plt.legend()
plt.show()
plt.show()

# Animation of final trajectory

import matplotlib.pyplot as plt
from matplotlib import animation, rc
from IPython.display import HTML
import numpy as np

fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(111, autoscale_on=False)
ax.axis([-1, 2, -1, 2])
ax.set_aspect('equal')
ax.grid()

lr = 0.1
lf = 0.1
dT = 0.1

# define trend lines
xtrend = x_real[0, :]
ytrend = x_real[1, :]
vtrend = x_real[3, :]
psitrend = x_real[2, :]

# set lines
linef, = ax.plot([], [], 'o-', lw=100)
liner, = ax.plot([], [], 'o-', lw=100)
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

def init():
    linef.set_data([], [])
    liner.set_data([], [])
    time_text.set_text('')
    return linef, liner, time_text

def animate(i):
    Xf = xtrend[i] + lf*np.cos(psitrend[i])
    Yf = ytrend[i] + lf*np.sin(psitrend[i])
    Xr = xtrend[i] - lr*np.cos(psitrend[i])
    Yr = ytrend[i] - lr*np.sin(psitrend[i])

    thisxf = [xtrend[i], Xf]
    thisyf = [ytrend[i], Yf]
    thisxr = [xtrend[i], Xr]
    thisyr = [ytrend[i], Yr]

    linef.set_data(thisxf, thisyf)
    liner.set_data(thisxr, thisyr)
    time_text.set_text(time_template % (i*dT))
    return linef, liner, time_text

ani = animation.FuncAnimation(fig, animate, range(1, len(xtrend[0:-1])),
                              interval=dT*1000, blit=True, init_func=init)
rc('animation', html='jshtml')
ani
