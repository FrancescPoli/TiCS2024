# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 11:26:37 2023

@author: U661121
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az
import seaborn as sns
from scipy.stats import norm, zscore
from sklearn.preprocessing import normalize


# Toy example on how to track prediction error, learning progress, uncertainty and information gain.
# every animal as N distonctive features. Think about this as the exact length, color of the tail, teeth, so on.
N = 1000
# We start by defining a prior. You expect to see a cat (id number = 20)
y = np.repeat(30,N)
# we are not fully sure it will be a cat. Let's introduce some uncertainty
var = 0.5
y = np.random.normal(y, var)
# this is our prior:
az.plot_kde(y)

# now we spot an animal in the bushes. It is a monkey (id number = 30) but we do not yet.
r = np.repeat(34,N) # r stays for reality. Let's define it, but we haven't observed yet...
# in one condition, the animal is going further back into the bush, while in the other, it is coming out

# let's start with the animal coming out
# we have to define how uncertainty is decreasing over time (i.e., the bushes getting less think)
time = np.arange(0,3000,1) # time in seconds
var_t_mean = np.empty(np.shape(time))
var_start = 24
var_change = -0.0001
var_t_mean[0:1000] = var_start + var_change*time[0:1000]


var_change = -0.02
var_t_mean[1000:2000] = var_start+ var_change*time[0:1000]


var_change = -0.0001
var_t_mean[2000:3000] = var_t_mean[2000-1] + var_change*time[0:1000]



plt.plot(time, var_t_mean)
# again, let's introduce some noise
var_t = np.random.normal(var_t_mean,0.2)
plt.plot(time, var_t)

# now let's make the monkey get out of those bushes!
r = np.tile(r, (len(time),1)) # now we have a monkey for every time point
# now we introduce the noise from the bushes
for t in time:
    r[t,:] = np.random.normal(r[t,:],var_t[t])
    #let's see how the evidence changes across time
    if t in np.arange(0,len(time),500):
        sns.kdeplot(data=r[t,:])


# now let's look at that monkey. We're like a reinforcement learning algorithm
# updating our beliefs as we keep watching at the animal moving in the bushes

# first off, let's define how quickly we update our prior (i.e., the learning rate)
alpha = 0.001

#second, the update rule:
def update(y_current, y_real, alpha):
    y_next = y_current + alpha * (y_real-y_current)
    return(y_next)

# let's apply the learning rule to our data
beliefs = np.empty(np.shape(r))
y = np.random.normal(y, var)
#for every time point
for t in time:
    # for every feature
    for i in range(len(y)):
        # let's update each feature
        y[i] = update(y[i],r[t,i],alpha)
        # yes, learning is this easy. Let's just save what we just learned
        beliefs[t,:] = y

plt.figure(figsize=(4, 2.5))
m = 14
colors = sns.color_palette("BuGn_r", 20)
for n in np.arange(0,len(time),200):
    sns.kdeplot(data=beliefs[n,:], color = colors[m])
    m-=1
sns.despine(bottom = False, left = True)
plt.tick_params(labelleft=False, left=False)
plt.xlim((25,40))
plt.ylim((0,2.5))
plt.savefig('precision_come.png', dpi=300)

# now that we've realized we're actually looking at a monkey, let's get to the juicy part:
# measuring our curiosity across time...

# prediction error is the difference between prediction and reality
pe = np.mean(np.abs(r-beliefs), axis = 1)
plt.plot(time, pe)

# uncertainty is the variance in our beliefs
u = np.std(beliefs, axis = 1)
plt.plot(time, u)

# lerning progress is how prediction error changes over time
# we can track it in windows of 50ms
chunk = 40
lp = [np.nan]
pe2 = [np.mean(pe[0:chunk])]
u2 = [np.mean(u[0:chunk])]
t_prev = 0
for t in np.arange(chunk,len(time),chunk):
    pe_prev_t = np.mean(pe[t_prev:t])
    
    pe_t = np.mean(pe[t:t+chunk])
    
    lp.append(pe_prev_t-pe_t)
    pe2.append(pe_t)
    u2.append(np.mean(u[t:t+chunk]))
    t_prev = t

red_time = range(len(lp))



# finally... information gain. 
mean_b = np.mean(beliefs, axis = 1)


kl = [np.nan]
for t in np.arange(chunk,len(time),chunk):
    b_prev_t = np.mean(mean_b[t_prev:t],)
    
    b_t = np.mean(mean_b[t:t+chunk])
    
    u_prev_t = np.mean(u[t_prev:t])
    
    u_t = np.mean(u[t:t+chunk])
    
    kl_t = np.log(u_t/u_prev_t) + (u_prev_t**2 + (b_prev_t - b_t)**2)/(2*u_t**2) - 0.5
    #kl_t = np.log(u_prev_t/u_t) + (u_t**2 + (b_t - b_prev_t)**2)/(2*u_prev_t**2) - 0.5
    
    t_prev = t
    
    kl.append(kl_t)
    

plt.plot(range(len(kl)),kl)


pe3 = pe[np.arange(0,len(time),chunk)]
u3 = u[np.arange(0,len(time),chunk)]
noise3 = var_t[np.arange(0,len(time),chunk)]

lp_norm = (lp-np.nanmin(lp))/(np.nanmax(lp)-np.nanmin(lp))
kl_norm = (kl-np.nanmin(kl))/(np.nanmax(kl)-np.nanmin(kl))
pe_norm = (pe3-np.nanmin(pe3))/(np.nanmax(pe3)-np.nanmin(pe3))
u_norm = (u3-np.nanmin(u3))/(np.nanmax(u3)-np.nanmin(u3))


fig, ax = plt.subplots(figsize=(4.4, 3.3))
#ax.set_title("Motives of Curiosity")

sns.lineplot(x = red_time, y = kl_norm, color = 'red', label='Information Gain') # plots the first set
sns.lineplot(x = red_time, y = pe_norm, color = 'orange', label='Prediction Error') # plots the second set 
sns.lineplot(x = red_time, y = u_norm, color = 'green', label='Uncertainty') # plots the third set 
sns.lineplot(x = red_time, y = lp_norm-0.18, color = 'steelblue', label='Learning Progress') # plots the third set 

sns.despine(bottom = False, left = False)

ax.set(yticklabels=[])  
ax.set(ylim=(-1.2, 2))

ax.set(ylabel='Curiosity')
ax.set(xlabel='Time (s)')

sns.move_legend(ax, "upper center", frameon =False, bbox_to_anchor=(0.75, 1.0), title=None)

plt.tight_layout()
plt.savefig('curiosity_come.png', dpi=300)



sns.heatmap([var_t]*2, cmap='Greens', shading='gouraud', vmin=0, vmax=45)
plt.ylim((0,0.05))
plt.savefig('noise_come.png', dpi=300)



fig, ax = plt.subplots(figsize=(4.4, 3))
#ax.set_title("Motives of Curiosity")

sns.lineplot(x = red_time, y = lp_norm-.18+kl_norm, color = 'black', label='Learning Progress + Information Gain') # plots the third set 

sns.despine(bottom = False, left = False)

ax.set(yticklabels=[])  

ax.set(ylabel='Curiosity')
ax.set(xlabel='Time (s)')

sns.move_legend(ax, "upper center", frameon =False)#bbox_to_anchor=(1.2, 1), title=None, frameon=False)
ax.set(ylim=(-.75, 1.5))

plt.tight_layout()
plt.savefig('curiosity_int_approach.png', dpi=300)



# and now... let's make the monkey get further into the bushes.


# Toy example on how to track prediction error, learning progress, uncertainty and information gain.
# every animal as N distonctive features. Think about this as the exact length, color of the tail, teeth, so on.
N = 1000
# We start by defining a prior. You expect to see a cat (id number = 20)
y = np.repeat(30,N)
# we are not fully sure it will be a cat. Let's introduce some uncertainty
var = 0.5
y = np.random.normal(y, var)
# this is our prior:
az.plot_kde(y)

# now we spot an animal in the bushes. It is a monkey (id number = 30) but we do not yet.
r = np.repeat(34,N) # r stays for reality. Let's define it, but we haven't observed yet...
# in one condition, the animal is going further back into the bush, while in the other, it is coming out

# let's start with the animal coming out
# we have to define how uncertainty is decreasing over time (i.e., the bushes getting less think)
time = np.arange(0,3000,1) # time in seconds
var_t_mean = np.empty(np.shape(time))
var_start = 24
var_change = -0.0001
var_t_mean[0:1000] = var_start + var_change*time[0:1000]


var_change = 0.02
var_t_mean[1000:2000] = var_start+ var_change*time[0:1000]


var_change = -0.0001
var_t_mean[2000:3000] = var_t_mean[2000-1] + var_change*time[0:1000]



plt.plot(time, var_t_mean)
# again, let's introduce some noise
var_t = np.random.normal(var_t_mean,0.2)
plt.plot(time, var_t)

# now let's make the monkey get out of those bushes!
r = np.tile(r, (len(time),1)) # now we have a monkey for every time point
# now we introduce the noise from the bushes
for t in time:
    r[t,:] = np.random.normal(r[t,:],var_t[t])
    #let's see how the evidence changes across time
    if t in np.arange(0,len(time),500):
        sns.kdeplot(data=r[t,:])


# now let's look at that monkey. We're like a reinforcement learning algorithm
# updating our beliefs as we keep watching at the animal moving in the bushes

# first off, let's define how quickly we update our prior (i.e., the learning rate)
alpha = 0.001

#second, the update rule:
def update(y_current, y_real, alpha):
    y_next = y_current + alpha * (y_real-y_current)
    return(y_next)

# let's apply the learning rule to our data
beliefs = np.empty(np.shape(r))
y = np.random.normal(y, var)
#for every time point
for t in time:
    # for every feature
    for i in range(len(y)):
        # let's update each feature
        y[i] = update(y[i],r[t,i],alpha)
        # yes, learning is this easy. Let's just save what we just learned
        beliefs[t,:] = y

m = 14
plt.figure(figsize=(4, 2.5))
colors = sns.color_palette("BuGn_r", 20)
for n in np.arange(0,len(time),200):
    sns.kdeplot(data=beliefs[n,:], color = colors[m])
    m-=1
sns.despine(bottom = False, left = True)
plt.tick_params(labelleft=False, left=False)
plt.xlim((25,40))
plt.ylim((0,2.5))
plt.savefig('precision_go.png', dpi=300)


# now that we've realized we're actually looking at a monkey, let's get to the juicy part:
# measuring our curiosity across time...

# prediction error is the difference between prediction and reality
pe = np.mean(np.abs(r-beliefs), axis = 1)
plt.plot(time, pe)

# uncertainty is the variance in our beliefs
u = np.std(beliefs, axis = 1)
plt.plot(time, u)

# lerning progress is how prediction error changes over time
# we can track it in windows of 50ms
chunk = 40
lp = [np.nan]
pe2 = [np.mean(pe[0:chunk])]
u2 = [np.mean(u[0:chunk])]
t_prev = 0
for t in np.arange(chunk,len(time),chunk):
    pe_prev_t = np.mean(pe[t_prev:t])
    
    pe_t = np.mean(pe[t:t+chunk])
    
    lp.append(pe_prev_t-pe_t)
    pe2.append(pe_t)
    u2.append(np.mean(u[t:t+chunk]))
    t_prev = t

red_time = range(len(lp))



# finally... information gain. 
mean_b = np.mean(beliefs, axis = 1)


kl = [np.nan]
for t in np.arange(chunk,len(time),chunk):
    b_prev_t = np.mean(mean_b[t_prev:t],)
    
    b_t = np.mean(mean_b[t:t+chunk])
    
    u_prev_t = np.mean(u[t_prev:t])
    
    u_t = np.mean(u[t:t+chunk])
    
    kl_t = np.log(u_t/u_prev_t) + (u_prev_t**2 + (b_prev_t - b_t)**2)/(2*u_t**2) - 0.5
    #kl_t = np.log(u_prev_t/u_t) + (u_t**2 + (b_t - b_prev_t)**2)/(2*u_prev_t**2) - 0.5
    
    t_prev = t
    
    kl.append(kl_t)


#lp = np.abs(lp)

pe3 = pe[np.arange(0,len(time),chunk)]
u3 = u[np.arange(0,len(time),chunk)]

lp_norm = (lp-np.nanmin(lp))/(np.nanmax(lp)-np.nanmin(lp))
kl_norm = (kl-np.nanmin(kl))/(np.nanmax(kl)-np.nanmin(kl))
pe_norm = (pe3-np.nanmin(pe3))/(np.nanmax(pe3)-np.nanmin(pe3))
u_norm = (u3-np.nanmin(u3))/(np.nanmax(u3)-np.nanmin(u3))


fig, ax = plt.subplots(figsize=(4.4, 3.3))
#ax.set_title("Motives of Curiosity")

sns.lineplot(x = red_time, y = kl_norm, color = 'red', label='Information Gain') # plots the first set
sns.lineplot(x = red_time, y = pe_norm+0.9, color = 'orange', label='Prediction Error') # plots the second set 
sns.lineplot(x = red_time, y = u_norm+0.65, color = 'green', label='Uncertainty') # plots the third set 
sns.lineplot(x = red_time, y = lp_norm-.7, color = 'steelblue', label='Learning Progress') # plots the third set 

sns.despine(bottom = False, left = False)

ax.set(yticklabels=[])  

ax.set(ylabel='Curiosity')
ax.set(xlabel='Time (s)')

ax.get_legend().remove()
ax.set(ylim=(-1.2, 2))

plt.tight_layout()
plt.savefig('curiosity_away2.png', dpi=300)



fig, ax = plt.subplots(figsize=(4.4, 3))
#ax.set_title("Motives of Curiosity")

sns.lineplot(x = red_time, y = lp_norm-.7+kl_norm, color = 'black', label='Learning Progress + Information Gain') # plots the third set 

sns.despine(bottom = False, left = False)

ax.set(yticklabels=[])  

ax.set(ylabel='Curiosity')
ax.set(xlabel='Time (s)')

ax.get_legend().remove()
ax.set(ylim=(-.75, 1.5))

plt.tight_layout()
plt.savefig('curiosity_int_away.png', dpi=300)




sns.heatmap([var_t]*2, cmap='Greens', shading='gouraud', vmin=0, vmax=45)
plt.ylim((0,0.05))
plt.savefig('noise_go.png', dpi=300)



#kl_norm 
#pe_norm+0.9
#u_norm+0.65
#lp_norm-.7


import scipy.special as special
score = 0
for n in range(1000):
    # decide when to stop sampling
    keep_lp = 1
    for t in range(len(lp_norm)):
        if lp_norm[t]>=0:
            if np.random.binomial(1, lp_norm[t]) == 0:
                score += (1/(u[t]**2))/t
score_lp = score/1000

score = 0
for n in range(1000):
    # decide when to stop sampling
    keep_lp = 1
    for t in range(len(lp_norm)):
        if kl_norm[t]>=0:
            if np.random.binomial(1, kl_norm[t]) == 0:
                score += 1/(u[t]**2)/t     
score_kl = score/1000

score = 0
for n in range(1000):
    # decide when to stop sampling
    keep_lp = 1
    for t in range(1,len(pe_norm)):
        if pe_norm[t]>=0:
            if np.random.binomial(1, .9) == 0:
                score += 1/(u[t]**2)/t     
score_pe = score/1000

score = 0
for n in range(1000):
    # decide when to stop sampling
    keep_lp = 1
    for t in range(1,len(pe_norm)):
        if kl_norm[t]>=0:
            if np.random.binomial(1, (kl_norm[t]*.7 + lp_norm[t]*.3)) == 0:
                score += 1/(u[t]**2)/t     
score_opt = score/1000






















fig, ax1 = plt.subplots(figsize=(5, 4))
ax1.set_title("Motives of Curiosity")
ax2 = ax1.twinx()
ax3 = ax1.twinx()
ax4 = ax1.twinx()

g1 = sns.lineplot(x = red_time, y = kl, ax = ax1, color = 'red', label='Information Gain') # plots the first set
g2 = sns.lineplot(x = red_time, y = pe3, ax = ax2, color = 'orange', label='Prediction Error') # plots the second set 
g3 = sns.lineplot(x = red_time, y = u3, ax = ax3, color = 'green', label='Uncertainty') # plots the third set 
g4 = sns.lineplot(x = red_time, y = lp, ax = ax4, color = 'steelblue', label='Learning Progress') # plots the third set 

sns.despine(bottom = False, left = False)

g1.set(yticklabels=[])  
g2.set(yticklabels=[])
g3.set(yticklabels=[])
g4.set(yticklabels=[])

g1.set(ylabel='Curiosity')
g1.set(xlabel='Time (s)')

handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right', ncol=3, bbox_to_anchor=(.75, 0.98))


plt.savefig('curiosity.png', dpi=300)



#lp_norm = (lp[2:]-np.min(lp[2:]))/(np.max(lp[2:])-np.min(lp[2:]))
#kl_norm = (kl[2:]-np.min(kl[2:]))/(np.max(kl[2:])-np.min(kl[2:]))

#c = (10*lp_norm)*kl_norm
#plt.plot(range(len(c)),np.abs(c))


