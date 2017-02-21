# -*- coding: utf-8 -*-
"""
Created on Sun Jan  1 16:09:08 2017
#Particle swarm optimization
@author: Sunrise
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, cm

class PSO:
    
    def __init__(self, lossfun, part_num=50):
        self.lossfun = lossfun
        self.part_num = part_num
    
    def part_init(self, lower, upper):
        self.lower = lower
        self.upper = upper
        self.df = lower.shape[0]
        
        self.current_pos = np.zeros([self.part_num, self.df])
        self.current_vel = np.zeros([self.part_num, self.df])
        self.current_loss = np.zeros(self.part_num)
        for i in range(self.df):
            self.current_pos[:,i] = np.random.uniform(self.lower[i], 
                self.upper[i], self.part_num)
            vel_range = self.upper[i] - self.lower[i]
            self.current_vel[:,i] = np.random.uniform(-vel_range, vel_range, 
                self.part_num)
        for p in range(self.part_num):
            self.current_loss[p] = self.lossfun(self.current_pos[p,:])
        
        self.min_pos = np.zeros([self.part_num, self.df]) 
        self.min_loss = np.zeros(self.part_num)
        self.min_pos[:,:] = self.current_pos[:,:]
        self.min_loss[:] = self.current_loss[:]
        
        self.global_pos = np.zeros(self.df) 
        self.global_pos[:] = self.current_pos[np.argmin(self.current_loss),:]
        self.global_loss = min(self.current_loss)                                                                    
        
    def fit(self, coe, iter_num=500):
        self.iter_num = iter_num
        self.coe = coe
        
        def update_current(p, i):
            temp_vel = self.current_vel[p,i]*coe[0] + \
                r_part*(self.min_pos[p,i]-self.current_pos[p,i])*coe[1] + \
                        r_global*(self.global_pos[i]-
                                  self.current_pos[p,i])*coe[2]
            if temp_vel + self.current_pos[p,i] < self.lower[i]:
                self.current_vel[p,i] = self.lower[i] - self.current_pos[p,i] 
                self.current_pos[p,i] = self.lower[i]
            elif temp_vel + self.current_pos[p,i] > self.upper[i]:
                self.current_vel[p,i] = self.upper[i] - self.current_pos[p,i]
                self.current_pos[p,i] = self.upper[i]   
            else:
                self.current_vel[p,i] = temp_vel
                self.current_pos[p,i] = temp_vel + self.current_pos[p,i]
#        print(self.current_pos)
        for n in range(iter_num):
#            if n%100 == 1:
#                print(self.current_pos)
            for p in range(self.part_num):
                r_part, r_global = np.random.uniform(0,1,2)
                for i in range(self.df):
                    update_current(p,i)
#                    print(p,i)
#                    print(self.current_pos)
                self.current_loss[p] = self.lossfun(self.current_pos[p,:])
#                print(p)
#                print(self.current_pos)
#                print(self.current_loss)
                if self.min_loss[p] > self.current_loss[p]:
                    self.min_loss[p] = self.current_loss[p]
                    self.min_pos[p] = self.current_pos[p,:]
#                    print(self.min_loss[p]==self.lossfun(self.current_pos[p,:]))
                    if self.global_loss > self.current_loss[p]:
#                        print('******')
                        self.global_loss = self.current_loss[p]
                        self.global_pos = self.current_pos[p,:]

                        
#def loss(x):
#    r = np.sqrt(sum(x**2)) + np.random.uniform(0,0.5)
#    return -(np.sin(r)/r)
#  
##plot
#N = 100
#x = np.linspace(l[0], u[0], N)
#y = np.linspace(l[1], u[1], N)
#X, Y = np.meshgrid(x, y)
#Z = np.zeros([N,N])
#for i in range(N):
#    for j in range(N):
#        Z[i,j] = loss(np.append(X[i,j],Y[i,j]))
#levels = np.linspace(np.amin(Z),np.amax(Z), 1000)
##plt.figure(figsize=(10,10))
#plt.contourf(X, Y, Z, levels, cmap=cm.rainbow)
#plt.colorbar()
#    
##Optimization    
#l = np.array([-20, -20])
#u = np.array([20, 20])
#coe = np.array([0.1,0.7,0.5])
#pso = PSO(loss,50)
#pso.part_init(l,u)
#plt.plot(pso.current_pos[:,0],pso.current_pos[:,1],'ko')
##print(pso.current_pos)
##print(pso.current_loss)
##print(pso.global_pos)
##print(pso.global_loss)
#pso.fit(coe,500)
##print(pso.current_pos)
#plt.plot(pso.current_pos[:,0],pso.current_pos[:,1],'ro')




