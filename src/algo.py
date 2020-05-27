'''
 *  Author : Bao Jiarong
 *  Contact: bao.salirong@gmail.com
 *
 *  Created  On: 2020-05-27
 *  Modified On: 2020-05-27
 '''
import random
import numpy as np

def uniform(low,high):
    return int(low+((high-low)*random.random()))

def randfloat(low,high):
    return low+((high-low)*random.random())

def rand_floats(low,high,n):
    a =[]
    for i in range(n):
        t = low+((high-low)*random.random())
        a.append(round(t,5))
    return a


class Algo:
    #----------------------------------------------------------
    # Constructor
    #----------------------------------------------------------
    def __init__(self,T,f,eps=1e-3,verbose=False):
            self.T   = T
            self.f   = f
            self.err = []
            self.eps  = eps
            self.verbose = verbose

    #----------------------------------------------------------
    def set_verbose(self,status):
        self.verbose = status

    def plot(self):
        import matplotlib.pyplot as plt
        fig= plt.figure()
        ax = fig.add_axes([0.1,0.1,0.85,0.85])
        ax.grid(color='b', ls = '-.', lw = 0.25)
        ax.set_xlabel("iteration")
        ax.set_ylabel("Progress")
        x = np.arange(0, len(self.err))
        ax.plot(x, self.err)
        plt.show()

    def display(self,a,b):
        if self.verbose == True:
            if a % 10 == 0:
                print("iteration =",a,
                      "\tx =",round(b,5),
                      "\tf(x) =",round(self.f(b),5))
