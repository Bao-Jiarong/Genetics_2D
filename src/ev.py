'''
 *  Author : Bao Jiarong
 *  Contact: bao.salirong@gmail.com
 *
 *  Created  On: 2020-05-27
 *  Modified On: 2020-05-27
 '''
import random
import numpy as np
from .algo import *

class EV(Algo):
    #--------------------------------------------------------
    # Constructor
    #---------------------------------------------------------
    def __init__(self,f,T,low,high,p_size,eps=1e-3,selection="tournament",\
                 replacement="uniform",verbose=False):
        Algo.__init__(self,T,f,eps,verbose)
        self.low    =  low
        self.high   =  high
        self.p_size =  p_size
        self.selection_name   = selection
        self.replacement_name = replacement

    #------------------Genetic Operators---------------------
    def creation(self):
        return randfloat(self.low,self.high)

    def crossover(self,a,b,alpha = 0.25):
        return alpha*a+(1-alpha)*b

    def mutation(self,a):
        r = random.random() + self.eps
        return a + r * a

    def reproduction(self,p):
        n = len(p)
        p1 = []

        for i in range(int(n/4)):
            t = self.creation()
            p1.append(t)

        for i in range(int(n/4)):
            n = len(p)
            t1 = uniform(0,n)
            t2 = uniform(0,n)
            a = p[t1]
            b = p[t2]
            t = self.crossover(a,b)
            p1.append(t)

        for i in range(int(n/4)):
            a = p[int(randfloat(0,n))]
            t = self.mutation(a)
            p1.append(t)
        return p1

    #-------------------Replacement---------------------------
    def uniform_replacement(self,p,p1):
        N = len(p)
        n1 = len(p)
        n2 = len(p1)
        t = []
        for i in range(N):
            a = uniform(0,n1)
            b = uniform(0,n2)
            x = p[a]
            y = p1[b]
            if self.f(x)>self.f(y):
                best = y
            else:
                best = x
            t.append(best)
        return t

    def elitist_replacement(self,p,p1):
        A = p + p1
        n = len(p)
        m = len(A)
        t = 0
        A = [round(a,5) for a in A]
        B = [self.f(a) for a in A]

        for i in range(m):
            k = B[i]
            for j in range(i,m,1):
                if k>B[j]:
                    k=B[j]
                    t = j
            x = B[i]
            B[i] = B[t]
            B[t] = x

            x = A[i]
            A[i] = A[t]
            A[t] = x
        return A[:n]

    def replacement(self,p,p1):
        if self.replacement_name == "uniform":
            return self.uniform_replacement(p,p1)
        elif self.replacement_name == "elitist":
            return self.elitist_replacement(p,p1)

    #-------------------Selection-----------------------------
    def tourname_selection(self,p,s,k):
        R = []
        n = len(p)
        for i in range(s):
            a = uniform(0,n)
            for j in range(k):
                b = uniform(0,n)
                x = p[a]
                y = p[b]
                if self.f(x)>self.f(y):
                    a = b
            R.append(p[a])
        return R

    def roulette_wheel(self,p,s):
        m = len(p)
        n = 0
        A = []
        B = []
        for i in range(m):
            n = n + self.f(p[i])
        for i in range(m):
            A.append(self.f(p[i])/(n+self.eps))
        for i in range(m):
            b = 0
            for j in range(i):
                b = b + A[j]
            B.append(b)
        p1 = []
        for n in range(s):
            r = uniform(0,1)
            for i in range(m):
                if r <= B[i]:
                    p1.append(p[i])
        return p1

    def selection(self,p,s,k=25):
        if self.selection_name == "tournament":
            return self.tourname_selection(p,s,k)
        elif self.selection_name == "roulette":
            return self.roulette_wheel(p,s)

    #-----------------Evolutionary Algorithms------------------

    #----------------------------------------------------------
    # Evolutionary Programming
    # Info (En): https://en.wikipedia.org/wiki/Evolutionary_programming
    # Info (Ch):
    #----------------------------------------------------------
    def evolutionary_programming(self):
        self.err  = []
        p = rand_floats(self.low,self.high,self.p_size)
        n = len(p)

        for t in range (self.T):
            p1 = []
            for i in range(n):
                p1.append(self.mutation(p[i]))
            # p = self.elitist_replacement(p,p1)
            # p = self.uniform_replacement(p,p1)
            p = self.replacement(p,p1)

            # Loss
            self.err.append(self.f(p[0]))

            # Progress
            self.display(t,p[0])

            # Early stopping
            if abs(self.f(p[0])) < self.eps:
                break

        # p2 = (self.f(x) for x in p)
        # p2 = np.array(p2)
        # index = np.argmin(p2)
        return p[0]

    #----------------------------------------------------------
    # Evolution Strategy
    # Info (En): https://en.wikipedia.org/wiki/Evolution_strategy
    # Info (Ch):
    #----------------------------------------------------------
    def evolution_strategy(self,u=15,k=25,lamda=40):
        self.err = []
        p = rand_floats(self.low,self.high,self.p_size)

        for t in range (self.T):
            p1 = []
            p2 = []
            C = self.selection(p,u,k)
            n = len(C)

            for i in range(lamda):
                a = C[uniform(0,n)]
                b = C[uniform(0,n)]
                p1.append(self.crossover(a,b))

            k = len(p1)
            for i in range(k):
                p2.append(self.mutation(p1[i]))

            p = self.replacement(p,p2)

            # Loss
            self.err.append(self.f(p[0]))

            # Progress
            self.display(t,p[0])

            # Early stopping
            if abs(self.f(p[0])) < self.eps:
                break

        return p[0]

    #----------------------------------------------------------
    # Genetic Algorithm
    # Info (En): https://en.wikipedia.org/wiki/Genetic_algorithm
    # Info (Ch): https://zh.wikipedia.org/wiki/遗传算法
    #----------------------------------------------------------
    def genetic_algorithm(self,u=15,k=25):
        self.err = []
        p = rand_floats(self.low,self.high,self.p_size)
        n = len(p)

        for t in range (self.T):
            p1 = []
            p2 = []
            p3 = np.array([])

            p1 = self.selection(p,u,k)
            p2 = self.reproduction(p1)
            p = self.replacement(p,p2)

            # Loss
            self.err.append(self.f(p[0]))

            # Progress
            self.display(t,p[0])

            # Early stopping
            if abs(self.f(p[0])) < self.eps:
                break

        return p[0]
