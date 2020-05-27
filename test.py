'''
 *  Author : Bao Jiarong
 *  Contact: bao.salirong@gmail.com
 *  Created  On: 2020-05-27
 *  Modified On: 2020-05-27
 '''
import src.ev as envo

#---------------------------na-----------------------------
def h(x):
    return (x-4)**2-4

ev = envo.EV(f     = h,               # the function to be minimized
             T     = 100,             # maximum number of iterations
             low   = -1000,           # search domain
             high  = 1000,            # search domain
             p_size= 100,             # population size
             selection  = "roulette", # others: tournament
             replacement= "elitist")  # others: uniform

#----------------------------ev--------------------------------
# # ev.set_verbose(True)
#--------------------------------
print("Evolutionary Programming")
x = ev.evolutionary_programming()
print("x =",x,"h(x) =",h(x))
#--------------------------------
print("Evolution Strategy")
x = ev.evolution_strategy(lamda= 40,
                          u    = 15)
print("x =",x,"h(x) =",h(x))
#--------------------------------
print("Genetic Algorithm")
x = ev.genetic_algorithm()
print("x =",x,"h(x) =",h(x))

# ev.plot()
