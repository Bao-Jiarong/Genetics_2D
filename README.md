## Genetics in Python
The implemented genetics are:

* Evolutionary Programming (EP),
* Evolution Strategy (ES),
* Genetic Algorithm (GA).

All of the implemented algorithms can be used to find the minimum of 2D function.  
For example : f(x) = (x-2)^2.

### Requirement
```
python==3.7.0
numpy==1.18.1
```
### How to use

Open test.py you will find some examples
```
import src.ev as envo

def h(x):
    return (x-4)**2-4

ev = envo.EV(f = h,
             T = 100,
             low   = -1000,
             high  = 1000,
             p_size= 100,
             selection  = "roulette",\
             replacement= "elitist")

x = ev.evolutionary_programming()
print("x =",x,"h(x) =",h(x))
```
