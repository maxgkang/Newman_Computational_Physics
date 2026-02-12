import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
#%%

# Solving Ordinary Differential Equations

# 1. Euler's Method

# Define the function

def f(t, x):
    return -1 * x ** 3 + np.sin(t)

a = 0
b = 10
N = 1000
h = 1/100
x = 0
A = []
t = np.arange(a, b, h)

for j in range(1000):
    A.append(x)
    x += h * f(t[j], x)

plt.plot(t, A)
plt.show() 

# %%

# In practice, we never actually use Euler's Method. We use Runge Kutta Method
# 4차 Rutte Kutta Method(A physicist's main tool)

import numpy as np
import matplotlib.pyplot as plt


a = 0
b = 10
N = 1000
h = 1/100
x = 0
A = []
t = np.linspace(a, b, N)
k_1 = 0
k_2 = 0
k_3 = 0
k_4 = 0


def f(t, x):
    return - x ** 3 + np.sin(t)


for j in range(N):
    k_1 = h * f(t[j], x)
    k_2 = h * f(t[j] + 0.5 * h, x + 0.5 * k_1)
    k_3 = h * f( t[j] + 0.5 * h, x + 0.5 * k_2,)
    k_4 = h * f(t[j] + h, x + k_3, )
    x += 1/6 * (k_1 + 2 * k_2 + 2 * k_3 + k_4)
    A.append(x)

plt.plot(t, A)
plt.show()



# %%

# Differential Equations with more than one variable

# ex) dx/dt = xy -x , dx/dt = f_x(x,y,t) , dy/dt = f_y(x,y,t) , f(r,t) = (f_x(r,t), f_y(r,t)) , r= (x, y) 
import numpy as np
import matplotlib.pyplot as plt

def f(r,t):
    x = r[0]
    y = r[1]
    f_x = x * y - x
    f_y = y - x* y + np.sin(t) ** 2
    return np.array([f_x, f_y], float)
a = 0 
b = 10
N = 1000
h = (b-a)/N

tpoints = np.arange(a,b,h)
xpoints = []
ypoints = []

r = np.array([1, 0], float) # initial condition

for t in tpoints:
    xpoints.append(r[0])
    ypoints.append(r[1])
    k1 = h * f( r, t)
    k2 = h * f(r + 1/2 * k1 , t + 1/2 *h)
    k3 = h * f(r + 1/2 * k2, t + 1/2 * h)
    k4 = h * f(r + k3, t + h)
    r += (k1 + 2*k2 + 2*k3 +k4)/6

# take a dot, plot it up, r changes, 

plt.plot(tpoints, xpoints) #x가 안 보였던건 초깃값 때문!
plt.plot(tpoints, ypoints)
plt.show
#%% 
# 발산하는 함수 
from numpy import arange
from pylab import plot,xlabel,ylabel,xlim,show

def g(x,u):
    return 1/(x**2*(1-u)**2+u**2)

a = 0.0
b = 1.0
N = 100
h = (b-a)/N

upoints = arange(a,b,h)
tpoints = []
xpoints = []

x = 1.0
for u in upoints:
    tpoints.append(u/(1-u))
    xpoints.append(x)
    k1 = h*g(x,u)
    k2 = h*g(x+0.5*k1,u+0.5*h)
    k3 = h*g(x+0.5*k2,u+0.5*h)
    k4 = h*g(x+k3,u+h)
    x += (k1+2*k2+2*k3+k4)/6

plot(tpoints,xpoints)
xlim(0,80)
xlabel("t")
ylabel("x(t)")
show()
# %%

#  2nd order differential equations

import numpy as np
import matplotlib.pyplot as plt
g = 9.81 
l = 1.1
r = np.array([1, 1], float)
def f(r, t):
    theta = r[0]
    w = r[1]
    ftheta = w
    fw = -(g/l) * np.sin(theta)
    return np.array([ftheta,fw], float)

# f는 각속도와 각가속도를 배출한다. 
# w는 그냥 업데이트 된 r이다 
a = 0 
b = 10
N = 1000
h = (b-a)/N

tpoints = np.arange(a,b,h)
xpoints = []
ypoints = []

for t in tpoints:
    # 계산된 r과 해당 t로 각각 함수에 대입
    xpoints.append(r[0])
    ypoints.append(r[1])
    k1 = h * f( r, t)
    k2 = h * f(r + 1/2 * k1 , t + 1/2 *h)
    k3 = h * f(r + 1/2 * k2, t + 1/2 * h)
    k4 = h * f(r + k3, t + h) 
    r += (k1 + 2*k2 + 2*k3 +k4)/6

# f로 하나로 쓰이지만, 사실 f 내에 작동하는 함수는 두 개다.

plt.plot(tpoints, xpoints)
plt.plot(tpoints, ypoints)
plt.show
# %%

# Other methods for DE

# 1. The leapfrog method

# Why use? Because 4th RK method can have varying energ in the pendulum
# Teh leapfrog method does not give analytical results, but it is useful for
# time result symmetric. 

# The verlet method, the modified midpoint method, the Bulirsch-Stoer Method
# All of them are different ways to do DE
 
#%%

# The BS method, the king of solving DEs

from math import sin,pi
from numpy import empty,array,arange
from pylab import plot,show

g = 9.81
l = 0.1
theta0 = 179*pi/180

a = 0.0
b = 10.0
N = 100          # Number of "big steps"
H = (b-a)/N      # Size of "big steps"
delta = 1e-8     # Required position accuracy per unit time

def f(r):
    theta = r[0]
    omega = r[1]
    ftheta = omega
    fomega = -(g/l)*sin(theta)
    return array([ftheta,fomega],float)

tpoints = arange(a,b,H)
thetapoints = []
r = array([theta0,0.0],float)

# Do the "big steps" of size H
for t in tpoints:

    thetapoints.append(r[0])

    # Do one modified midpoint step to get things started
    n = 1
    r1 = r + 0.5*H*f(r)
    r2 = r + H*f(r1)

    # The array R1 stores the first row of the
    # extrapolation table, which contains only the single
    # modified midpoint estimate of the solution at the
    # end of the interval
    R1 = empty([1,2],float)
    R1[0] = 0.5*(r1 + r2 + 0.5*H*f(r2))

    # Now increase n until the required accuracy is reached
    error = 2*H*delta
    while error>H*delta:

        n += 1
        h = H/n

        # Modified midpoint method
        r1 = r + 0.5*h*f(r)
        r2 = r + h*f(r1)
        for i in range(n-1):
            r1 += h*f(r2)
            r2 += h*f(r1)

        # Calculate extrapolation estimates.  Arrays R1 and R2
        # hold the two most recent lines of the table
        R2 = R1
        R1 = empty([n,2],float)
        R1[0] = 0.5*(r1 + r2 + 0.5*h*f(r2))
        for m in range(1,n):
            epsilon = (R1[m-1]-R2[m-1])/((n/(n-1))**(2*m)-1)
            R1[m] = R1[m-1] + epsilon
        error = abs(epsilon[0])

    # Set r equal to the most accurate estimate we have,
    # before moving on to the next big step
    r = R1[n-1]

# Plot the results
plot(tpoints,thetapoints)
plot(tpoints,thetapoints,"b.")
show()

#%%

# The Shooting Method
# It starts with a correct solution to the differential equation that may not match the
# boundary conditions until it does


# Take a guess first, and keep modifying until you get the right one


from numpy import array,arange

g = 9.81         # Acceleration due to gravity
a = 0.0          # Initial time
b = 10.0         # Final time
N = 1000         # Number of Runge-Kutta steps
h = (b-a)/N      # Size of Runge-Kutta steps
target = 1e-10   # Target accuracy for binary search

# Function for Runge-Kutta calculation
def f(r):
    x = r[0]
    y = r[1]
    fx = y
    fy = -g
    return array([fx,fy],float)

# Function to solve the equation and calculate the final height
def height(v):
    r = array([0.0,v],float)
    for t in arange(a,b,h):
        k1 = h*f(r)
        k2 = h*f(r+0.5*k1)
        k3 = h*f(r+0.5*k2)
        k4 = h*f(r+k3)
        r += (k1+2*k2+2*k3+k4)/6
    return r[0]

# Main program performs a binary search
v1 = 0.01
v2 = 1000.0
h1 = height(v1)
h2 = height(v2)

while abs(h2-h1)>target:
    vp = (v1+v2)/2
    hp = height(vp)
    if h1*hp>0:
        v1 = vp
        h1 = hp
    else:
        v2 = vp
        h2 = hp

v = (v1+v2)/2
print("The required initial velocity is",v,"m/s")

#%%

# The Eigenvalue Problem

from numpy import array,arange

# Constants
m = 9.1094e-31     # Mass of electron
hbar = 1.0546e-34  # Planck's constant over 2*pi
e = 1.6022e-19     # Electron charge
L = 5.2918e-11     # Bohr radius
N = 1000
h = L/N

# Potential function
def V(x):
    return 0.0

def f(r,x,E):
    psi = r[0]
    phi = r[1]
    fpsi = phi
    fphi = (2*m/hbar**2)*(V(x)-E)*psi
    return array([fpsi,fphi],float)

# Calculate the wavefunction for a particular energy
def solve(E):
    psi = 0.0
    phi = 1.0
    r = array([psi,phi],float)

    for x in arange(0,L,h):
        k1 = h*f(r,x,E)
        k2 = h*f(r+0.5*k1,x+0.5*h,E)
        k3 = h*f(r+0.5*k2,x+0.5*h,E)
        k4 = h*f(r+k3,x+h,E)
        r += (k1+2*k2+2*k3+k4)/6

    return r[0]

# Main program to find the energy using the secant method
E1 = 0.0
E2 = e
psi2 = solve(E1)

target = e/1000
while abs(E1-E2)>target:
    psi1,psi2 = psi2,solve(E2)
    E1,E2 = E2,E2-psi2*(E2-E1)/(psi2-psi1)

print("E =",E2/e,"eV")