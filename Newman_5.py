# Partial Differential Equations

# Basic classifications

# Poisson Equation: laplacian U = constant
# Heat Equation: laplacian U = constant x \frac{\partial U}{\partial t}
# Wave Equation: laplacian U = constant x \frac{\partial U ^ 2}{\partial ^ 2 t}

# Boundary conditions
# Drichlet Condition: At boundaries, there exists values of the dependent variables
# Neumann Condition: The normal derivative of the dependent variable is specified on the boundary
# Cauchy Condition: Both are specified

# Existence of a solution
# Heat Equation: Drichlet Open
# Poisson Equation: Drichlet Closed
# Wave Equation: Caucchy Open

#%%

# Finite-difference method

# df/dx ~ f(x+h) - f(x) / h , min error of O(h)
# df/dx ~ f(x) - f(x-h) / h , min error of O(h)
# df/dx ~ f(x+h) - f(x-h) / h , min error of O(h^2), due to symmetry

# Error Analysis

# 2 Dimensional Laplace equation

# Jacobi Method, 1x1 m^2 box, grid spacing a = 1cm , always numerically stable
# When calculation diverges, it is numerically unstable

import numpy as np
import matplotlib.pyplot as plt

# Constants
M = 100         # Grid squares on a side
V = 1.0         # Voltage at top wall
epsilon = 1e-6   # error

phi = np.zeros([M+1,M+1],float) # Sheet A (배경)
phi[0,:] = V 
phiprime = np.empty([M+1,M+1],float) # Sheet B (업데이트 할 것) 

# Main loop
delta = 1.0
while delta > epsilon : # Calculate new values of the potential
    for i in range(M+1):
        for j in range(M+1):
            if i==0 or i==M or j==0 or j==M:
                phiprime[i,j] = phi[i,j]
            else:
                phiprime[i,j] = (phi[i+1,j] + phi[i-1,j] + phi[i,j+1] + phi[i,j-1])/4
    delta = np.max(abs(phi-phiprime)) # phi and phiprime array updated -> update delta
    
    # Swap the two arrays around
    phi,phiprime = phiprime,phi

# Matplotlib converts all inputs to NumPy arrays internally before plotting

plt.imshow(phi, cmap = 'bwr') # turns your 2D array of numbers into a visual "heatmap" or image.
plt.show()

# 색깔
#  Intensity of Blue & Red: seismic > bwr > coolworm
#  plt.gray() is also usable 
 
# %%
# start_time = time.time() # 2. Mark the start time
# your code
# end_time = time.time() # 3. Mark the end time

#%%

# Gauss - Seidel Method


import numpy as np
import matplotlib.pyplot as plt

# 1. Settings
M = 60                # Grid size (60x60). Kept small for faster execution in pure Python.
V = 100.0             # Voltage at the boundary
target = 1e-4         # Target accuracy (convergence criteria)

# 2. Initialize Grid
# phi is a 2D array of floats, initially all 0
phi = np.zeros([M+1, M+1], float)

# 3. Set Boundary Conditions
# User requested: phi(x=0, y) = V.
# In a matrix phi[y, x], x=0 is the left column (index :, 0).
phi[:, 0] = V         # Left Boundary = V
phi[:, -1] = 0        # Right Boundary = 0
phi[0, :] = 0         # Bottom Boundary = 0
phi[-1, :] = 0        # Top Boundary = 0

# 4. Main Loop (Gauss-Seidel)
delta = 1.0
iterations = 0

print("Starting calculation...")

while delta > target:
    old_phi = phi.copy()  # Save copy to calculate error later
    
    # Iterate over interior points
    # Note: We do NOT create a new 'phiprime' array. We update 'phi' directly.
    for i in range(1, M):          # rows (y)
        for j in range(1, M):      # columns (x)
            # Update using the average of 4 neighbors
            # Since we modify 'phi' in place, phi[i-1, j] and phi[i, j-1] 
            # are ALREADY updated for this iteration!
            phi[i, j] = 0.25 * (phi[i+1, j] + phi[i-1, j] + phi[i, j+1] + phi[i, j-1])

    # Calculate maximum change (error)
    delta = np.max(np.abs(phi - old_phi))
    
    iterations += 1
    if iterations % 100 == 0:
        print(f"Iteration {iterations}, Delta: {delta:.5f}")

print(f"Converged in {iterations} iterations.")

# 5. Plotting
plt.figure(figsize=(8, 6))
plt.imshow(phi, cmap='inferno', origin='lower')
plt.colorbar(label='Potential (V)')
plt.title(f'2D Laplace Solution (Gauss-Seidel)\nLeft Edge = {V}V')
plt.xlabel('x (Grid Points)')
plt.ylabel('y (Grid Points)')
plt.show()

# %%

# Forward-Time Centered Space[FTCS] Method
# Good for initial values varying in time
# Von Neumann Stability Analysis

import numpy as np
from pylab import plot,xlabel,ylabel,show

# Constants
L = 0.01      # Thickness of steel in meters
D = 4.25e-6   # Thermal diffusivity
N = 100       # Number of divisions in grid
a = L/N       # Grid spacing
h = 1e-4      # Time-step
epsilon = h/1000

Tlo = 0.0     # Low temperature in Celcius
Tmid = 20.0   # Intermediate temperature in Celcius
Thi = 50.0    # Hi temperature in Celcius

t1 = 0.01
t2 = 0.1
t3 = 0.4
t4 = 1.0
t5 = 10.0
tend = t5 + epsilon

# Create arrays
T = np.empty(N+1,float)
T[0] = Thi
T[N] = Tlo
T[1:N] = Tmid
Tp = np.empty(N+1,float)
Tp[0] = Thi
Tp[N] = Tlo

# Main loop
t = 0.0
c = h*D/(a*a)
while t<tend:

    # Calculate the new values of T
    for i in range(1,N):
        Tp[i] = T[i] + c*(T[i+1]+T[i-1]-2*T[i])
    T,Tp = Tp,T
    t += h

    # Make plots at the given times
    if abs(t-t1)<epsilon:
        plot(T)
    if abs(t-t2)<epsilon:
        plot(T)
    if abs(t-t3)<epsilon:
        plot(T)
    if abs(t-t4)<epsilon:
        plot(T)
    if abs(t-t5)<epsilon:
        plot(T)

xlabel("x")
ylabel("T")

#%%

# Crank-Nicholson Method
# Stable, Fast, good for 1D and 2D
# Mathematically complex (requires solving tridiagonal matrices).
# Hard to implement on non-rectangular geometries.

import numpy as np
import matplotlib.pyplot as plt

def thomas_algorithm(a, b, c, d):
    """
    Solves the Tridiagonal matrix system Ax = d.
    a: lower diagonal
    b: main diagonal
    c: upper diagonal
    d: right hand side vector
    """
    n = len(d)
    c_prime = np.zeros(n)
    d_prime = np.zeros(n)
    x = np.zeros(n)

    # Forward sweep
    c_prime[0] = c[0] / b[0]
    d_prime[0] = d[0] / b[0]
    
    for i in range(1, n):
        temp = b[i] - a[i-1] * c_prime[i-1]
        if i < n-1:
            c_prime[i] = c[i] / temp
        d_prime[i] = (d[i] - a[i-1] * d_prime[i-1]) / temp

    # Back substitution
    x[-1] = d_prime[-1]
    for i in range(n-2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i+1]
        
    return x

# --- Parameters ---
N = 50              # Grid size (50x50)
L = 1.0             # Domain length
V = 100.0           # Voltage at x=0
dx = L / (N - 1)
dy = dx
dt = 0.001          # Time step (pseudo-time)
alpha = 1.0         # Diffusion coefficient
r = alpha * dt / (2 * dx**2) # Courant number factor

# Initialize Grid
phi = np.zeros((N, N))

# --- Boundary Conditions ---
# phi(x=0, y) = V  -> Left Edge
phi[:, 0] = V
# Others are already 0

# --- Main Loop (ADI Method) ---
# We solve until the solution stops changing (Steady State)
max_iter = 2000
tolerance = 1e-4

print("Starting Crank-Nicolson (ADI) solver...")

for t in range(max_iter):
    phi_old = phi.copy()
    
    # --- Step 1: Implicit in X, Explicit in Y ---
    # We solve for each row (horizontal lines)
    for j in range(1, N-1): # For each row y_j
        # Setup Tridiagonal System for the row
        # A*phi_new = B*phi_old + boundary_terms
        
        # Diagonals for implicit X part
        a = -r * np.ones(N-2) # Lower
        b = (1 + 2*r) * np.ones(N-2) # Main
        c = -r * np.ones(N-2) # Upper
        
        # RHS (Explicit in Y)
        # d term includes the current Y-neighbors
        d = (1 - 2*r) * phi[j, 1:-1] + r * (phi[j+1, 1:-1] + phi[j-1, 1:-1])
        
        # Add X-boundary conditions to the RHS vector
        d[0] += r * phi[j, 0]   # Left boundary (V)
        d[-1] += r * phi[j, -1] # Right boundary (0)
        
        # Solve for the interior of this row
        phi[j, 1:-1] = thomas_algorithm(a, b, c, d)

    # --- Step 2: Implicit in Y, Explicit in X ---
    # We solve for each column (vertical lines)
    for i in range(1, N-1): # For each column x_i
        # Diagonals for implicit Y part
        a = -r * np.ones(N-2)
        b = (1 + 2*r) * np.ones(N-2)
        c = -r * np.ones(N-2)
        
        # RHS (Explicit in X using the half-step values we just computed)
        d = (1 - 2*r) * phi[1:-1, i] + r * (phi[1:-1, i+1] + phi[1:-1, i-1])
        
        # Add Y-boundary conditions
        d[0] += r * phi[0, i]   # Bottom boundary (0)
        d[-1] += r * phi[-1, i] # Top boundary (0)
        
        phi[1:-1, i] = thomas_algorithm(a, b, c, d)
        
    # Re-enforce corners/boundaries just in case
    phi[:, 0] = V
    phi[:, -1] = 0
    phi[0, :] = 0
    phi[-1, :] = 0

    # Check for convergence (steady state)
    max_diff = np.max(np.abs(phi - phi_old))
    if max_diff < tolerance:
        print(f"Converged at iteration {t}")
        break

# --- Plotting ---
plt.figure(figsize=(7, 6))
plt.imshow(phi, cmap='inferno', origin='lower', extent=[0, L, 0, L])
plt.colorbar(label='Potential (V)')
plt.title(f'Crank-Nicolson (ADI) Solution\nConvergence: {max_diff:.2e}')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# %%

# Finite Element Method

# Instead of a grid of squares, FEM breaks the object into small triangles
# Complex Geometries: Perfect for curved shapes, round holes, or irregular boundaries (like a car engine block).
# Higher Accuracy: You can use more triangles in areas where the voltage changes quickly.
# Hard to Code: Writing an FEM solver from scratch is much harder than FDM.
# Computationally heavy: Requires solving a massive system of linear equations ($Ax=b$).


#%%

# Spectral Method / Fast Fourier Transform
# Very Fast, infinite precision
# Restricted Geometry: Only works on simple shapes (rectangles, circles, cylinders). 
# You cannot easily use this for an L-shaped room.

# %%

# For homework/learning: Stick to Gauss-Seidel or SOR. They are easy to debug.

# For a curved domain (e.g., airflow over a wing): You must use FEM.

# For a massive simulation on a square (e.g., weather): Use FFT or Multigrid methods.

# %%

# 만약 윗 방법으로 풀고 싶다면, 
# write a python code of 2 dimensional laplace equation with initial condition
#  phi(x = 0 , y)= V , phi(x, y=0) = 0 , phi(x, y = m) = 0 , phi(x = m , y) = 0. 
# try not to use scipy's partial
# 으로 풀면 된다.

