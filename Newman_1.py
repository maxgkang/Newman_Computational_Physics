# %% 
import numpy as np
# import scipy.integrate
# import matplotlib.pyplot
# %%

matrix_A = np.zeros([4,4], float)

matrix_B = np.empty([4,4])
array_1 = np.arange(1, 17)
matrix_B = np.reshape(array_1, (4,4))
print(matrix_B)

# np.loadtxt(r"C:\Users\username\Desktop\data.txt")

# "C:\Users\user\Desktop\coding\python codes\Computational_Physics\textfile.txt" 
# Shift 누른 채 우클릭, a(경로로 복사) 하면 바로 복사된다

textfile = np.loadtxt(r"C:\Users\user\Desktop\coding\python codes\Computational_Physics\textfile.txt", float) # 앞에 r을 붙여주니까 이제야 잘 된다.

print(textfile)
data_1 = textfile[:][2] # Row vector
print(data_1)

col_vector = data_1.reshape(-1, 1)
print(col_vector)

# %%
G = 100.10203
def gravitational_force(m, M, r):
    if r <= 0:
        print("거리 오류!")
        return None
    return G * (m * M) / ( r ** 2)

result = gravitational_force(2, 3, 0)

if result == None:
    pass

if result is not None:
    print(result)
# Both works.
# return None: "Stop right now and leave the function.
# pass: "Do nothing and continue to the next line

# %%
a = np.reshape(np.arange(16), (4,4))
b = a
c = np.copy(a)

for i in list(range(0, 3)):
    a[0][i] = 0

print(a, b, c, sep = '\n') # This makes the output perpendicular to each other
# %% 

vector = np.arange(1, 16)
vector_logged = np.log(vector)
print(vector_logged)

# %%


# Jupyter 상 정리정돈된 값들을 보려면 배경 우클릭, run in an interactive window, run all codes 하면 다 뜬다

