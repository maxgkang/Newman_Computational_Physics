
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
# import seaborn
# import pandas as pd
# import statsmodels as std 


# 위에 한 것 빨리 하려면 ctrl + / 누르면 된다
# 창 닫기 빠르게 하는 방법은 ctrl + w 하면 된다
# shift + 위/아래 키 누르면 드래그가 쉽다

# Chapter 4
Alpha = 1.602e-19 # spacing should be removed
print(Alpha)
print(type(Alpha)) # this is a float, not a string


# python pi is different from actual pi

Beta = 2.9999999999999

print("Same") if Beta == 3 else print("Not Same")

# define epsilon 

epsilon = 0.00000001
print("Now it's same") if abs(3 - Beta) < epsilon else print("Still not same")

root_2 = np.sqrt(2) - epsilon
print(root_2)

# %%
# 적분
import numpy as np
from scipy.integrate import nquad

f = lambda x, y: (2**x)*y
ranges = [[-1, 1], [-2, 4]]

Result_of_integration = nquad(
    f, ranges
)
print(Result_of_integration) # 이렇게만 하면, (I, error estimates) 가 돼버림
I, error_estimates_1 = Result_of_integration
print(I)
# %%

# 범위가 다른 변수로 제한된 함수적분

import numpy as np
from scipy.integrate import nquad

f = lambda y, x: x + y

# 보기도 좋고 에러도 안 나는 깔끔한 형태
ranges = [
    lambda x: [0, x],  # y 범위: 0 ~ x
    [0, 1]                       # x 범위: 0 ~ 1
]

Result, error_estimates_2 = nquad(f, ranges)

print("결과:", Result)



# %%
import numpy as np
from scipy.integrate import nquad

# 1. 적분할 함수 (순서 중요: z -> y -> x)
# 안쪽에서 적분할 변수(z)를 가장 앞에 씁니다.
f = lambda z, y, x: np.exp(x + y + z)

# 2. 적분 범위 설정 (함수 인자 순서와 일치해야 함: z범위, y범위, x범위)
ranges = [
    lambda y, x: [0, x + y], 
    lambda x: [0, x],        
    [0, 1]                   
]

result, error = nquad(f, ranges)

print("결과:", result)

# %%

import numpy as np
from scipy import integrate

g = lambda x, y, z: (x**3)*(np.e ** (-x+y))*z*3

Result_of_integration_2 = integrate.nquad(g,
    [
        lambda y, z: [0, y+z],
        lambda y: [1, 3 * y],
        [0, 3]
    ]
)

RResult, error_estimates_3 = Result_of_integration_2
print(RResult)
# %%

