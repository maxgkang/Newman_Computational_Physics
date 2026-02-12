# %%
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
# %%


textfile_2 = np.loadtxt(r"C:\Users\user\Desktop\coding\python codes\Computational_Physics\textfile_2.txt", str )

print(textfile_2) # 창닫기 쉽게 하는 법: ctrl + w

y_value = textfile_2[1:, 1]

x_value = np.arange(len(y_value))

y_value = y_value.astype(float)

plt.plot(x_value, y_value)
plt.show()


# %%

x_values = np.linspace(0, 2 * np.pi, 100)
y_values = np.random.normal(0, 1, size = 100) #평균, 표준편차, 벡터내원소 갯수

plt.plot(x_values, y_values)
plt.show()


plt.plot(x_values, y_values, "o")

# x 좌표 pi 단위로 바꾸기
tick_positions = [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]
tick_labels = [r'$0$', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$']
plt.xticks(tick_positions, tick_labels)

# y 좌표 e 단위로 바꾸기
y_tick_positions = [-np.e, 0, np.e]
y_tick_labels = [r'$-e$', r'$0$', r'$e$']
plt.yticks(y_tick_positions, y_tick_labels)

# 축 이름 붙이기
plt.xlabel("Angular velocity")
plt.ylabel("Speed of Light")
plt.title("Physics Experiment")
plt.xlim(2 * np.pi , 0) 
plt.ylim(-np.e , np.e)

plt.show()


# %%

# 산점도 

# Sample data
x = np.random.normal(0, 1, 100)
y = np.random.normal(0, 1, 100)

plt.scatter(x, y)
plt.show()

# %%

# 막대그래프


# 데이터 (예: 도시와 기온)
cities = ['Seoul', 'Tokyo', 'Hanoi', 'New York']
temps = [10, 12, 22, 5]

# 막대그래프 그리기 (x축, y축)
plt.bar(cities, temps, color=['red', \
                              'blue', 'green', 'orange'],\
                                  width=0.6, edgecolor='black')

# 제목 및 라벨 추가 (선택사항)
plt.title("City Temperatures")
plt.xlabel("City")
plt.ylabel("Temp (C)")

plt.show()
# %%

# plotting more than 2 graphs simultaneously

x_new_values = np.linspace(0,np.pi ** 2, 100)
y_new_values_1 = np.random.normal(7, 1/4, size = 100)
y_new_values_2 = np.random.normal(10, 2, size = 100 )

plt.plot(x_new_values, y_new_values_1, "o")
plt.plot(x_new_values, y_new_values_2)


plt.show()



# %%

# 3D graphics

import vpython

# %%

