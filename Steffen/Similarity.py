import numpy as np
import matplotlib.pyplot as plt

N = 1
d = 1 * np.sqrt(1 / N)
c_sigma = np.sqrt(1 / (N + 1))
z_k = np.random.randn(1000, N)
s_sigma = (1 - c_sigma) * np.ones((1000, N))

x_1_values = []
x_2_values = []
x_3_values = []

for i in range(1000):
    s_sigma[i] = (1 - c_sigma) * s_sigma[i] + c_sigma * z_k[i]

    x_1 = np.exp((c_sigma / d) * (np.linalg.norm(s_sigma[i]) / np.sqrt(N) - 1))  # eq. 10
    x_2 = np.exp((c_sigma / (2 * d)) * (np.linalg.norm(s_sigma[i] ** 2) / N - 1))  # eq. 10
    x_3 = np.exp(1 / 2 / d / N * ((np.linalg.norm(s_sigma[i])) ** 2 - N)) # eq. 10

    x_1_values.append(x_1)
    x_2_values.append(x_2)
    x_3_values.append(x_3)

plt.scatter(z_k.flatten(), x_1_values, label='x_1')
plt.scatter(z_k.flatten(), x_2_values, label='x_2')
plt.scatter(z_k.flatten(), x_3_values, label='x_3')
plt.legend()
plt.show()