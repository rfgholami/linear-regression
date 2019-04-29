from compute_cost import compute_cost
from gradient_descent import gradient_descent
from predict import predict
from utils import load_dataset, feature_normalize, add_x0
import numpy as np
import matplotlib.pyplot as plt

data = load_dataset()

X_Original = data[:, 0:2]
y = data[:, 2:3]

X_Original, mu, sigma = feature_normalize(X_Original)

X = add_x0(X_Original)
m = X.shape[0]
n = X.shape[1]
learning_rate = .01
theta = np.zeros((n, 1))

his = np.zeros((200, 1))

for i in range(200):
    grad = gradient_descent(X, y, theta, learning_rate, m)
    theta = theta - grad
    cost = compute_cost(X, y, theta)
    his[i, :] = cost

plt.plot(his, label='cost')

plt.ylabel('cost')
plt.xlabel('step')
plt.title("linear regression'")

legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()

plt.show()

x = np.array([[1203, 3]], dtype=float)
predicted = str(predict(x, theta, mu, sigma))

print ("final cost = " + str(his[199, :]))
print ("theta = " + str(theta))
print("predicted value for " + str(x) + " = " + predicted)
