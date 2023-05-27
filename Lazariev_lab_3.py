import math
import random
import matplotlib.pyplot as plt

print("Lab 3 by Viktor Lazariev, IKM-220d")
# 1) За наведеним вище алгоритмом промоделювати процес навчання та роботи двошарового персептрону.
# 2) Дослідити вплив початкових умов вагових коефіцієнтів на ефективність навчання:
# провести моделювання для рекомендованих значень та для інших, вільно обраних значень.
# Порівнювати варіанти за кількістю епох навчання, необхідних для нормальної роботи мережі.
# 3) Дослідити вплив коефіцієнту швидкості навчання на його ефективність.
# 4) Дослідити залежність ефективності алгоритму навчання від параметра k, що входить до ступеню в функції активації.
# 5) Для рекомендованого варіанту побудувати залежність E* від кількості епох навчання та визначити оптимальну кількість.
# У звіті про лабораторну роботу відобразити результати за пп.2-5 та навести найкращі вагові коефіцієнти.
# Зробити відповідні висновки.

# X_input=input("enter input vector X: ")
# print("X_input= ", X_input)     #00,01,10,11

def Fu(u):# F(u)
    k = 1
    return 1 / (1 + math.pow(math.e, (-k) * u))

def learningDataGeneration():
    data = list()
    for i in range(1000):
        # x1 = round(X_input[0])
        # x2 = round(X_input[1])
        x1 = round(random.uniform(0, 1), 4)
        # print("x1=", x1)
        x2 = round(random.uniform(0, 1), 4)
        # print("x2=", x2)
        if (x1 + 0.5) > x2 > (x1 - 0.5):
            d = 0
        else:
            d = 1
        data.append([x1, x2, d])
    return data

def learningAlgorithm(wCurrent, data):
    row_sum = 0
    u = [[0.0, 0.0], [0.0, 0.0]]
    F = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    wNew = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

    for i in range(len(data)):
        xFirst = data[i - 1][0]
        xSecond = data[i - 1][1]
        d = data[i - 1][2]
        b=1

        u[0][0] = wCurrent[0][0] * b + wCurrent[0][1] * xFirst + wCurrent[0][2] * xSecond
        u[0][1] = wCurrent[1][0] * b + wCurrent[1][1] * xFirst + wCurrent[1][2] * xSecond
        y1 = Fu(u[0][0])
        y2 = Fu(u[0][1])
        u[1][0] = wCurrent[2][0] * b + wCurrent[2][1] * y1 + wCurrent[2][2] * y2
        y = Fu(u[1][0])
        epsilon = d - y
        epsilon_end = (d - y) ** 2
        row_sum = row_sum + epsilon_end

        F[0][0] = Fu(u[0][0]) * (1 - Fu(u[0][0]))
        F[0][1] = Fu(u[0][1]) * (1 - Fu(u[0][1]))
        F[1][0] = Fu(u[1][0]) * (1 - Fu(u[1][0]))

        wNew[2][0] = wCurrent[2][0] + eta * epsilon * F[1][0]
        wNew[2][1] = wCurrent[2][1] + eta * epsilon * F[1][0] * y1
        wNew[2][2] = wCurrent[2][2] + eta * epsilon * F[1][0] * y2

        wNew[0][0] = wCurrent[0][0] + eta * epsilon * F[1][0] * wCurrent[2][1] * F[0][0]
        wNew[0][1] = wCurrent[0][1] + eta * epsilon * F[1][0] * wCurrent[2][1] * F[0][0] * xFirst
        wNew[0][2] = wCurrent[0][2] + eta * epsilon * F[1][0] * wCurrent[2][1] * F[0][0] * xSecond

        wNew[1][0] = wCurrent[1][0] + eta * epsilon * F[1][0] * wCurrent[2][2] * F[0][1]
        wNew[1][1] = wCurrent[1][1] + eta * epsilon * F[1][0] * wCurrent[2][2] * F[0][1] * xFirst
        wNew[1][2] = wCurrent[1][2] + eta * epsilon * F[1][0] * wCurrent[2][2] * F[0][1] * xSecond

        wCurrent = wNew
    return row_sum, wCurrent

def errorCalculation(w):
    xFirst = [[0, 0], [0, 1], [1, 0], [1, 1]]
    d = [0, 1, 1, 0]
    u = [[0.0, 0.0], [0.0, 0.0]]
    b = 1
    sum = 0

    for i in range(4):
        u[0][0] = w[0][0] * b + w[0][1] * xFirst[i - 1][0] + w[0][2] * xFirst[i - 1][1]
        u[0][1] = w[1][0] * b + w[1][1] * xFirst[i - 1][0] + w[1][2] * xFirst[i - 1][1]
        y1 = Fu(u[0][0])
        y2 = Fu(u[0][1])
        u[1][0] = w[2][0] * b + w[2][1] * y1 + w[1][2] * y2
        y = Fu(u[1][0])
        epsilon_end = (d[i - 1] - y) ** 2
        sum = sum + epsilon_end
    return sum / 2

eta = 1 # η коеф навч
initialW = [[0.1, -0.3, 0.4], [-0.7, -0.1, 0.01], [0.4, -0.2, 0.1]] # рекомендовані
# initialW = [[0.5, 0.2, -0.8], [0.3, -0.3, 0.04], [-0.74, 0.23, 0.11]] # власні з довільними знаками
# initialW = [[0.2, -0.4, 0.7], [-0.1, -0.04, 0.2], [0.5, -0.1, 0.09]] # власні з такими ж знаками
data = learningDataGeneration()
print(data)

print(errorCalculation(initialW))
NumberOfEpochs = 20
epochsArray = []
etaArray = []
for i in range(NumberOfEpochs):
    etaN, initialW = learningAlgorithm(initialW, data)
    print("Ẽ = ", str(errorCalculation(initialW)))
    print(round(etaN / 1000, 4))
    epochsArray.append(i)
    etaArray.append(errorCalculation(initialW))

plt.figure(num = "Lab 3 by Viktor lazariev")
plt.plot(epochsArray, etaArray, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.show()