import matplotlib.pyplot as plt
import numpy as np


def si_function(x: np.ndarray) -> np.ndarray:
    return np.sin(x)/x


def rectangle(x: np.ndarray) -> np.ndarray:
    return np.where(abs(x) <= 0.5, 1, 0)


if __name__ == '__main__':
    data_values = np.arange(-5, 5, 0.1)
    n = np.arange(1, 4, 1)
    # Calculate y Values
    sinus = np.sin(data_values)

    sinus_n = list()
    print(data_values * 2)
    for i in n:
        print(i)
        sinus_n.append(np.sin(data_values*i))

    tangens_hyperbolic = np.tanh(data_values)
    rect = rectangle(data_values)
    si = si_function(data_values)

    fig, ax = plt.subplots(nrows=2, ncols=3, sharex='all', sharey='all')
    ax[0, 0].plot(data_values, sinus)
    ax[0, 0].set_title('Sinus(x)')

    for i in range(len(sinus_n)):
        ax[0, 1].plot(data_values, sinus_n[i])
    ax[0, 1].set_title('Sinus(nx)')

    ax[0, 2].plot(data_values, tangens_hyperbolic)
    ax[0, 2].set_title('Tanh(x)')

    ax[1, 0].plot(data_values, rect)
    ax[1, 0].set_title('Rectangle(x)')

    ax[1, 1].plot(data_values, si)
    ax[1, 1].set_title('Si(x)')

    for j in n:
        ax[1, 2].plot(data_values, si_function(j*data_values))
    ax[1, 2].set_title('Si(nx)')

    plt.show()
