import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


def batch(x):
    x = x.reshape(-1, 10)
    y = []
    for i in range(1, len(x)):
        y.append(np.concatenate([x[i - 1], x[i]]))
    print("0. ", y[0])
    print("1. ", y[1])
    return y


def new_batch(x):
    return x.reshape(-1, 10)


if __name__ == "__main__":
    # Create Continuous signal
    t = np.array([i for i in range(2000)])
    x = np.array([np.sin(10 * 2 * np.pi / 1000 * i) for i in t])

    # Create filter bank
    b0, a0 = signal.butter(2, [8, 12], "bandpass", fs=1000)

    # continuous filter
    y_cont = signal.lfilter(b0, a0, x=x)
    plt.plot([i for i in range(60)], y_cont[:60])
    plt.savefig("y-cont.png")
    plt.clf()

    # batch filtering
    x_batch = batch(x)
    y_batch = []
    zi = [0, 0, 0, 0]
    for x_sample in x_batch:
        y, zi = signal.lfilter(b0, a0, x=x_sample, zi=zi)
        # y = signal.lfilter(b0, a0, x=x_sample)
        y_batch.append(y)

    plt.plot([i for i in range(0, 20)], x_batch[0])
    plt.plot([i for i in range(10, 30)], x_batch[1])
    plt.plot([i for i in range(20, 40)], x_batch[2])
    plt.plot([i for i in range(30, 50)], x_batch[3])
    plt.plot([i for i in range(40, 60)], x_batch[4])
    plt.savefig("x-batch.png")
    plt.clf()
    plt.plot([i for i in range(0, 20)], y_batch[0])
    plt.plot([i for i in range(10, 30)], y_batch[1])
    plt.plot([i for i in range(20, 40)], y_batch[2])
    plt.plot([i for i in range(30, 50)], y_batch[3])
    plt.plot([i for i in range(40, 60)], y_batch[4])
    plt.savefig("y-batch-old.png")
    plt.clf()

    # NEW batch filtering
    x_batch = new_batch(x)
    y_batch = []
    zi = [0, 0, 0, 0]
    for x_sample in x_batch:
        y, zi = signal.lfilter(b0, a0, x=x_sample, zi=zi)
        y_batch.append(y)

    plt.plot([i for i in range(1, 11)], y_batch[0])
    plt.plot([i for i in range(10, 20)], y_batch[1])
    plt.plot([i for i in range(20, 30)], y_batch[2])
    plt.plot([i for i in range(30, 40)], y_batch[3])
    plt.plot([i for i in range(40, 50)], y_batch[4])
    plt.plot([i for i in range(50, 60)], y_batch[5])
    plt.savefig("y-batch-new.png")
    plt.clf()
    plt.plot(t, np.array(y_batch).reshape(-1,))
    plt.savefig("y-batch-new-all.png")
    plt.clf()
