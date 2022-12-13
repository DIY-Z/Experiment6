import gzip
import os
import numpy as np
import matplotlib.pyplot as plt

local_file = '/data/HomeWork/Experiment6/data'
files = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz',
        't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']


def load_local_mnist(filename):# 加载文件
    paths = []
    file_read = []
    for file in files:
        paths.append(os.path.join(filename, file))
    for path in paths:
        file_read.append(gzip.open(path, 'rb'))
    # print(file_read)

    train_labels = np.frombuffer(file_read[1].read(), np.uint8, offset=8)#文件读取以及格式转换
    train_images = np.frombuffer(file_read[0].read(), np.uint8, offset=16) \
    .reshape(len(train_labels), 28, 28)
    test_labels = np.frombuffer(file_read[3].read(), np.uint8, offset=8)
    test_images = np.frombuffer(file_read[2].read(), np.uint8, offset=16) \
    .reshape(len(test_labels), 28, 28)
    return (train_images, train_labels), (test_images, test_labels)


def main():
    (x_train, y_train), (x_test, y_test) = load_local_mnist(local_file)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    fig, ax = plt.subplots(nrows=6, ncols=6, sharex=True, sharey=True)#显示图像
    ax = ax.flatten()
    for i in range(36):
        img=x_test[i].reshape(28,28)
        # img = x_train[y_train == 8][i].reshape(28, 28) # 显示标签为8的数字图像
        ax[i].set_title(y_test[i])
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
