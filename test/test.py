import numpy as np
import os
import imageio


class father(object):
    def __init__(self):
        self.my_print()

    def my_print(self):
        print("father")


class son(father):
    def __init__(self):
        self.my_print()
        super(son, self).__init__()

    def my_print(self):
        print("son")


def toGif(path, name):
    def func(key):
        index = int(key.split(name)[1].split('.')[0])
        return index
    file_list = os.listdir(path)
    file_list = sorted(file_list, key=func)
    frames = []
    for png in file_list:
        frames.append(imageio.imread(os.path.join(path, png)))
    imageio.mimsave("result.gif", frames, 'GIF', duration=0.3)


if __name__ == "__main__":
    toGif("E:\qlearning\qlearning\examples\images", "Catch")
    # a = np.zeros((3,3))
    # b = [a]*10
    # print(np.expand_dims(b, 0).shape)

