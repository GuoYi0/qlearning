import numpy as np
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


if __name__ == "__main__":
    a = np.zeros((3,3))
    b = [a]*10
    print(np.expand_dims(b, 0).shape)

