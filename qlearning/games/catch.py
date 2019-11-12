import numpy as np
from .game import Game


class Catch(Game):
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.won = False
        super(Catch, self).__init__()

    def reset(self):
        n = np.random.randint(0, self.grid_size, size=1)  # TODO 是不是不减一？
        m = np.random.randint(1, self.grid_size-1, size=1)  # TODO 是不是应该减一?
        # 随机初始化一个状态，[0, n]表示水果所在位置，0行n列；m表示盘子中心点位置
        self.state = np.asarray([0, n, m])[np.newaxis]

    @property
    def name(self):
        return "Catch"

    @property
    def nb_actions(self):
        """
        左走，右走，不动，三种
        :return:
        """
        return self._nb_actions

    @nb_actions.setter
    def nb_actions(self, value):
        if not isinstance(value, int):
            raise ValueError("nb_actions must be an integers!")
        if value <=0 or value > 100:
            raise ValueError("nb_actions must between 1 ~ 100!")
        self._nb_actions = value

    def play(self, action):
        """
        玩一次游戏，产生的后果是盘子水平动一下，水果下降一行
        :param action:
        :return:
        """
        state = self.state  # 获取当前状态
        if action == 0:  # 左走
            action = -1
        elif action == 1:  # 不动
            action = 0
        else:  # 右走
            action = 1
        f0, f1, basket = state[0]  # (f0, f1)表示水果所在位置，basket表示盘子所在位置
        new_basket = min(max(1, basket+action), self.grid_size-2)
        f0 += 1  # 水果下降一行
        out = np.asarray([f0, f1, new_basket])[np.newaxis]
        assert len(out.shape) == 2
        self.state = out

    def get_state(self):
        """
        获取当前游戏界面。水果所在位置，和盘子所在位置为1，其他为0。
        :return:
        """
        im_size = (self.grid_size, )*2
        state = self.state[0]
        canvas = np.zeros(im_size)
        canvas[state[0], state[1]] = 1
        canvas[-1, state[2]-1:state[2]+2] = 1
        return canvas

    def get_score(self):
        row, col, basket = self.state[0]
        if row == self.grid_size-1:  # 水果到最后一行了
            if abs(col - basket) <= 1:  # 盘子接住了，赢了，得一分
                self.won = True
                return 1
            else:
                return -1   # 没接住，得-1分
        else:  # 水果没有到最后一行，得0分
            return 0

    def is_over(self):
        if self.state[0, 0] == self.grid_size-1:  # 水果到达最后一行了，游戏结束
            return True
        return False

