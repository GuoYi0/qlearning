# -*- coding:utf-8 -*-
class Game(object):
    """
    定义一个基类
    """
    def __init__(self):
        self.state = None
        self._nb_actions = 0
        self.reset()

    @property
    def name(self):
        return "Game"

    @property
    def nb_actions(self):
        return 0

    def reset(self):
        raise NotImplementedError("This method has not been implemented")

    def play(self, action):
        pass

    def get_state(self):
        return None

    def get_score(self):
        raise NotImplementedError("This method has not been implemented")

    def is_over(self):
        return False

    def is_won(self):
        return False

    def get_frame(self):
        return self.get_state()

    def draw(self):
        return self.get_state()

    def get_possible_actions(self):
        return range(self._nb_actions)

