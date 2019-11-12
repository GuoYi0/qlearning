import numpy as np
from random import sample
from keras import backend as K


class Memory(object):
    def __init__(self):
        pass

    def remember(self, S, a, r, S_prime, game_over):
        pass

    def get_batch(self, model, batch_size):
        pass


class ExperienceReplay(Memory):
    def __init__(self, memory_size=100, fast=True):
        self.fast = fast
        self.memory = []
        self._memory_size = memory_size
        self.input_shape = None
        super(ExperienceReplay, self).__init__()

    def reset_memory(self):
        """清空memory"""
        self.memory = []

    def remember(self, s, a, r, s_prime, game_over):
        """
        决策之前的状态，所做的动作，产生的分数，决策之后的状态，游戏是否结束
        :param s:
        :param a:
        :param r:
        :param s_prime:
        :param game_over:
        :return:
        """
        self.input_shape = s.shape[1:]  # [nb_frames, grid_size, grid_size]
        self.memory.append(np.concatenate([s.flatten(), np.array(a).flatten(),
                                           np.array(r).flatten(), s_prime.flatten(),
                                           1 * np.array(game_over).flatten()]))
        if 0 < self._memory_size < len(self.memory):
            self.memory.pop(0)

    def get_batch(self, model, batch_size, gamma=0.9):
        if self.fast:
            return self.get_batch_fast(model, batch_size, gamma)

    def get_batch_fast(self, model, batch_size, gamma):
        if len(self.memory) < batch_size:
            return None
        samples = np.array(sample(self.memory, batch_size))  # 随机采样出一个batch
        if not hasattr(self, 'batch_function'):
            self.set_batch_function(model, self.input_shape, batch_size, model.get_output_shape_at(0)[-1], gamma)
        S, targets = self.batch_function([samples])
        return S, targets

    def set_batch_function(self, model, input_shape, batch_size, nb_actions, gamma):
        input_dim = np.prod(input_shape)  # [nb_frames, grid_size, grid_size]
        samples = K.placeholder(shape=(batch_size, input_dim * 2 + 3))
        S = samples[:, 0: input_dim]  # 上一个状态
        a = samples[:, input_dim]  # 执行的动作
        r = samples[:, input_dim + 1]  # 得分
        S_prime = samples[:, input_dim + 2: 2 * input_dim + 2]  # 产生的新状态
        game_over = samples[:, 2 * input_dim + 2: 2 * input_dim + 3]  # 游戏是否结束
        r = K.reshape(r, (batch_size, 1))
        r = K.repeat(r, nb_actions)  # [batch_size, nb_actions, 1]
        r = K.reshape(r, (batch_size, nb_actions))
        game_over = K.repeat(game_over, nb_actions)
        game_over = K.reshape(game_over, (batch_size, nb_actions))
        S = K.reshape(S, (batch_size, ) + input_shape)
        S_prime = K.reshape(S_prime, (batch_size, ) + input_shape)
        X = K.concatenate([S, S_prime], axis=0)
        Y = model(X)
        Qsa = K.max(Y[batch_size:], axis=1)  # 产生新状态以后，后续的动作
        Qsa = K.reshape(Qsa, (batch_size, 1))
        Qsa = K.repeat(Qsa, nb_actions)  # [batch_size, nb_actions, 1]
        Qsa = K.reshape(Qsa, (batch_size, nb_actions))
        delta = K.reshape(self.one_hot(a, nb_actions), (batch_size, nb_actions))  # 执行的动作
        # Y[:batch_size] 的 shape 是(batch_size, nb_actions)
        targets = (1 - delta) * Y[:batch_size] + delta * (r + gamma * (1 - game_over) * Qsa)  # TODO 这里怎么理解？
        self.batch_function = K.function(inputs=[samples], outputs=[S, targets])

    def one_hot(self, seq, num_classes):
        return K.one_hot(K.reshape(K.cast(seq, "int32"), (-1, 1)), num_classes)  # [batch_size, 1, num_classes]

