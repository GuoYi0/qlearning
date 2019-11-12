# -*- coding: utf-8 -*-
from .memory import ExperienceReplay
import numpy as np
import os
import matplotlib.pyplot as plt
import imageio
import shutil

class Agent(object):
    """一个玩家的行为"""
    def __init__(self, model, memory=None, memory_size=1000, nb_frames=None):
        assert len(model.get_output_shape_at(0)) == 2, "Model's output shape should be (nb_samples, nb_actions)."
        if memory:
            self.memory = memory
        else:
            self.memory = ExperienceReplay(memory_size)  # 缓存

        if not nb_frames and not model.get_input_shape_at(0)[1]:
            raise Exception("Missing argument : nb_frames not provided")
        elif not nb_frames:
            nb_frames = model.get_input_shape_at(0)[1]
        elif model.get_input_shape_at(0)[1] and model.get_input_shape_at(0)[1] != nb_frames:
            raise Exception("Dimension mismatch : time dimension of model should be equal to nb_frames.")
        self.model = model
        self.nb_frames = nb_frames  # 每次训练多少帧
        self.frames = None

    def train(self, game, nb_epoch=1000, batch_size=64, gamma=0.9, epsilon=(1.0, 0.1),
              epsilon_rate=0.5, reset_memory=False, observe=0, checkpoint=None):
        """
        :param game:  所定义的游戏，一个游戏类
        :param nb_epoch:
        :param batch_size:
        :param gamma:
        :param epsilon: 决定是随机做一个决策，还是用模型从输出做决策.
        默认值(1.0, 0.1)表示训练初期以概率1.0随机决策，后期慢慢衰减到以概率0.1随机决策
        :param epsilon_rate:
        :param reset_memory:
        :param observe:  # 先观察几下，不必急着决策
        :param checkpoint:
        :return:
        """
        self.check_game_compatibility(game)
        if type(epsilon) in {tuple, list}:
            delta = ((epsilon[0] - epsilon[1]) / (nb_epoch * epsilon_rate))
            final_epsilon = epsilon[1]
            epsilon = epsilon[0]
        else:
            final_epsilon = epsilon
            delta = 0.0

        model = self.model
        win_count = 0
        for epoch in range(nb_epoch):  # 开始训练
            loss = 0.0
            game.reset()  # 每个epoch都从随机初始化的状态开始
            self.clear_frames()
            if reset_memory:
                self.reset_memory()  # 清空memory
            game_over = False
            S = self.get_game_data(game)  # 获取训练数据，一个帧数为nb_frames的视频序列[1, nb_frames, grid_size, grid_size]
            while not game_over:
                if np.random.random() < epsilon or epoch < observe:
                    # 随机决策
                    a = int(np.random.randint(game.nb_actions))
                else:
                    q = model.predict(S)
                    a = int(np.argmax(q[0]))  # 选择得分最大的那个决策
                game.play(a)  # 执行这个决策
                r = game.get_score()  # 然后获得这个决策产生的一个分数
                S_prime = self.get_game_data(game)  # 获取更新以后的视频序列
                game_over = game.is_over()  # 如果水果到达最后一行，游戏就结束
                # 决策之前的状态，所做的动作，产生的分数，决策之后的状态，游戏是否结束
                transition = [S, a, r, S_prime, game_over]
                self.memory.remember(*transition)  # 记下这一步
                S = S_prime
                if epoch >= observe:
                    batch = self.memory.get_batch(model=model, batch_size=batch_size, gamma=gamma)
                    if batch:
                        inputs, targets = batch
                        loss += float(model.train_on_batch(inputs, targets))
                if checkpoint and ((epoch + 1 - observe) % checkpoint == 0 or epoch + 1 == nb_epoch):
                    model.save_weights('weights.dat')

            if game.is_won():
                win_count += 1
            if epsilon > final_epsilon and epoch >= observe:
                epsilon -= delta
            print("Epoch {:03d}/{:03d} | Loss {:.4f} | Epsilon {:.2f} | Win count {}".format(epoch + 1, nb_epoch, loss,
                                                                                             epsilon, win_count))

    def check_game_compatibility(self, game):
        """检查输入输出的匹配性"""
        if len(self.model.input_layers_node_indices) != 1:
            raise Exception('Multi node input is not supported.')
        game_output_shape = (1, None) + game.get_frame().shape  # (1, None, grid_size, grid_size)
        if len(game_output_shape) != len(self.model.get_input_shape_at(0)):  # 4
            raise Exception('Dimension mismatch. Input shape of the model should be compatible with the game.')
        else:
            for i in range(len(self.model.get_input_shape_at(0))):
                if self.model.get_input_shape_at(0)[i] and game_output_shape[i] \
                        and self.model.get_input_shape_at(0)[i] != game_output_shape[i]:
                    raise Exception('Dimension mismatch. Input shape of the model should be compatible with the game.')
        if len(self.model.get_output_shape_at(0)) != 2 or self.model.get_output_shape_at(0)[1] != game.nb_actions:
            raise Exception('Output shape of model should be (nb_samples, nb_actions).')

    def clear_frames(self):
        self.frames = None

    def reset_memory(self):
        self.memory.reset_memory()

    def get_game_data(self, game):
        """
        获取训练数据，shape是 [1, nb_frames, grid_size, grid_size]
        :param game:
        :return:
        """
        frame = game.get_frame()  # 获取当前游戏界面。盘子所在位置，和盘子所在位置为1，其他为0。
        if self.frames is None:
            self.frames = [frame] * self.nb_frames
        else:
            self.frames.append(frame)
            self.frames.pop(0)
        return np.expand_dims(self.frames, 0)

    def play(self, game, nb_epoch=10, epsilon=0., visualize=True):
        self.check_game_compatibility(game)
        model = self.model
        win_count = 0
        frames = []
        for epoch in range(nb_epoch):
            game.reset()  # 随机初始化一个状态
            self.clear_frames()  # 帧清空
            S = self.get_game_data(game)
            if visualize:
                frames.append(game.draw())  # 水果所在位置和盘子所在位置为1，其他为0。
            game_over = False
            while not game_over:
                if np.random.rand() < epsilon:
                    print("random")
                    action = int(np.random.randint(0, game.nb_actions))
                else:
                    q = model.predict(S)[0]
                    possible_actions = game.get_possible_actions()
                    q = [q[i] for i in possible_actions]
                    action = possible_actions[np.argmax(q)]
                game.play(action)
                S = self.get_game_data(game)
                if visualize:
                    frames.append(game.draw())
                game_over = game.is_over()
            if game.is_won():
                win_count += 1
        print("Accuracy {} %".format(100. * win_count / nb_epoch))
        if visualize:
            gifs = []
            if 'images' not in os.listdir('.'):
                os.mkdir('images')
            else:
                shutil.rmtree("images")
                os.mkdir('images')
            for i in range(len(frames)):
                ret = plt.imshow(frames[i], interpolation='none')
                gifs.append(ret)
                plt.savefig("images/" + game.name + str(i) + ".png")
            toGif("images", game.name)


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
