# -*- coding:utf-8 -*-
from keras.models import Sequential
from keras.layers import Flatten, Dense
from qlearning.games import Catch
from keras.optimizers import *
from qlearning import Agent

if __name__ == "__main__":
    grid_size = 10  # 每一帧都是10*10的方格
    hidden_size = 128
    nb_frames = 1  # 一帧
    catch = Catch(grid_size=grid_size)  # 定义一个游戏
    catch.nb_actions = 3  # 游戏有三种决策，左走、右走、不动
    model = Sequential()
    model.add(Flatten(input_shape=(nb_frames, grid_size, grid_size)))  # input_shape (None, 1, 10, 10)
    model.add(Dense(hidden_size, activation='relu'))
    model.add(Dense(hidden_size, activation="relu"))
    model.add(Dense(catch.nb_actions, activation='tanh'))  # 输出神经元个数是3，代表左走、右走、保持不动
    # model.add(Dense(catch.nb_actions))  # 输出神经元个数是3，代表左走、右走、保持不动
    model.compile(sgd(lr=0.2), "mse")  # sgd优化器，学习率0.2
    agent = Agent(model=model)   # 定义一个玩家
    agent.train(catch, batch_size=10, nb_epoch=1000, epsilon=.1)  # 玩家开始训练游戏
    agent.play(catch)
