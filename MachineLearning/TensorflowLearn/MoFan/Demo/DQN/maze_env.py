

"""
Reinforcement learning maze example.

Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].

This script is the environment part of this example.
The RL is in RL_brain.py.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""


import numpy as np
import time
# import sys
import tkinter as tk

import random
import os

# if sys.version_info.major == 2:
#     import Tkinter as tk
# else:
#     import tkinter as tk

UNIT = 40   # pixels
MAZE_H = 15  # grid height
MAZE_W = 15  # grid width


class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']   #上下左右四个动作
        self.n_actions = len(self.action_space)
        self.n_features = 2
        self.title('maze')
        self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_H * UNIT))
        
        self.enemys = []
        for _ in range(5):
            position = [random.randint(0, MAZE_W)*UNIT + 20, random.randint(0, MAZE_H)*UNIT + 20]
            self.enemys.append(position)
        
        self.origin = np.array([20, 20])
        self._build_maze()
        
        
    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white', height=MAZE_H * UNIT, width=MAZE_W * UNIT)
        
        # create grids
        for c in range(0, MAZE_W * UNIT, UNIT):   # 画竖线
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_H * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)
        
#         # hell    黑色障碍物
#         hell1_center = self.origin + np.array([UNIT * 10, UNIT * 2])  # 黑色障碍物的位置  
#         hell2_center = self.origin + np.array([UNIT * 2, UNIT * 12])
#         
#         self.hell1 = self.canvas.create_rectangle(
#             hell1_center[0] - 15, hell1_center[1] - 15,
#             hell1_center[0] + 15, hell1_center[1] + 15,
#             fill='black')
#         self.hell2 = self.canvas.create_rectangle(
#             hell2_center[0] - 15, hell2_center[1] - 15,
#             hell2_center[0] + 15, hell2_center[1] + 15,
#             fill='black')

        # create oval    终点
        oval_center = self.origin + UNIT * 14
        self.destination = self.canvas.create_oval(
            oval_center[0] - 15, oval_center[1] - 15,
            oval_center[0] + 15, oval_center[1] + 15,
            fill='yellow')
        
        # create red rect
        self.player = self.canvas.create_rectangle(
            self.origin[0] - 15, self.origin[1] - 15,
            self.origin[0] + 15, self.origin[1] + 15,
            fill='red')
        
        self.enemys_draw = []
        for enemy in self.enemys:
            self.enemys_draw.append(self.canvas.create_rectangle(
                    enemy[0]-15, enemy[1]-15,
                    enemy[0]+15, enemy[1]+15,
                    fill = "black"
                    )
                )
        
        # pack all
        self.canvas.pack()
    
    def reset(self):
        self.update()
        time.sleep(1)
        self.canvas.delete(self.player)
        self.player = self.canvas.create_rectangle(
            self.origin[0] - 15, self.origin[1] - 15,
            self.origin[0] + 15, self.origin[1] + 15,
            fill='red')
        
        self.enemys.clear()
        for _ in range(5):
            position = [random.randint(0, MAZE_W)*UNIT + 20, random.randint(0, MAZE_H)*UNIT + 20]
            self.enemys.append(position)
        # return observation     归一化 0 - 1
        return (np.array(self.canvas.coords(self.player)[:2]) - np.array(self.canvas.coords(self.destination)[:2]))/(MAZE_H*UNIT)

    def step(self, action):
        
        s = self.canvas.coords(self.player)  #获取玩家当前位置
        base_action = np.array([0, 0])  #动作矩阵
        if action == 0:   # up
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:   # down
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:   # right
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:   # left
            if s[0] > UNIT:
                base_action[0] -= UNIT
        
        self.canvas.move(self.player, base_action[0], base_action[1])  # move agent
        next_coords = self.canvas.coords(self.player)  # next state    正方向的对角线顶点坐标
        ####################################################################################
        # 障碍物前进一步
        for i in range( len(self.enemys_draw) ):
            enemy_draw = self.enemys_draw[i]  # 绘制信息
            enemy = self.enemys[i]   # 实体信息
            
            enemy_option = np.array([0, 0])
            p = self.canvas.coords(enemy_draw)
            
            option = random.randint(0, 3)
            if option == 0:  # up
                if p[1] > UNIT:
                    enemy_option[1] -= UNIT
            elif option == 1:   # down
                if p[1] < (MAZE_H - 1) * UNIT:
                    enemy_option[1] += UNIT
            elif option == 2:   # right
                if p[0] < (MAZE_W - 1) * UNIT:
                    enemy_option[0] += UNIT
            elif option == 3:   # left
                if p[0] > UNIT:
                    enemy_option[0] -= UNIT
            
            self.canvas.move(enemy_draw, enemy_option[0], enemy_option[1])
            enemy_now_coord = self.canvas.coords(enemy_draw)
            enemy[0] = enemy_now_coord[0] + 15          # 更新实体数据
            enemy[1] = enemy_now_coord[1] + 15
        
        ############################################################################
        # reward function  判断玩家前进一步之后到了哪里
        for i in range( len(self.enemys_draw) ):
            enemy_draw = self.enemys_draw[i]
            if next_coords in [self.canvas.coords(enemy_draw)]:
                reward = -1
                done = True
        
        if next_coords == self.canvas.coords(self.destination):
            reward = 1
            done = True
#         elif next_coords in [self.canvas.coords(self.hell1)]:     # 障碍物
#             reward = -1
#             done = True
#         elif next_coords in [self.canvas.coords(self.hell1)]:
#             reward = -1
#             done = True
        else:
            reward = 0
            done = False
        
        s_ = (np.array(next_coords[:2]) - np.array(self.canvas.coords(self.destination)[:2]))/(MAZE_H*UNIT)
        time.sleep(0.1)
        return s_, reward, done

    def render(self):
        # time.sleep(0.01)
        self.update()




























