

from collections import deque
from math import sin, cos, atan2, pi, hypot
import random


class Player(object):

    def __init__(self, position, speed = 0.5, direction = 0 * pi):  #设置初始位置和目的地
        
        self.position = position
        self.speed = speed  #小球运动速度设置，应当设置成变量
        self.direction = direction  #范围在0-360之间，以弧度为单位
        
        self.target = (random.random() * 400, random.random() * 400)
        self.history = deque(maxlen = 64) #历史轨迹点
    
    #本对象向前一步走，同时更新其他内容
    def update(self, action, dt):  #传入全局bots，后面可以获取周边环境
        
        px, py = self.position
        tx, ty = self.target
        angle = atan2(ty - py, tx - px)  #这里写避碰算法
        
        px %= 400
        py %= 400
        self.direction %= 2 * pi
        
        self.speed += action[0] * dt
        self.direction += action[1] * dt
        self.speed %= 20
        self.direction %= 2 * pi
        
        self.position = (px + cos(self.direction) * self.speed, py + sin(self.direction) * self.speed)
        if random.random() > 0.7:
            self.history.append(self.position)
        
    
    def reset(self):
        # self.position = (random.random() * 400, random.random() * 400)
        self.speed = random.random() * 5 + 1
        self.direction = random.random() * 2 * pi
        
        self.target = (random.random() * 400, random.random() * 400)
        self.history = deque(maxlen = 64) #历史轨迹点
    
    def dis_with_target(self):
        px, py = self.position
        tx, ty = self.target
        return hypot(tx - px, ty - py)
    
    
    
    
    
    
    
    
    
    