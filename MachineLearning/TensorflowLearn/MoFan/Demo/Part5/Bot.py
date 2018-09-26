


from collections import deque
from math import sin, cos, atan2, pi, hypot
import random


class Bot(object):
    
    def __init__(self, position, speed = 0.5, direction = 0 * pi):  #设置初始位置和目的地   direction弧度表示和存储
        
        self.position = position
        self.speed = speed  #小球运动速度设置，应当设置成变量
        self.direction = direction  #范围在0-360之间，以弧度为单位
        
        self.target = (random.random() * 400, random.random() * 400)
        self.history = deque(maxlen = 64) #历史轨迹点
    
    #本对象向前一步走，同时更新其他内容
    def update(self, bots, action):  #传入全局bots，后面可以获取周边环境
        
        px, py = self.position
        tx, ty = self.target
        angle = atan2(ty - py, tx - px)  #这里写避碰算法
        # 判断是否到达目的地，到达 奖励 
        if hypot(tx - px, ty - py) < 10:  #如果达到目的地，重新设置目标和速度
            self.target = self.set_target()
            self.speed = random.random() * 5 + 0.1;
        
        #边界判断，一个边界进入，另一个边界出来    ===  新公式，取余计算
        px %= 400
        py %= 400
        self.direction = angle
        self.direction %= 2 * pi
        
        #  前进并计入历史轨迹
        self.position = (px + cos(self.direction) * self.speed, py + sin(self.direction) * self.speed)
        if random.random() > 0.7:
            self.history.append(self.position)
































