



from collections import deque
from math import sin, cos, atan2, pi, hypot
import random


class Bot(object):
    
    def __init__(self, width, height, position, speed = 0.5, direction = 0 * pi):  #设置初始位置和目的地
        self.width = width
        self.height = height
        
        self.position = position
        self.speed = speed  #小球运动速度设置，应当设置成变量
        self.direction = direction  #范围在0-360之间，以弧度为单位
        
        self.target = self.set_target()
        self.history = deque(maxlen = 64) #历史轨迹点
    
    #本对象向前一步走，同时更新其他内容
    def update(self, bots, action = 0):  #传入全局bots，后面可以获取周边环境
        
        px, py = self.position
        tx, ty = self.target
        angle = atan2(ty - py, tx - px)  #这里写避碰算法
        
        if action == 0:
            self.direction = angle
        elif action == 1:
            self.direction += 1/6 * pi
        else:
            self.direction -= 1/6 * pi
        
        for bot in bots:
            bx, by = bot.position
            if hypot(px - bx, py - by) < 100:
                self.direction += 1/6 * pi
        # 判断是否到达目的地，到达 奖励 
        if hypot(tx - px, ty - py) < 10:  #如果达到目的地，重新设置目标和速度
            self.target = self.set_target()
            self.speed = random.random() * 5 + 0.1;
        
        #边界判断，一个边界进入，另一个边界出来
        if px >= self.width:
            px = 0
        elif px <= 0:
            px = self.width
        if py >= self.height:
            py = 0
        elif py <= 0:
            py = self.height
        
        if self.direction > 2 * pi:
            self.direction -= 2 * pi
        elif self.direction < 0:
            self.direction -= 2 * pi
        
        #  前进并计入历史轨迹
        self.position = (px + cos(self.direction) * self.speed, py + sin(self.direction) * self.speed)
        if random.random() > 0.7:
            self.set_history()
        
    
    def set_target(self):  #在圆内生成设置的目标点
        
        x = random.random() * self.width
        y = random.random() * self.height
        return (x, y)
    
    def set_history(self):  #历史轨迹更新
        self.history.append(self.position)












