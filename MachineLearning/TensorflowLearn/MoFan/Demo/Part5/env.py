



import numpy as np
import pyglet
import random
from math import sin, cos, atan2, pi, hypot
from MoFan.Demo.Part5.Bot import Bot
from MoFan.Demo.Part5.Player import Player


class ArmEnv(object):
    viewer = None
    dt = .1    # refresh rate  刷新率   转动的速度和dt有关
    action_bound = [-1, 1]
    goal = {'x': 100., 'y': 100., 'l': 40}  #目标地点
    state_dim = 9  #状态维度  观测值
    action_dim = 2  #动作维度
    ##########################################################
    viewer = None
    dt = .1
    action_bound = [0, 2 * pi]
    state_dim = 9  #状态维度  观测值
    action_dim = 2  #动作维度     ------- 例子中是两个手臂的角度，这里是速度大小和方向
    goal = {"x" : 400, "y" : 50}
    
    def __init__(self):  #初始化基础数据
        self.arm_info = np.zeros(    # 2 行  2 列
                2, dtype=[('l', np.float32), ('r', np.float32)]
            )
        self.arm_info['l'] = 100        # 2 arms length
        self.arm_info['r'] = np.pi/6    # 2 angles information  30 degree
        self.on_goal = 0   #判断手臂端口  在蓝色目标快停留了多长时间
        #########################################################################
        #  初始化界面避碰信息
        self.player = Player((random.random() * 400, random.random() * 400), random.random() * 5 + 1, random.random() * 2 * pi)
        self.bots = []
        for _ in range(random.randint(0, 20)):
            position = (random.random() * 200, random.random() * 200)
            bot = Bot(position, random.random() * 5 + 0.1, random.random() * 2 * pi)
            self.bots.append(bot)
        
        # self.goal = {"x" : 400, "y" : 50}
        self.on_goal = 0
        

    def step(self, action):    #action是一个二维数组，表示手臂的旋转角度 =========================================================
        done = False
        action = np.clip(action, *self.action_bound)   #把输入控制在边界
        self.arm_info['r'] += action * self.dt
        self.arm_info['r'] %= np.pi * 2    # normalize

        (a1l, a2l) = self.arm_info['l']  # radius, arm length
        (a1r, a2r) = self.arm_info['r']  # radian, angle
        a1xy = np.array([200., 200.])    # a1 start (x0, y0)      这里需要搞清楚角度是以什么为基准的
        a1xy_ = np.array([np.cos(a1r), np.sin(a1r)]) * a1l + a1xy  # a1 end and a2 start (x1, y1)
        finger = np.array([np.cos(a1r + a2r), np.sin(a1r + a2r)]) * a2l + a1xy_  # a2 end (x2, y2)
        # normalize features
        dist1 = [(self.goal['x'] - a1xy_[0]) / 400, (self.goal['y'] - a1xy_[1]) / 400]
        dist2 = [(self.goal['x'] - finger[0]) / 400, (self.goal['y'] - finger[1]) / 400]
        r = -np.sqrt(dist2[0]**2+dist2[1]**2)    #这个可以表示第二根手臂的长度计算值
        
        # done and reward
        if self.goal['x'] - 40/2 < finger[0] < self.goal['x'] + 40/2:
            if self.goal['y'] - 40/2 < finger[1] < self.goal['y'] + 40/2:
                r += 1.
                self.on_goal += 1
                if self.on_goal > 50:   #达到目标地的次数50为一个回合，在目标点稳定一定时间
                    done = True
        else:
            self.on_goal = 0

        # state   状态标记是与动作相关的状态
        s = np.concatenate((a1xy_/200, finger/200, dist1 + dist2, [1. if self.on_goal else 0.]))    #状态记录
        return s, r, done    #done表示本次回合是否结束
        
        ################################################################################
        done = False
        action = np.clip(action, *self.action_bound)
        
        self.arm_info['r'] += action * self.dt
        self.arm_info['r'] %= np.pi * 2    # normalize
        self.player[""]
        
        (a1l, a2l) = self.arm_info['l']  # radius, arm length
        (a1r, a2r) = self.arm_info['r']  # radian, angle
        a1xy = np.array([200., 200.])    # a1 start (x0, y0)      这里需要搞清楚角度是以什么为基准的
        a1xy_ = np.array([np.cos(a1r), np.sin(a1r)]) * a1l + a1xy  # a1 end and a2 start (x1, y1)
        finger = np.array([np.cos(a1r + a2r), np.sin(a1r + a2r)]) * a2l + a1xy_  # a2 end (x2, y2)
        # normalize features
        dist1 = [(self.goal['x'] - a1xy_[0]) / 400, (self.goal['y'] - a1xy_[1]) / 400]
        dist2 = [(self.goal['x'] - finger[0]) / 400, (self.goal['y'] - finger[1]) / 400]
        r = -np.sqrt(dist2[0]**2+dist2[1]**2)    #这个可以表示第二根手臂的长度计算值
        
        # done and reward
        if self.goal['x'] - self.goal['l']/2 < finger[0] < self.goal['x'] + self.goal['l']/2:
            if self.goal['y'] - self.goal['l']/2 < finger[1] < self.goal['y'] + self.goal['l']/2:
                r += 1.
                self.on_goal += 1
                if self.on_goal > 50:   #达到目标地的次数50为一个回合，在目标点稳定一定时间
                    done = True
        else:
            self.on_goal = 0

        # state   状态标记是与动作相关的状态
        s = np.concatenate((a1xy_/200, finger/200, dist1 + dist2, [1. if self.on_goal else 0.]))    #状态记录
        return s, r, done    #done表示本次回合是否结束
    
    
    

    def reset(self):
        self.arm_info['r'] = 2 * np.pi * np.random.rand(2)   #随机生成两个手臂的角度
        self.on_goal = 0
        (a1l, a2l) = self.arm_info['l']  # radius, arm length
        (a1r, a2r) = self.arm_info['r']  # radian, angle
        a1xy = np.array([200., 200.])  # a1 start (x0, y0)
        a1xy_ = np.array([np.cos(a1r), np.sin(a1r)]) * a1l + a1xy  # a1 end and a2 start (x1, y1)
        finger = np.array([np.cos(a1r + a2r), np.sin(a1r + a2r)]) * a2l + a1xy_  # a2 end (x2, y2)
        # normalize features
        dist1 = [(self.goal['x'] - a1xy_[0])/400, (self.goal['y'] - a1xy_[1])/400]
        dist2 = [(self.goal['x'] - finger[0])/400, (self.goal['y'] - finger[1])/400]
        # state
        s = np.concatenate((a1xy_/200, finger/200, dist1 + dist2, [1. if self.on_goal else 0.]))
        return s    #  返回状态
        ##############################################################################
        #
        self.player.reset()
        
        
        
        

    def render(self):
        if self.viewer is None:
            self.viewer = Viewer(self.arm_info, self.player, self.goal, self.bots)
        self.viewer.render()

    def sample_action(self):
        return np.random.rand(2)-0.5    # two radians   范围在-0.5 - 0.5之间
        
        return np.random.rand(2)


class Viewer(pyglet.window.Window):
    bar_thc = 5  #手臂的宽度
    
    def __init__(self, arm_info, player, goal, bots):  #画出手臂
        # vsync=False to not use the monitor FPS, we can speed up training
        super(Viewer, self).__init__(width=400, height=400, resizable=False, caption='Arm', vsync=False)
        pyglet.gl.glClearColor(1, 1, 1, 1)  #背景颜色
        self.arm_info = arm_info
        self.center_coord = np.array([200, 200])
        
        self.batch = pyglet.graphics.Batch()    # display whole batch at once
        self.goal = self.batch.add(
                4, pyglet.gl.GL_QUADS, None,    # 4 corners
                ('v2f', [goal['x'] - 40 / 2, goal['y'] - 40 / 2,                # location
                         goal['x'] - 40 / 2, goal['y'] + 40 / 2,
                         goal['x'] + 40 / 2, goal['y'] + 40 / 2,
                         goal['x'] + 40 / 2, goal['y'] - 40 / 2]),
                ('c3B', (86, 109, 249) * 4)
            )
        self.arm1 = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [250, 250,                # location
                     250, 300,
                     260, 300,
                     260, 250]),
            ('c3B', (249, 86, 86) * 4,))    # color
        self.arm2 = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [100, 150,              # location
                     100, 160,
                     200, 160,
                     200, 150]), 
            ('c3B', (249, 86, 86) * 4,))
        
        ##################################################################################
        #  自己修改的地方
        #super(Viewer, self).__init__(width=400, height=400, resizable=False, caption='Arm', vsync=False)
        #pyglet.gl.glClearColor(1, 1, 1, 1)  #背景颜色
        self.player = player
        self.batch = pyglet.graphics.Batch()
        self.goal = self.batch.add(
                4, pyglet.gl.GL_QUADS, None,    # 4 corners
                ('v2f', [goal['x'] - 40 / 2, goal['y'] - 40 / 2,                # location
                         goal['x'] - 40 / 2, goal['y'] + 40 / 2,
                         goal['x'] + 40 / 2, goal['y'] + 40 / 2,
                         goal['x'] + 40 / 2, goal['y'] - 40 / 2]),
                ('c3B', (86, 109, 249) * 4)
            )
        self.arm1 = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [250, 250,                # location
                     250, 300,
                     260, 300,
                     260, 250]),
            ('c3B', (249, 86, 86) * 4,))    # color
        self.arm2 = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [100, 150,              # location
                     100, 160,
                     200, 160,
                     200, 150]), 
            ('c3B', (249, 86, 86) * 4,))
        
        

    def render(self):  #刷新并呈现在屏幕上
        self._update_arm()
        self.switch_to()
        self.dispatch_events()
        self.dispatch_event('on_draw')
        self.flip()

    def on_draw(self):  # 更新手臂位置，将图形全部重新绘制
        self.clear()
        self.batch.draw()

    def _update_arm(self):  #更新手臂位置信息     ============================================================================
        (a1l, a2l) = self.arm_info['l']     # radius, arm length
        (a1r, a2r) = self.arm_info['r']     # radian, angle
        a1xy = self.center_coord            # a1 start (x0, y0)
        a1xy_ = np.array([np.cos(a1r), np.sin(a1r)]) * a1l + a1xy   # a1 end and a2 start (x1, y1)
        a2xy_ = np.array([np.cos(a1r+a2r), np.sin(a1r+a2r)]) * a2l + a1xy_  # a2 end (x2, y2)
        #第一段手臂 信息
        a1tr, a2tr = np.pi / 2 - self.arm_info['r'][0], np.pi / 2 - self.arm_info['r'].sum()
        xy01 = a1xy + np.array([-np.cos(a1tr), np.sin(a1tr)]) * self.bar_thc
        xy02 = a1xy + np.array([np.cos(a1tr), -np.sin(a1tr)]) * self.bar_thc
        xy11 = a1xy_ + np.array([np.cos(a1tr), -np.sin(a1tr)]) * self.bar_thc
        xy12 = a1xy_ + np.array([-np.cos(a1tr), np.sin(a1tr)]) * self.bar_thc
        #  第二段手臂 信息
        xy11_ = a1xy_ + np.array([np.cos(a2tr), -np.sin(a2tr)]) * self.bar_thc
        xy12_ = a1xy_ + np.array([-np.cos(a2tr), np.sin(a2tr)]) * self.bar_thc
        xy21 = a2xy_ + np.array([-np.cos(a2tr), np.sin(a2tr)]) * self.bar_thc
        xy22 = a2xy_ + np.array([np.cos(a2tr), -np.sin(a2tr)]) * self.bar_thc

        self.arm1.vertices = np.concatenate((xy01, xy02, xy11, xy12))  #更新矩形的信息
        self.arm2.vertices = np.concatenate((xy11_, xy12_, xy21, xy22))
        ###########################################################################
        # 更新界面上所有障碍物，更新player的属性
        


if __name__ == '__main__':
    env = ArmEnv()
    while True:
        env.render()
        env.step(env.sample_action())




