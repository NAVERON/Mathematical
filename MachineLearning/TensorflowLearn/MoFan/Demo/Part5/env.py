



import numpy as np
import pyglet
import random
from math import sin, cos, atan2, pi, hypot
from MoFan.Demo.Part5.Bot import Bot
from MoFan.Demo.Part5.Player import Player


class ArmEnv(object):
    ##########################################################
    viewer = None
    dt = .1
    action_bound = [-2 * pi, 2 * pi]
    state_dim = 9  #状态维度  观测值
    action_dim = 2  #动作维度     ------- 例子中是两个手臂的角度，这里是速度大小和方向
    
    def __init__(self):  #初始化基础数据
        #########################################################################
        #  初始化界面避碰信息
        self.player = Player((random.random() * 400, random.random() * 400), random.random() * 5 + 1, random.random() * 2 * pi)
        self.bots = []
        for _ in range(random.randint(0, 20)):
            position = (random.random() * 200, random.random() * 200)
            bot = Bot(position, random.random() * 5 + 1, random.random() * 2 * pi)
            self.bots.append(bot)
        
        self.on_goal = 0
        

    def step(self, action):    #action是一个二维数组，表示手臂的旋转角度 ===================================================
        ################################################################################
        done = False
        action = np.clip(action, *self.action_bound)
        self.player.update(action, self.dt)   # 根据动作更新自身的属性
        dis = self.player.dis_with_target()
        r = -dis
        
        for bot in self.bots:
            pass
        
        px, py = self.player.position
        tx, ty = self.player.target
        if hypot(tx - px, ty - py) < 10:  #如果达到目的地，重新设置目标和速度，这里的判断还需要判断
            r += 1
            self.on_goal += 1
            if self.on_goal > 50:
                done = True
                self.player.target = (random.random() * 400, random.random() * 400)
        else:
            self.on_goal = 0
        
        dis = self.player.dis_with_target()
        near = []           # 这个引用还没哟用到
        near_count = 0
        for bot in self.bots:
            tx, ty = bot.position
            if hypot(px - tx, py - ty) < 100:   # 如果附近有其他障碍物，则添加进去
                near_count += 1
                near.append(bot)
        # state   状态标记是与动作相关的状态
        s = np.concatenate(           #  记录是9个记录
            (self.player.position,
             self.player.target,
             self.player.speed,
             self.player.direction,
             dis,
             near_count,
             [1. if self.on_goal else 0.]
            )
        )    #状态记录
        return s, r, done    #done表示本次回合是否结束

    def reset(self):
        ##############################################################################
        self.player.reset()
        self.bots.clear()
        for _ in range(random.randint(0, 20)):
            position = (random.random() * 200, random.random() * 200)
            bot = Bot(position, random.random() * 5 + 1, random.random() * 2 * pi)
            self.bots.append(bot)
        self.on_goal = 0
        dis = self.player.dis_with_target()
        
        px, py = self.player.position
        near = []           # 这个引用还没哟用到
        near_count = 0
        for bot in self.bots:
            tx, ty = bot.position
            if hypot(px - tx, py - ty) < 100:   # 如果附近有其他障碍物，则添加进去
                near_count += 1
                near.append(bot)
        
        # 计算当前状态
        s = np.concatenate(            # 9 demintion
            (self.player.position,
             self.player.target,
             self.player.speed,
             self.player.direction,
             dis,
             near_count,
             [1. if self.on_goal else 0.]
             )
        )
        return s    #  返回状态
        
    def render(self):
        if self.viewer is None:
            self.viewer = Viewer(self.player, self.bots)
        self.viewer.render()

    def sample_action(self):
        return np.random.rand(2) * 2 + 0.1


class Viewer(pyglet.window.Window):
    bar_thc = 5  #手臂的宽度
    
    def __init__(self, player, bots):  #画出手臂
        # vsync=False to not use the monitor FPS, we can speed up training
        ##################################################################################
        super(Viewer, self).__init__(width=400, height=400, resizable=False, caption='Arm', vsync=False)
        pyglet.gl.glClearColor(1, 1, 1, 1)  #背景颜色
        self.player = player
        self.bots = bots
        
        self.batch = pyglet.graphics.Batch()
        px, py = self.player.position
        self.player_draw = self.batch.add(
                4, pyglet.gl.GL_QUADS, None,    # 4 corners
                ('v2f', [ px - 10, py - 10,                # location
                         px - 10, py + 10,
                         px + 10, py + 10,
                         px + 10, py - 10
                         ]
                ),
                ('c3B', (86, 109, 249) * 4)
            )
        tx, ty = self.player.target
        self.goal = self.batch.add(
                4, pyglet.gl.GL_QUADS, None,    # 4 corners
                ('v2f', [tx - 10, ty - 10,                # location
                         tx - 10, ty + 10,
                         tx + 10, ty + 10,
                         tx + 10, ty - 10
                         ]
                ),
                ('c3B', (86, 109, 249) * 4)
            )
        for bot in self.bots:
            bx, by = bot.position
            self.batch.add(
                    4, pyglet.gl.GL_QUADS, None,
                    ('v2f', [bx - 10, by - 10,  # location
                         bx - 10, by + 10,
                         bx + 10, by + 10,
                         bx - 10, by - 10]
                    ),
                    ('c3B', (249, 86, 86) * 4)
                )
        

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
        #更新player本身，并且重新放入batch进行更新绘制
        px, py = self.player.position
        near = []           # 这个引用还没哟用到
        near_count = 0
        for bot in self.bots:
            tx, ty = bot.position
            if hypot(px - tx, py - ty) < 100:   # 如果附近有其他障碍物，则添加进去
                near_count += 1
                near.append(bot)
        
        self.player_draw.vertices = np.concatenate()
        

if __name__ == '__main__':
    env = ArmEnv()
    while True:
        env.render()
        env.step(env.sample_action())




