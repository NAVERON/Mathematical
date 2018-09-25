



from MyDemo.My.Bot import Bot
import random
from math import sin, cos, atan2, pi, hypot
import sys
import wx


import tensorflow as tf
import cv2
import numpy as np
from collections import deque
# import My.game_core


ACTIONS = 2  # number of valid actions  动作数量
GAMMA = 0.99  # decay rate of past observations
OBSERVE = 100.  # timesteps to observe before training   100000.
EXPLORE = 200.  # frames over which to anneal epsilon   2000000.

FINAL_EPSILON = 0.0001  # final value of epsilon
INITIAL_EPSILON = 0.0001  # starting value of epsilon

REPLAY_MEMORY = 500  # number of previous transitions to remember
BATCH = 32  # size of minibatch
FRAME_PER_ACTION = 1

#   deep Q learning
class Model(object):  # 全局计算
    
    def __init__(self, width, height):
        self.width = width
        self.height = height
        
        self.bots = self.create_bots(20)
        self.player = Bot(self.width, self.height, self.select_point(), random.random() * 5 + 0.1, random.random() * 2 * pi)  # 训练这个对象
        
        self.times = 0  # 学习次数
        self.score = 0  # 游戏分数
        
        self.sess = tf.InteractiveSession()
        self.s, self.readout, self.h_fc1 = self.createNetwork()
        # self.trainNetwork(self.s, self.readout, self.h_fc1, self.sess)
        
        self.a = tf.placeholder("float", [None, ACTIONS])
        self.y = tf.placeholder("float", [None])
        self.readout_action = tf.reduce_sum(tf.multiply(self.readout, self.a), reduction_indices=1)
        self.cost = tf.reduce_mean(tf.square(self.y - self.readout_action))
        self.train_step = tf.train.AdamOptimizer(1e-6).minimize(self.cost)
        # open up a game state to communicate with emulator
        
        # store the previous observations in replay memory
        self.D = deque()
        # get the first state by doing nothing and preprocess the image to 80x80x4
        self.do_nothing = np.zeros(ACTIONS)   # [0, 0]
        self.do_nothing[0] = 1 # [1, 0]
        self.r_0, self.terminal = self.frame_step(self.do_nothing)
        
#         self.x_t, self.r_0, self.terminal = self.frame_step(self.do_nothing)  # frame_step 传入 一个列表
#         self.x_t = cv2.cvtColor(cv2.resize(self.x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
#         self.ret, self.x_t = cv2.threshold(self.x_t, 1, 255, cv2.THRESH_BINARY)
#         self.s_t = np.stack((self.x_t, self.x_t, self.x_t, self.x_t), axis=2)
        # saving and loading networks
        self.saver = tf.train.Saver()
        self.sess.run(tf.initialize_all_variables())
        
        self.checkpoint = tf.train.get_checkpoint_state("saved_networks")
        if self.checkpoint and self.checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, self.checkpoint.model_checkpoint_path)
            print("Successfully loaded:", self.checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")
        
        # start training
        self.epsilon = INITIAL_EPSILON
        
    
    def reset(self, width, height):
        self.width = width
        self.height = height
        
        self.bots.clear()
        self.bots = self.create_bots(20)
        self.player = Bot(self.width, self.height, self.select_point(), random.random() * 5 + 0.1, random.random() * 2 * pi)
    
    def create_bots(self, count):  # 随机创建一些bot对象
        result = []
        for i in range(count):
            position = self.select_point()
            bot = Bot(self.width, self.height, position, random.random() * 5 + 0.1, random.random() * 2 * pi)
            result.append(bot)
        return result
    
    def select_point(self):  # 在圆内生成随机点
        cx = self.width / 2.0
        cy = self.height / 2.0
        radius = min(self.width, self.height) * 0.4  # 半径
        angle = random.random() * 2 * pi
        # 在圆形内部生成随机点
        x = cx + cos(angle) * radius
        y = cy + sin(angle) * radius
        return (x, y)
    
    def update(self, dt):  # 更新，这里以后可以放学习模块
        px, py = self.player.position
        
        for bot in self.bots:
            bot.update(self.bots)
            bx, by = bot.position
            if self.is_collision(self.player, bot):
                print("第  %d 次学习" % self.times)  # 惩罚
                self.times += 1
                break
            if hypot(px - bx, py - by) < 100:
                self.player.direction += 0.2 * pi
        
        #
        # 从train方法里面有 训练方法，复制过来的 
        # readout_t = self.readout.eval(feed_dict={self.s : [self.s_t]})[0]
        self.a_t = np.zeros([ACTIONS])
        self.action_index = 0  #  动作索引
        if self.times % FRAME_PER_ACTION == 0:
            if random.random() <= self.epsilon:
                print("----------Random Action----------")
                action_index = random.randrange(ACTIONS)
                self.a_t[random.randrange(ACTIONS)] = 1
            else:
                action_index = 1
                self.a_t[action_index] = 1
        else:
            self.a_t[0] = 1  # do nothing
        
        # scale down epsilon
        if self.epsilon > FINAL_EPSILON and self.times > OBSERVE:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # run the selected action and observe next state and reward
        self.r_t, self.terminal = self.frame_step(self.a_t)
        
        # self.x_t1_colored, self.r_t, self.terminal = self.frame_step(self.a_t)  #  r_t表示奖励  a_t代表动作
#         self.x_t1 = cv2.cvtColor(cv2.resize(self.x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
#         self.ret, self.x_t1 = cv2.threshold(self.x_t1, 1, 255, cv2.THRESH_BINARY)
#         self.x_t1 = np.reshape(self.x_t1, (80, 80, 1))
#         # s_t1 = np.append(x_t1, s_t[:,:,1:], axis = 2)
#         self.s_t1 = np.append(self.x_t1, self.s_t[:, :, :3], axis=2)
        
        # store the transition in D
        self.D.append((self.player.position, self.a_t, self.player.position, self.terminal))
        if len(self.D) > REPLAY_MEMORY:
            self.D.popleft()
            
        # only train if done observing
        if self.times > OBSERVE:
            # sample a minibatch to train on
            self.minibatch = random.sample(self.D, BATCH)
            
            # get the batch variables
            self.s_j_batch = [d[0] for d in self.minibatch]
            self.a_batch = [d[1] for d in self.minibatch]
            self.r_batch = [d[2] for d in self.minibatch]
            self.s_j1_batch = [d[3] for d in self.minibatch]
            
            self.y_batch = []
            self.readout_j1_batch = self.readout.eval(feed_dict={self.s : self.s_j1_batch})
            for i in range(0, len(self.minibatch)):
                self.terminal = self.minibatch[i][4]
                # if terminal, only equals reward
                if self.terminal:
                    self.y_batch.append(self.r_batch[i])
                else:
                    self.y_batch.append(self.r_batch[i] + GAMMA * np.max(self.readout_j1_batch[i]))
            
            # perform gradient step
            self.train_step.run(feed_dict={
                self.y : self.y_batch,
                self.a : self.a_batch,
                self.s : self.s_j_batch}
            )
        
        # update the old values
        self.s_t = self.player.update(self.bots)
        self.times += 1
        
        # save progress every 10000 iterations
        if self.times % 10000 == 0:
            self.saver.save(self.sess, 'saved_networks/' + "new" + '-dqn', global_step=self.times)
            
        # print info
        state = ""
        if self.times <= OBSERVE:
            state = "observe"
        elif self.times > OBSERVE and self.times <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"
            
        print("TIMESTEP", self.times, "/ STATE", state, \
            "/ EPSILON", self.epsilon, "/ ACTION", action_index, "/ REWARD", self.r_t, \
            "/ Q_MAX %e" % np.max([1.23, 1.5]))  # 这里更改部分
        # write info to files
  

    def is_collision(self, a, b):
        ax, ay = a.position
        bx, by = b.position
        if hypot(ax - bx, ay - by) < 10:
            return True
        else:
            return False
    
    def Get_Screen_Bmp(self):
        s = wx.GetDisplaySize()
        bmp = wx.EmptyBitmap(s.x, s.y)
        dc = wx.ScreenDC()
        memdc = wx.MemoryDC()
        memdc.SelectObject(bmp)
        memdc.Blit(0, 0, s.x, s.y, dc, 0, 0)
        memdc.SelectObject(wx.NullBitmap)
        return bmp
    
    #############################################################################
    ############################################################################
    
    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial)
    
    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)
    
    def conv2d(self, x, W, stride):
        return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")
    
    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    
    def createNetwork(self):
        # network weights
        W_conv1 = self.weight_variable([8, 8, 4, 32])
        b_conv1 = self.bias_variable([32])
    
        W_conv2 = self.weight_variable([4, 4, 32, 64])
        b_conv2 = self.bias_variable([64])
    
        W_conv3 = self.weight_variable([3, 3, 64, 64])
        b_conv3 = self.bias_variable([64])
    
        W_fc1 = self.weight_variable([1600, 512])
        b_fc1 = self.bias_variable([512])
    
        W_fc2 = self.weight_variable([512, ACTIONS])
        b_fc2 = self.bias_variable([ACTIONS])
    
        # input layer  输入层
        s = tf.placeholder("float", [None, 80, 80, 4])
    
        # hidden layers
        h_conv1 = tf.nn.relu(self.conv2d(s, W_conv1, 4) + b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)
    
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2, 2) + b_conv2)
        # h_pool2 = max_pool_2x2(h_conv2)
    
        h_conv3 = tf.nn.relu(self.conv2d(h_conv2, W_conv3, 1) + b_conv3)
        # h_pool3 = max_pool_2x2(h_conv3)
    
        # h_pool3_flat = tf.reshape(h_pool3, [-1, 256])
        h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])
    
        h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)
    
        # readout layer  输出层
        readout = tf.matmul(h_fc1, W_fc2) + b_fc2
        return s, readout, h_fc1
    
    def trainNetwork(self, s, readout, h_fc1, sess):
        # define the cost function
        a = tf.placeholder("float", [None, ACTIONS])
        y = tf.placeholder("float", [None])
        readout_action = tf.reduce_sum(tf.multiply(readout, a), reduction_indices=1)
        cost = tf.reduce_mean(tf.square(y - readout_action))
        train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)
        
        # open up a game state to communicate with emulator
        game_state = self.gameStste()
        # game_state = My.game_core().GameState()
        
        # store the previous observations in replay memory
        D = deque()
        
        # printing
        '''
        a_file = open("logs_" + GAME + "/readout.txt", 'w')
        h_file = open("logs_" + GAME + "/hidden.txt", 'w')
        '''
        
        # get the first state by doing nothing and preprocess the image to 80x80x4
        do_nothing = np.zeros(ACTIONS)
        do_nothing[0] = 1
        x_t, r_0, terminal = game_state.frame_step(do_nothing)
        x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
        s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
        
        # saving and loading networks
        saver = tf.train.Saver()
        sess.run(tf.initialize_all_variables())
        checkpoint = tf.train.get_checkpoint_state("saved_networks")
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")
            
        # start training
        epsilon = INITIAL_EPSILON
        t = 0
        while "flappy bird" != "angry bird":
            # choose an action epsilon greedily
            readout_t = readout.eval(feed_dict={s : [s_t]})[0]
            a_t = np.zeros([ACTIONS])
            action_index = 0  #  动作索引
            if t % FRAME_PER_ACTION == 0:
                if random.random() <= epsilon:
                    print("----------Random Action----------")
                    action_index = random.randrange(ACTIONS)
                    a_t[random.randrange(ACTIONS)] = 1
                else:
                    action_index = np.argmax(readout_t)
                    a_t[action_index] = 1
            else:
                a_t[0] = 1  # do nothing
    
            # scale down epsilon
            if epsilon > FINAL_EPSILON and t > OBSERVE:
                epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
    
            # run the selected action and observe next state and reward
            x_t1_colored, r_t, terminal = game_state.frame_step(a_t)  #  r_t表示奖励
            x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
            ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
            x_t1 = np.reshape(x_t1, (80, 80, 1))
            # s_t1 = np.append(x_t1, s_t[:,:,1:], axis = 2)
            s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)
            
            # store the transition in D
            D.append((s_t, a_t, r_t, s_t1, terminal))
            if len(D) > REPLAY_MEMORY:
                D.popleft()
    
            # only train if done observing
            if t > OBSERVE:
                # sample a minibatch to train on
                minibatch = random.sample(D, BATCH)
    
                # get the batch variables
                s_j_batch = [d[0] for d in minibatch]
                a_batch = [d[1] for d in minibatch]
                r_batch = [d[2] for d in minibatch]
                s_j1_batch = [d[3] for d in minibatch]
    
                y_batch = []
                readout_j1_batch = readout.eval(feed_dict={s : s_j1_batch})
                for i in range(0, len(minibatch)):
                    terminal = minibatch[i][4]
                    # if terminal, only equals reward
                    if terminal:
                        y_batch.append(r_batch[i])
                    else:
                        y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))
    
                # perform gradient step
                train_step.run(feed_dict={
                    y : y_batch,
                    a : a_batch,
                    s : s_j_batch}
                )
    
            # update the old values
            s_t = s_t1
            t += 1
    
            # save progress every 10000 iterations
            if t % 10000 == 0:
                saver.save(sess, 'saved_networks/' + "new" + '-dqn', global_step=t)
    
            # print info
            state = ""
            if t <= OBSERVE:
                state = "observe"
            elif t > OBSERVE and t <= OBSERVE + EXPLORE:
                state = "explore"
            else:
                state = "train"
    
            print("TIMESTEP", t, "/ STATE", state, \
                "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, \
                "/ Q_MAX %e" % np.max(readout_t))
            # write info to files
            '''
            if t % 10000 <= 100:
                a_file.write(",".join([str(x) for x in readout_t]) + '\n')
                h_file.write(",".join([str(x) for x in h_fc1.eval(feed_dict={s:[s_t]})[0]]) + '\n')
                cv2.imwrite("logs_tetris/frame" + str(t) + ".png", x_t1)
            '''
            
    def playGame(self):
        sess = tf.InteractiveSession()
        s, readout, h_fc1 = self.createNetwork()  
        self.trainNetwork(s, readout, h_fc1, sess)
        
    def frame_step(self, input_actions):
        reward = 0.1
        terminal = False
        
        if sum(input_actions) != 1:
            raise ValueError('Multiple input actions!')
        # input_actions[0] == 1: do nothing
        # input_actions[1] == 1: flap the bird
        if input_actions[1] == 1:
            px, py = self.player.position
            tx, ty = self.player.target
            angle = atan2(ty - py, tx - px)  #这里写避碰算法
            self.player.direction = angle
            reward = 0.0001
        
        return reward, terminal
##############################################################################










