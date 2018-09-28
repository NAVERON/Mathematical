

"""
Make it more robust.
Stop episode once the finger stop at the final position for 50 steps.
Feature & reward engineering.
"""


from MoFan.Demo.Part5.env import ArmEnv
from MoFan.Demo.Part5.rl import DDPG

MAX_EPISODES = 500
MAX_EP_STEPS = 1000
ON_TRAIN = True

# set env
env = ArmEnv()
s_dim = env.state_dim
a_dim = env.action_dim
a_bound = env.action_bound   #动作边界

# set RL method (continuous)
rl = DDPG(a_dim, s_dim, a_bound)

steps = []
def train():
    # start training
    for i in range(MAX_EPISODES):
        s = env.reset()    #从环境获取初始的状态量等
        ep_r = 0.
        for j in range(MAX_EP_STEPS):    # 单局步数
            env.render()
            a = rl.choose_action(s)  #根据当前状态选择一个动作

            s_, r, done = env.step(a)  #根据动作更新环境，更新界面的图形    返回更新后的    状态和奖励值，done代表是否到达了目标点，完成任务
            # print(a[0], " , ", a[1])  # [-0.07847165 -0.33180785]
            rl.store_transition(s, a, r, s_)

            ep_r += r
            if rl.memory_full:
                # start to learn once has fulfilled the memory
                rl.learn()

            s = s_
            if done or j == MAX_EP_STEPS-1:
                print('Ep: %i | %s | ep_r: %.1f | step: %i' % (i, '---' if not done else 'done', ep_r, j))
                # print(env.arm_info)  # [(100., 2.6605449) (100., 4.790004 )]
                break
    rl.save()


def eval():  #读取图   并绘制图形
    rl.restore()
    env.render()
    env.viewer.set_vsync(True)
    while True:
        s = env.reset()
        for _ in range(200):
            env.render()
            a = rl.choose_action(s)
            s, r, done = env.step(a)
            if done:
                break


if ON_TRAIN:
    train()
else:
    eval()




















