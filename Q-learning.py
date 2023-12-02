import copy
import numpy as np
import pandas as pd

np.random.seed(2)  # reproducible
row = 5
col = 6
ACTIONS = ['up', 'right', 'down', 'left']  # available actions
EPSILON = 0.9  # greedy police
ALPHA = 0.1  # learning rate
GAMMA = 0.9  # discount factor
MAX_EPISODES = 5000  # maximum episodes
FRESH_TIME = 0.3  # fresh time for one move
targetXY = [4, 4]
env_list = ['--+---', '-*--*-', '--+--+', '*--*--', '-+--T-']
repeatRes = 0
minStep = 1e9


def build_q_table(row, col, actions):
    table = pd.DataFrame(
        np.zeros((row * col, len(actions))),  # q_table initial values
        columns=actions,  # actions' name
    )
    # print(table)    # show table
    return table


def getR(S):
    str = env_list[S[0]][S[1]]
    if str == '-':
        return -1
    elif str == '*':
        return -100
    elif str == '+':
        return 1
    else:
        return 100


def choose_action(state, q_table):
    # This is how to choose an action
    state_actions = q_table.iloc[state[0] * col + state[1], :]
    if (np.random.uniform() > EPSILON) or ((state_actions == 0).all()):  # act non-greedy or state-action have no value
        action_name = np.random.choice(ACTIONS)
    else:  # act greedy
        action_name = state_actions.idxmax()
        # replace argmax to idxmax as argmax means a different function in newer version of pandas
    return action_name


def get_env_feedback(S, A):
    # This is how agent will interact with the environment
    if A == 'up':  # move up
        if S[0] == targetXY[0]+1 and S[1] == targetXY[1]:  # 到达终点
            S_ = 'terminal'
            R = 100
        elif S[0] == 0:  # 向上碰壁
            S_ = S
            R = -1
        else:  # 正常移动
            S_ = [S[0] - 1, S[1]]
            R = getR(S_)
            if R == -100:  # 碰到炸弹直接结束
                S_ = 'terminal'
    elif A == 'right':  # move right
        if S[0] == targetXY[0] and S[1] == targetXY[1] - 1:  # 到达终点
            S_ = 'terminal'
            R = 100
        elif S[1] == col - 1:  # 向右碰壁
            S_ = S
            R = -1
        else:  # 正常移动
            S_ = [S[0], S[1] + 1]
            R = getR(S_)
            if R == -100:  # 碰到炸弹直接结束
                S_ = 'terminal'
    elif A == 'down':  # move down
        if S[0] == row - 1:  # 向下碰壁
            S_ = S
            R = -1
        elif S[0] == targetXY[0] - 1 and S[1] == targetXY[1]:  # 到达终点
            S_ = 'terminal'
            R = 100
        else:  # 正常移动
            S_ = [S[0] + 1, S[1]]
            R = getR(S_)
            if R == -100:  # 碰到炸弹直接结束
                S_ = 'terminal'
    else:  # move left
        if S[0] == targetXY[0] and S[1] == targetXY[1] + 1:  # 到达终点
            S_ = 'terminal'
            R = 100
        elif S[1] == 0:  # 向左碰壁
            S_ = S
            R = -1
        else:  # 正常移动
            S_ = [S[0], S[1] - 1]
            R = getR(S_)
            if R == -100:  # 碰到炸弹直接结束
                S_ = 'terminal'
    return S_, R


def update_env(S, episode, step_counter, is_terminated):
    # This is how environment be updated
    if is_terminated == True:
        interaction = 'Episode %s: total_steps = %s' % (episode + 1, step_counter)
        print(interaction)
    else:
        tmp_env_list = copy.deepcopy(env_list)  # 深拷贝
        temp = list(tmp_env_list[S[0]])
        temp[S[1]] = 'o'
        tmp_env_list[S[0]] = ''.join(temp)
        for i in range(row):
            print(tmp_env_list[i])
        print("***************************")


def rl():
    # main part of RL loop
    q_table = build_q_table(row, col, ACTIONS)  # 创建Q表
    for episode in range(MAX_EPISODES):
        step_counter = 0
        S = [0, 0]  # 状态信息
        is_terminated = False  # 结束标志
        update_env(S, episode, step_counter, is_terminated)  # 当前状态可视化
        while not is_terminated:
            A = choose_action(S, q_table)
            S_, R = get_env_feedback(S, A)  # take action & get next state and reward
            q_predict = q_table.loc[S[0] * col + S[1], A]
            if S_ != 'terminal':
                q_target = R + GAMMA * q_table.iloc[S_[0] * col + S_[1], :].max()  # next state is not terminal
            else:
                q_target = R  # next state is terminal
                is_terminated = True  # terminate this episode

            q_table.loc[S[0] * col + S[1], A] = (1 - ALPHA) * q_predict + ALPHA * q_target  # update
            S = S_  # move to next state

            update_env(S, episode, step_counter + 1, is_terminated)
            step_counter += 1
    return q_table


if __name__ == "__main__":
    q_table = rl()
    print('\nQ-table:\n')
    print(q_table)
