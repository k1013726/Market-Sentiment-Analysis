import gym
import numpy as np

# 创建 FrozenLake 环境
env = gym.make('FrozenLake-v1')

# 初始化 Q-table，全部设为0
num_states = env.observation_space.n
num_actions = env.action_space.n
q_table = np.zeros((num_states, num_actions))

# 设置 Q-learning 参数
num_episodes = 10000
max_steps_per_episode = 100
learning_rate = 0.1
discount_rate = 0.99

# Q-learning 算法
for episode in range(num_episodes):
    state = env.reset()
    done = False
    
    for step in range(max_steps_per_episode):
        # 在 Q-table 中选择动作（ε-贪心策略）
        exploration_rate = np.random.uniform(0, 1)
        if exploration_rate > 0.5:  # 探索
            action = env.action_space.sample()
        else:  # 利用
            action = np.argmax(q_table[state])
        
        # 执行动作并观察下一个状态和奖励
        new_state, reward, done,info, _ = env.step(action)
        
        # 更新 Q-table
        q_table[state, action] = q_table[state, action] + learning_rate * (
            reward + discount_rate * np.max(q_table[new_state]) - q_table[state, action]
        )
        
        state = new_state
        
        if done:
            break

# 在环境中运行一个 episode，使用学习到的 Q-table
state = env.reset()
done = False
while not done:
    action = np.argmax(q_table[state])
    state, reward, done, _ = env.step(action)
    env.render()

env.close()
