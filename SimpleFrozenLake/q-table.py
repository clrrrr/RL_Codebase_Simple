import gym
import numpy as np 

# Load the environment
env = gym.make('FrozenLake-v0')

# Implement Q-table learning algorithm

# Initialize table with all zeros
Q = np.zeros([env.observation_space.n, env.action_space.n])
# Set learning parameters
lr = .8
gamma = .95
num_episodes = 2000
# Create lists to contain total rewards and steps per episode
# jList = []
rList = []
for i in range(num_episodes):
    # Reset environment and get first new observation
    s = env.reset()
    rAll = 0
    d = False
    j = 0
    # The Q-Table learning algorithm
    while j < 99: #这也是个重要的要调的超参，trajectory的最大长度
        j += 1
        # Choose an action by greedily(with noise) picking from Q table
        a = np.argmax(Q[s, :] + np.random.randn(1, env.action_space.n)*(1./(i+1))) #argmax就是取下标，所以a现在就是个整数
        #不加noise的话，它就可能会一直往一个地方走了
        #这个noise的大小感觉蛮重要的，要调的
        #啊别忘了，q-learning里面是没有policy这个咚咚的

        # Get new state and reward from environment
        sNew, r, d, _ = env.step(a) #我都怀疑s是不是也就是个整数而已
        # Update Q-Table with new knowledge
        Q[s, a] = Q[s, a] + lr*(r + gamma*np.max(Q[sNew, :]) - Q[s, a])
        #emmmmmm所以是每走一步就回头更新前一个点的q-value呢，这样做好在哪里呢？我提出的方法是更好还是更差了？
        rAll += r
        s = sNew
        if d == True: #所以d是判断有没有到达终点的，到达了可以提前掐掉
            break
        #jList.append(j)
        rList.append(rAll)

        #现在我有两个问题没搞懂：
        #掉到洞里了为什么不break，难道还能继续？
        #难道每个状态都能take所有action吗？比如在最左了还能向左走？代码里没有看出不合法。怎么handle？

print("Score over time: " + str(sum(rList)/num_episodes))
print("Final Q-Table Values")
print(Q)
