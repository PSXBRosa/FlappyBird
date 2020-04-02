from FBirdTS2 import *
import numpy as np
import gym
import tensorflow as tf

tf.random.set_seed(1)

gym_ple.main(100)
env = gym.make('FlappyBird-v0')
n_games = 20
agent = Agent(lr = 1, gamma=.9, n_actions = 2, epsilon = 0, batch_size= 100,
                  input_dims = (8,), decay = 0.9997, epsilon_min = 0, max_size = 100000,
                  file = 'dqnmodel.h5', load = True, saving = False)
agent.load_model()
print(agent.model.summary())
scores = []
eps_history = []

for i in range(1,n_games+1):
    score = 0
    done = False
    s = env.reset()
    s = np.array(list(env.game_state.getGameState().values()))
    while not done:
        env.render(mode='human')
        action = agent.choose_action(s)
        s2, reward, done, info = env.step(action)
        score += reward
        s2 = np.array(list(env.game_state.getGameState().values()))
        agent.memory.add((s, action, reward, s2, done))
        s = s2
        
    eps_history.append(agent.epsilon)
    scores.append(score)
    
    avg_score = np.mean(scores[max(0,i-100):(i+1)])
    print(f'episode {i}, score {(score):.2f}, avarege score {(avg_score):.2f}')
env.close()
