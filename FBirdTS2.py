import tensorflow as tf
import numpy as np
import gym
import gym_ple
from collections import deque
from datetime import datetime
import os


class Memory:
    def __init__(self, maxsize):
        self.buffer = deque(maxlen = maxsize)
                
    def add(self, experience):
        self.buffer.append(experience)
        
    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        experiences_choosen = np.random.choice(range(buffer_size),batch_size,replace=False)
        return [self.buffer[exp] for exp in experiences_choosen]


def build_dqn(lr, n_actions, input_dims):
    model = tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape=input_dims),
                tf.keras.layers.Dense(512, activation='linear'),
                tf.keras.layers.Dense(32, activation='elu'),
                tf.keras.layers.Dense(n_actions)])
    
    model.compile(
        optimizer = tf.keras.optimizers.Adam(lr = lr),
        #loss = tf.keras.losses.Huber(0.5))
        loss = 'mae')
    
    return model

def reward_adapt(reward, s2):
    if s2[2] > 0:
        nr = (50 - abs(s2[0] - ((s2[4] - s2[3])/2 + s2[3])))//500
    if reward > 0:
        reward*= 200
    else:
        reward*= 10
    return reward + nr 

class Agent(object):
    def __init__(self, lr, gamma, n_actions, epsilon, batch_size,
                 input_dims, decay = 0.9996, epsilon_min=0.01,
                 max_size = 100000, file = 'dqnmodel.h5', load = False, saving = True):
        self.saving = saving
        self.action_space = [i for i in range(n_actions)]
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.decay = decay
        self.epsilon_min = epsilon_min
        self.max_size = max_size
        self.step = 1
        self.file = file
            
        self.loaded = load
        
        self.memory = Memory(self.max_size)
        
        if not load:
            self.model = build_dqn(lr, n_actions, input_dims)
            
            self.target_net = build_dqn(lr, n_actions, input_dims)
        else:
            self.model = tf.keras.models.load_model(self.file)
            self.target_net = tf.keras.models.load_model(self.file)
        
    def choose_action(self,state):
        self.epsilon *= self.decay
        state = state[np.newaxis, :]
        rand = np.random.random()
        if rand < self.epsilon:
            weighted_actions = [1]*9 + [0]
            action = np.random.choice(weighted_actions)
        else:
            actions = self.model.predict(state)
            action = np.argmax(actions)
        return action
    
    def learn(self):
        if len(self.memory.buffer) < self.batch_size:
            return
        batch = self.memory.sample(self.batch_size)
        states = np.array([i[0] for i in batch])
        actions = np.array([i[1] for i in batch])
        rewards = np.array([i[2] for i in batch])
        next_states = np.array([i[3] for i in batch])
        dones = np.array([i[4] for i in batch])
        
        q_eval = self.model.predict(states)
        q_next = self.target_net.predict(next_states)
        
        q_target = q_eval.copy()
        
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        
        q_target[batch_index, actions] = rewards + self.gamma*np.max(q_next,axis=1)*dones
        
        _ = self.model.fit(states, q_target, verbose = 0)

        
    def copy_weights(self):
        self.target_net.set_weights(self.model.get_weights()) 
    
    def save_model(self, file = 0):
        if file == 0:
            file = self.file
        self.model.save(file)
        
    def load_model(self):
        self.model = tf.keras.models.load_model(self.file)

def main():
    gym_ple.main(100)
    tf.random.set_seed(1)
    env = gym.make('FlappyBird-v0')
    n_games = 300
    agent = Agent(lr = 0.00001, gamma=.95, n_actions = 2, epsilon = 0.2, batch_size= 100,
                  input_dims = (8,), decay = 0.999, epsilon_min = 0.01, max_size = 50000, load = True)

    scores = []
    eps_history = []
    max_score = 0
    
    for i in range(1,n_games+1):
        score = 0
        done = False
        s = env.reset()
        s = np.array(list(env.game_state.getGameState().values()))
        step = 0
        actions = []
        while not done and step < 5000:
            #env.render(mode='human')
            agent.step+=1
            action = agent.choose_action(s)
            actions.append(action)
            s2, reward, done, info = env.step(action)
            s2 = np.array(list(env.game_state.getGameState().values()))
            reward = reward_adapt(reward,s2)
            score += reward
            agent.memory.add((s, action, reward, s2, done))
            s = s2
            step += 1
            agent.learn()
        
        agent.copy_weights()
        eps_history.append(agent.epsilon)
        scores.append(score)
        
        avg_score = np.mean(scores[max(0,i-100):(i+1)])
        descidas = np.count_nonzero(actions)
        subidas = len(actions) - descidas
        if subidas > 0:
            razao = descidas/subidas
        else:
            razao = 999.999
        print(f'episode {i}, score {(score):.2f}, avarege score {(avg_score):.2f}, epsilon {(agent.epsilon):.4f}, descidas/subidas {(razao):.2f}')
        if i % 10 == 0 and i > 0 and agent.saving:
            agent.save_model()
            print(f'>>> model saved at {agent.file}')
        # if avg_score >= max_score and i >= 50:
        #     print('>>> max score updated')
        #     max_score = avg_score
        #     agent.save_model('maxscore.h5')
            
    # file = 'cartpole.png'
    # x = [i+1 for i in range(n_games)]
    # plotLearning(x,scores,eps_history,file)
    
if __name__== '__main__':
    main()
    
        
        
    
    
        
