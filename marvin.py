#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
import numpy as np
import math
import gym
import random
import copy
import pickle

parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group()
parser.add_argument("-w", "--walk", help="Display only walking process",
                    action="store_true")
group.add_argument("-l", "--load", help="Load weights for Marvin agent from a file",
                    action="store")
group.add_argument("-s", "--save", help="Save weights to a file after running the program",
                    action="store")
parser.add_argument("--b", help ="Set limit batches", nargs='?',const=50, type=int)
args = parser.parse_args()
if args.walk:
    print("walk turned on")
if args.load is not None:
    print("load turned on")
if args.save is not None:
    print("save turned on")
args = parser.parse_args()


# In[ ]:


train_batch = 200
first_x_elements = 20
mutation_n = 0.5
cross_over_porcentage = 0.4
mutation_porcentage = 0.90

def run2(NN):
    env = gym.make("Marvin-v0")
    observation = env.reset()
    r_sum = 0
    for t in range(3000):
        env.render()
        observation = observation.reshape((24,1))
        action = NN._feedforward(observation).flatten()
        observation, reward, done, info = env.step(action)
        r_sum = r_sum + reward
        if reward < -90:
            break;
        if r_sum < -10:
            break
    env.close()
    env.render(close=True)
    return r_sum

def save_file(ganadora,name):
    file = [ganadora.weights_array,ganadora.bias_array]
    with open(name , "wb") as fp:
        pickle.dump(file,fp)
        
def open_file(name):
    with open(name, "rb") as fp:
        b = pickle.load(fp)
    return b

def run3(NN):
    env = gym.make("Marvin-v0")
    observation = env.reset()
    r_sum = 0
    for t in range(2000):
        observation = observation.reshape((24,1))
        action = NN._feedforward(observation).flatten()
        observation, reward, done, info = env.step(action)
        if reward < -90:
            break;
        r_sum = r_sum + reward
        if r_sum < -10:
            break
    env.close()
    return r_sum

def run2(NN):
    env = gym.make("Marvin-v0")
    observation = env.reset()
    r_sum = 0
    for t in range(3000):
        env.render()
        observation = observation.reshape((24,1))
        action = NN._feedforward(observation).flatten()
        observation, reward, done, info = env.step(action)
        r_sum = r_sum + reward
        if reward < -90:
            break;
        if r_sum < -10:
            break
    env.close()
    env.render(close=True)
    return r_sum

def test_neuron(nn):
    sum_reward = 0
    for x in range(100):
        s = run3(nn)
        sum_reward = s + sum_reward
    return sum_reward / 100

def mutation(weights):
    r_n = random.random()
    if r_n <= mutation_n:
        for index, value in enumerate(weights):
            weights[index] = np.dot(weights[index],mutation_porcentage)
    return weights

def replicate(weights1 , weights2):
    for index, value in enumerate(weights1):
        weights1[index] = (weights1[index] + weights2[index]) / 2
    return weights1

def cross_over(NN1, NN2):
    NNCOPY = copy.deepcopy(NN1)
    weights_nn1 = NNCOPY.weights_array
    bias_nn1 = NNCOPY.bias_array 
    weights_nn2 = NN2.weights_array
    bias_nn2 = NN2.bias_array
    for index, nn1 in enumerate(weights_nn1):
        for i , n1 in enumerate(nn1):
            r =random.random()
            if r <= cross_over_porcentage:
                weights_nn1[index][i] = mutation(np.copy(weights_nn2[index][i]))
                bias_nn1[index][i] = mutation(np.copy(bias_nn2[index][i]))
    return NNCOPY

def Average(lst): 
    return sum(lst) / len(lst)

def cross_over2(NN1, NN2):
    NNCOPY = copy.deepcopy(NN1)
    weights_nn1 = NNCOPY.weights_array
    bias_nn1 = NNCOPY.bias_array 
    weights_nn2 = NN2.weights_array
    bias_nn2 = NN2.bias_array
    for index, nn1 in enumerate(weights_nn1):
        for i , n1 in enumerate(nn1):
            r = random.random()
            if r <= cross_over_porcentage:
                weights_nn1[index][i] = replicate(weights_nn1[index][i] ,weights_nn2[index][i] )
                bias_nn1[index][i] = replicate(bias_nn1[index][i],bias_nn2[index][i] )
    return NNCOPY


# In[ ]:


class NeuralNetwork:
    def __init__(self, numI, *argv):
        self.numI = numI
        self.weights_array= []
        self.bias_array = []
        self.number = 0
        for arg in argv:
            self.number = self.number + 1
            self.weights_array.append(np.random.rand(arg,self.numI) * 2 - 1)
            self.bias_array.append(np.random.rand(arg,1) * 2 - 1)
            self.numI = arg
    def _tanh(self, x):
        s =  np.tanh(x)
        return s
    def _feedforward(self, data):
        c_data = data
        for i in range(0 , self.number):
            if i == self.number - 1:
                w = self._tanh(np.dot(self.weights_array[i], c_data) + self.bias_array[i])
                return w
            w = self._tanh(np.dot(self.weights_array[i], c_data) + self.bias_array[i])
            c_data = w
        return w


# In[3]:


def train_marvin():
    epoch = 0
    lst2 = [0]
    neurone = []
    neurone_reward = []
    train_batch = 200
    for _ in range(train_batch):
        neurone.append(NeuralNetwork(24,15,8,4))
    for epoch in range(1000):
        if args.b == epoch:
            break
        for nn in neurone:
            rewards = run3(nn)
            neurone_reward.append([rewards,nn])
        sorted_rewards = sorted(neurone_reward, key=lambda tup: tup[0], reverse=True)
        neurone = []
        for neuron in sorted_rewards[:30]:
            neurone.append(neuron[1])
            for i in range(20):
                    neurone.append(cross_over2(sorted_rewards[random.randint(0,100)][1], neuron[1]))
                    neurone.append(cross_over2(sorted_rewards[random.randint(0,100)][1], NeuralNetwork(24,15,8,4)))
                    neurone.append(cross_over(sorted_rewards[random.randint(0,100)][1], neuron[1]))
                    neurone.append(cross_over(sorted_rewards[random.randint(0,100)][1], NeuralNetwork(24,15,8,4)))
        lst2 = [item[0] for item in sorted_rewards[0:5]]
        if not args.walk:
            print("Epoch", epoch)
            print("Top 5 rewards" , lst2)
        if lst2[0] > 100:
            if not args.walk:
                print("Calculating avg:....")
            avg = test_neuron(sorted_rewards[0][1])
            if not args.walk:
                print("Average:",avg)
            if avg > 100:
                run2(sorted_rewards[0][1])
                break
        run2(sorted_rewards[0][1])
        neurone_reward = []
    return sorted_rewards[0][1]


# In[1]:


if not isinstance(args.load, str):
    ganadora = train_marvin()
else:
    w_b = open_file(args.load)
    neuron = NeuralNetwork(24,15,8,4)
    neuron.weights_array = w_b[0]
    neuron.bias_array = w_b[1]
    reward = run2(neuron)
    print("Total Reward:", reward)
if isinstance(args.save, str):
    save_file(ganadora,args.save)


# In[ ]:




