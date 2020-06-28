ENV_NAME = 'FetchPickAndPlace-v1'

import torch.nn.functional as F
import random
from IPython import embed

from copy import deepcopy
from collections import deque
from IPython.display import clear_output
import gym
import torch
import torch.nn as nn
import torch.optim as opt
from torch import Tensor
from torch.autograd import Variable
from collections import namedtuple
from itertools import count
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
from os.path import join as joindir
from os import makedirs as mkdir
import pandas as pd
import numpy as np
import argparse
import datetime
import math
def select_action(action_mean, action_logstd, fctr):
    """
    given mean and std, sample an action from normal(mean, std)
    also returns probability of the given chosen
    """
    action_std = torch.exp(action_logstd)*fctr
    action = torch.normal(action_mean, action_std)
    return action
def eval_policy_50(fctr_used):
    reward_sum = 0
    succ_game = 0
    for display_i in range(50):
        #print(num_steps)
        env.reset()
        state = env.env._get_obs()
        state = np.concatenate((state['observation'],state['desired_goal'])) # state_extended
        episode = []
        env_list = []
        Succ_in_env = 0
        for t in range(args.max_step_per_round):
            #print(t)
            network.eval()
            action_mean, action_logstd, value = network(Tensor(state).unsqueeze(0))
            action_mean = action_mean.detach()
            action_logstd = action_logstd.detach()
            value = value.detach()

            action = select_action(action_mean, action_logstd, fctr_used)
            action = torch.clamp(action,-1,1)
            action = action.data.cpu().numpy()[0]
            next_state, reward, done, _ = env.step(action)

            if _['is_success'] !=0:
                Succ_in_env = 1
                break

            next_state = np.concatenate((next_state['observation'],next_state['desired_goal']))

            reward_sum += reward

            mask = 0 if done else 1

            if done:
                break
            state = next_state
        succ_game += Succ_in_env

    return succ_game/50


'''ablation study recorder'''
Eval_different_sigma_recorder = []
Tot_Ret_2 = []
Tot_Ret_1 = []
Tot_Ret_0 = []
Acceptance_rate = []
Eval_list_0 = []
Eval_list_0p1 = []
Eval_list_0p2 = []
Traj_num_recorder = []

for repeat in range(5):
    FACTOR = 1.5
    class args(object):
        env_name = ''
        seed = 1234 + repeat
        num_episode = 600
        batch_size = 2500
        max_step_per_round = 200
        gamma = 0.995
        lamda = 0.97
        log_num_episode = 1
        num_epoch = 30
        minibatch_size = 25
        clip = 0.2
        loss_coeff_value = 0.5
        loss_coeff_entropy = 0.01
        factor = FACTOR
        lr_ppo =0*3e-4 # 
        lr_hid = 3e-4
        future_p = 0.0 # param of HER
        Horizon_max = 8 # param of PCHID
        reward_pos = 0. # reward for success 

        num_parallel_run = 1
        # tricks
        schedule_adam = 'linear'
        schedule_clip = 'linear'
        layer_norm = True
        state_norm = False
        advantage_norm = True
        lossvalue_norm = True
        replay_buffer_size_IER = 100000
    
    FACTOR_LIST = []
    rwds = []
    Succ_recorder = []
    Horizon_list = [i+1 for i in range(args.Horizon_max)]
    losses = [[] for i in range(len(Horizon_list)) ]
    Transition = namedtuple('Transition', ('state', 'value', 'action', 'logproba', 'mask', 'next_state', 'reward'))
    EPS = 1e-10
    RESULT_DIR = 'results'
    mkdir(RESULT_DIR, exist_ok=True)




    class ActorCritic(nn.Module):
        def __init__(self, num_inputs, num_outputs, layer_norm=True):
            super(ActorCritic, self).__init__()

            self.actor_fc1 = nn.Linear(num_inputs, 64)
            self.actor_fc2 = nn.Linear(64, 64)
            self.actor_fc2_1 = nn.Linear(64,64)
            self.actor_fc3 = nn.Linear(64, num_outputs)
            self.actor_logstd = nn.Parameter(torch.zeros(1, num_outputs))

            self.critic_fc1 = nn.Linear(num_inputs, 64)
            self.critic_fc2 = nn.Linear(64, 64)
            self.critic_fc3 = nn.Linear(64, 1)

            if layer_norm:
                self.layer_norm(self.actor_fc1, std=1.0)
                self.layer_norm(self.actor_fc2, std=1.0)
                self.layer_norm(self.actor_fc2_1, std=1.0)
                self.layer_norm(self.actor_fc3, std=0.01)

                self.layer_norm(self.critic_fc1, std=1.0)
                self.layer_norm(self.critic_fc2, std=1.0)
                self.layer_norm(self.critic_fc3, std=1.0)

        @staticmethod
        def layer_norm(layer, std=1.0, bias_const=0.0):
            torch.nn.init.orthogonal_(layer.weight, std)
            torch.nn.init.constant_(layer.bias, bias_const)

        def forward(self, states):
            """
            run policy network (actor) as well as value network (critic)
            :param states: a Tensor2 represents states
            :return: 3 Tensor2
            """
            action_mean, action_logstd = self._forward_actor(states)
            critic_value = self._forward_critic(states)
            return action_mean, action_logstd, critic_value

        def _forward_actor(self, states):
            x = F.leaky_relu(self.actor_fc1(states))
            x = F.leaky_relu(self.actor_fc2(x))
            x = F.leaky_relu(self.actor_fc2_1(x))
            action_mean = self.actor_fc3(x)
            action_logstd = self.actor_logstd.expand_as(action_mean)
            assert action_logstd.mean() == 0
            return action_mean, action_logstd

        def _forward_critic(self, states):
            x = torch.tanh(self.critic_fc1(states))
            x = torch.tanh(self.critic_fc2(x))
            critic_value = self.critic_fc3(x)
            return critic_value

        def select_action(self, action_mean, action_logstd, return_logproba=True, factor = 1.0):
            """
            given mean and std, sample an action from normal(mean, std)
            also returns probability of the given chosen
            """
            action_std = torch.exp(action_logstd)*factor
            action = torch.normal(action_mean, action_std)
            if return_logproba:
                logproba = self._normal_logproba(action, action_mean, action_logstd, action_std)
            return action, logproba

        @staticmethod
        def _normal_logproba(x, mean, logstd, std=None):
            if std is None:
                std = torch.exp(logstd)

            std_sq = std.pow(2)
            logproba = - 0.5 * math.log(2 * math.pi) - logstd - (x - mean).pow(2) / (2 * std_sq)
            return logproba.sum(1)

        def get_logproba(self, states, actions):
            """
            return probability of chosen the given actions under corresponding states of current network
            :param states: Tensor
            :param actions: Tensor
            """
            action_mean, action_logstd = self._forward_actor(states)
            logproba = self._normal_logproba(actions, action_mean, action_logstd)
            return logproba

        
        
    class ReplayBuffer_imitation(object):
        def __init__(self, capacity):
            self.buffer = {'1step':deque(maxlen=capacity)}
            self.capacity = capacity
        def push(self, state, action, step_num):
            try:
                self.buffer[step_num]
            except:
                self.buffer[step_num] = deque(maxlen=self.capacity)
            self.buffer[step_num].append((state, action))


        def sample(self, batch_size,step_num):
            state, action= zip(*random.sample(self.buffer[step_num], batch_size))
            return np.stack(state), action

        def lenth(self,step_num):
            try:
                self.buffer[step_num]
            except:
                return 0
            return len(self.buffer[step_num])

        def __len__(self,step_num):
            try:
                self.buffer[step_num]
            except:
                return 0
            return len(self.buffer[step_num])


    env = gym.make(ENV_NAME)  
    num_inputs = env.observation_space.spaces['observation'].shape[0] +  env.observation_space.spaces['desired_goal'].shape[0] # extended state
    num_actions = env.action_space.shape[0]
    network = ActorCritic(num_inputs, num_actions, layer_norm=args.layer_norm)

    '''joint train'''
    model_imitation = network


    def espd(args):
        def compute_cross_ent_error(batch_size,step_num):
            if ier_buffer.lenth(step_num)==0:
                return None
            if batch_size>ier_buffer.lenth(step_num):
                return None
            state, action= ier_buffer.sample(batch_size,step_num)
            state          = torch.FloatTensor(state)#.to(device)
            action_target  = torch.FloatTensor(action)#.to(device)
            action_pred    = model_imitation(state)[0]

            loss_func = nn.MSELoss()
            loss = loss_func(action_pred,action_target)
            optimizer_imitation.zero_grad()
            loss.backward()
            optimizer_imitation.step()
            return loss
        def test_isvalid_multistep(step_lenth, state_start, environment_start,env):
            env_tim = env
            env_tim.sim.set_state(environment_start)
            env_tim.sim.forward()
            state_tim = deepcopy(state_start)
            for step_i in range(step_lenth):
                action_tim_mean, action_tim_logstd, value_tim = network(Tensor(state_tim).unsqueeze(0))
                action_tim_mean = torch.clamp(action_tim_mean,-1,1)
                action_tim = action_tim_mean.data.numpy()[0]
                next_state_tim, reward, done, _ = env_tim.step(action_tim)
                next_state_tim = np.concatenate((next_state_tim['observation'],next_state_tim['desired_goal']))

                next_state_tim[-3:] = deepcopy(state_tim[-3:])

                rwd_sim = env_tim.compute_reward(next_state_tim[3:6],next_state_tim[-3:],{'is_success': 0.0})
                if rwd_sim == 0:
                    if step_i <= step_lenth-1:
                        return 1 # should not learn
                    else:
                        return 0 # ok to learn
                state_tim = next_state_tim
            return 2 # learnable

                
        FACTOR = args.factor
        env = gym.make(args.env_name)
        num_inputs = env.observation_space.spaces['observation'].shape[0]+ env.observation_space.spaces['desired_goal'].shape[0] # extended state
        num_actions = env.action_space.shape[0]

        env.seed(args.seed)
        torch.manual_seed(args.seed)

        #optimizer = opt.RMSprop(network.parameters(), lr=args.lr_ppo)
        optimizer_imitation = opt.RMSprop(model_imitation.parameters(),lr = args.lr_hid)


        reward_record = []
        global_steps = 0

        #lr_now = args.lr_ppo
        clip_now = args.clip
        ier_buffer = ReplayBuffer_imitation(args.replay_buffer_size_IER)
        
        for i_episode in range(args.num_episode):
            episodic_pass_test_num = 0 
            num_steps = 0
            reward_list = []
            len_list = []
            Succ_num = 0

            game_num = 0
            succ_game = 0
            
            
            Ret_2 = [0*_ for _ in range(len(Horizon_list))]
            Ret_1 = [0*_ for _ in range(len(Horizon_list))]
            Ret_0 = [0*_ for _ in range(len(Horizon_list))]
            
            while num_steps < args.batch_size:
                
                '''interactions'''
                
                state = env.reset()
                
                game_num +=1
                state = np.concatenate((state['observation'],state['desired_goal'])) # state_extended

                reward_sum = 0
                episode = []
                env_list = []
                Succ_in_env = 0
                for t in range(args.max_step_per_round):
                    action_mean, action_logstd, value = network(Tensor(state).unsqueeze(0))
                    action, logproba = network.select_action(action_mean, action_logstd,factor = FACTOR)
                    
                    action = torch.clamp(action,-1,1)
                    action = action.data.numpy()[0]
                    logproba = logproba.data.numpy()[0]
                    
                    if len(Horizon_list) >= 2:
                        state_temp = env.env.sim.get_state()
                        env_list.append(state_temp)
                    
                    next_state, reward, done, _ = env.step(action)
                    if reward ==0:
                        Succ_in_env = 1
                        reward = args.reward_pos 
                        Succ_num+=1
                    next_state = np.concatenate((next_state['observation'],next_state['desired_goal']))

                    reward_sum += reward
                    mask = 0 if done else 1

                    episode.append((state, value, action, logproba, mask, next_state, reward))
                    if done:
                        break

                    state = next_state
                succ_game += Succ_in_env
                
                '''start learning'''
                
                for ind,(state, value, action, logproba, mask, next_state, reward) in enumerate(episode):
                    if len(Horizon_list)>=2:
                        assert len(env_list) == len(episode)
                    '''supervised learning'''
                    for t_ in Horizon_list:
                        try:
                            episode[t_+ind]
                        except:
                            break
                        
                        target_state_ = deepcopy(episode[t_+ind][-7])
                        state_ = deepcopy(state)
                        state_[-3:] = deepcopy(target_state_[3:6])
                        rwd_temp3 = np.linalg.norm(target_state_[3:6]-state_[3:6])
                        if rwd_temp3 > 0.05:
                            ret_tim = test_isvalid_multistep(t_, state_, env_list[ind],env)
                            if ret_tim==2:
                                ier_buffer.push(state_,action,'1step')
                                episodic_pass_test_num += 1
                                Ret_2[t_-1] +=1
                            elif ret_tim == 1:
                                Ret_1[t_-1] +=1
                            else:
                                Ret_0[t_-1] +=1
                
                num_steps += (t + 1)
                global_steps += (t + 1)
                reward_list.append(reward_sum)
                len_list.append(t + 1)
                Winrate = 1.0*succ_game/game_num
                Succ_recorder.append(Winrate)
            
            
            Traj_num_recorder.append(Ret_2)
            print('Return This Episode:',Ret_0,Ret_1,Ret_2)
            Acceptance_rate.append([round((Ret_2[_]/(Ret_2[_] + Ret_1[_] + Ret_0[_] + 1e-6))*100.0)/100.0 for _ in range(len(Ret_2))])
            Tot_Ret_2.append(Ret_2)
            Tot_Ret_1.append(Ret_1)
            Tot_Ret_0.append(Ret_0)
            
            
            reward_record.append({
                'episode': i_episode, 
                'steps': global_steps, 
                'meanepreward': np.mean(reward_list), 
                'meaneplen': np.mean(len_list)})

            rwds.extend(reward_list)
            batch_size = episodic_pass_test_num

            SR = 1.0*Succ_num/num_steps
            for i_epoch in range(int(args.num_epoch * batch_size / args.minibatch_size)):
                '''learning'''
                for h in [1]:
                    los_lst = []
                    flag = 0
                    loss1 = compute_cross_ent_error(args.minibatch_size,str(h)+'step')
                    if loss1 is not None:
                        flag = 1
                        losses[h-1].append(loss1.item())
                        los_lst.append('loss{}'.format(h))
            
            print('ier lenth',ier_buffer.lenth('1step'),ier_buffer.lenth('2step'),ier_buffer.lenth('3step'),ier_buffer.lenth('4step'),ier_buffer.lenth('5step'),ier_buffer.lenth('6step'),ier_buffer.lenth('7step'))
            if True:
                FACTOR = 1.0
                
            #FACTOR_LIST.append(FACTOR)
            #print(fct_eval_list)
            print('factor now is ',FACTOR)
            eval_0_temp = eval_policy_50(0.0)
            eval_0p1_temp = eval_policy_50(0.1)
            eval_0p2_temp = eval_policy_50(0.2)
            Eval_list_0.append(eval_0_temp)
            Eval_list_0p1.append(eval_0p1_temp)
            Eval_list_0p2.append(eval_0p2_temp)
            print('Eval_sr:',eval_0_temp,eval_0p1_temp,eval_0p2_temp)
            print('Acceptance Rate ',Acceptance_rate[-1])
            print('Traj length in this episode',Ret_2)
            #Eval_different_sigma_recorder.append(eval_each_factor_list)
            if args.schedule_clip == 'linear':
                ep_ratio = 1 - (i_episode / args.num_episode)
                clip_now = args.clip * ep_ratio

            if i_episode % args.log_num_episode == 0:
                print('Finished episode: {} Reward: {:.4f} SuccessRate{:.4f} WinRate{:.4f}' \
                    .format(i_episode, reward_record[-1]['meanepreward'],SR,Winrate))
                print('-----------------')
            #if i_episode==15:
            #    Horizon_list.append(3)
        return reward_record

    def test(args):
        record_dfs = []
        for i in range(args.num_parallel_run):
            args.seed += 1
            reward_record = pd.DataFrame(espd(args))
            reward_record['#parallel_run'] = i
            record_dfs.append(reward_record)
        record_dfs = pd.concat(record_dfs, axis=0)
        record_dfs.to_csv(joindir(RESULT_DIR, 'record-{}.csv'.format(args.env_name)))

   

    for envname in ['FetchPush-v1','FetchSlide-v1','FetchPickAndPlace-v1']:
        args.env_name = envname
        test(args)
        rwds_HER_HID= deepcopy(rwds)
        Succ_recorder_HER_HID= deepcopy(Succ_recorder)
        np.save('results/{0}_rewards_repeat{1}'.format(repeat,envname),rwds_HER_HID)
        np.save('results/{0}_SR_repeat{1}'.format(repeat,envname),Succ_recorder_HER_HID)
        np.save('results/{0}_Eval_list_0_repeat{1}'.format(repeat,envname),Eval_list_0)
        np.save('results/{0}_Eval_list_0p1_repeat{1}'.format(repeat,envname),Eval_list_0p1)
        np.save('results/{0}_Eval_list_0p2_repeat{1}'.format(repeat,envname),Eval_list_0p2)

        np.save('results/{0}_Tot_Ret_0_repeat{1}'.format(repeat,envname),Tot_Ret_0)
        np.save('results/{0}_Tot_Ret_1_repeat{1}'.format(repeat,envname),Tot_Ret_1)
        np.save('results/{0}_Tot_Ret_2_repeat{1}'.format(repeat,envname),Tot_Ret_2)
        np.save('results/{0}_Tot_Ret_2_repeat{1}'.format(repeat,envname),Acceptance_rate)
