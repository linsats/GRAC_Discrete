import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import datetime
import os

epsilon = 1e-6

class Actor(nn.Module):
  def __init__(self,state_dim,action_dim):
    super(Actor, self).__init__()
    # 64, 64
    self.fc_a = nn.Sequential(
      nn.Linear(state_dim, 256),
      nn.ReLU(),
      nn.Linear(256, 256),
      nn.ReLU(),
      nn.Linear(256, action_dim),
    )

  def forward(self, state):
    state = self.fc_a(state)
    a_prob = F.softmax(state)
    a_dist = Categorical(a_prob)
    action = a_dist.sample()
    action = action.cpu().data.numpy()
    dist_entropy = a_dist.entropy()
    return action, a_prob, dist_entropy

class Critic(nn.Module):
  def __init__(self,state_dim,action_dim):
    super(Critic, self).__init__()
    self.fc_q1 = nn.Sequential(
      nn.Linear(state_dim, 256),
      nn.ReLU(),
      nn.Linear(256, 256),
      nn.ReLU(),
      nn.Linear(256, action_dim),
    )
    self.fc_q2= nn.Sequential(
      nn.Linear(state_dim, 256),
      nn.ReLU(),
      nn.Linear(256, 256),
      nn.ReLU(),
      nn.Linear(256, action_dim),
    )

  def forward_all(self, state):
    q1 = self.fc_q1(state)
    q2 = self.fc_q2(state)
    return q1,q2

  def forward(self, state, action):
    q1,q2 = self.forward_all(state)
    q1 = q1.gather(1,action)
    q2 = q2.gather(1,action)
    return q1, q2


class GRAC():
  def __init__(
          self,
          env,
          state_dim,
          action_dim,
          batch_size = 256,
          discount = 0.99,
          alpha_start = 0.75,
          alpha_end = 0.85,
          n_repeat = 20,
          actor_lr = 3e-4,
          critic_lr = 3e-4,
          max_timesteps = 8e6,
          device = torch.device('cuda'),
    ):
    self.actor_lr = actor_lr
    self.critic_lr = critic_lr
    self.device = device
    self.state_dim = state_dim
    self.action_dim = action_dim
 
    self.actor = Actor(state_dim, action_dim).to(device)
    self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)

    self.critic = Critic(state_dim,action_dim).to(device)
    self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),lr=self.critic_lr)

    self.log_freq = 1
    self.discount = discount
    self.alpha_start = alpha_start 
    self.alpha_end = alpha_end

    self.n_repeat = n_repeat
    self.max_timesteps = max_timesteps
    self.selection_action_ceof = 1.0
    self.total_it = 0

  def select_action(self, state, writer=None, test=False):
    state = torch.FloatTensor(state.reshape((-1,self.state_dim))).to(self.device)
  
    if test is False:
      with torch.no_grad():
        if np.random.uniform(0,1) < 0.9:
          action_index, a_prob, dist = self.actor.forward(state)
          action_index = action_index[0]
          #q1,q2 = self.critic.forward_all(state)
          #q1 = q1.cpu().data.numpy().flatten()
          #action_index = np.argmax(q1)
        else:
          action_index = np.random.choice(self.action_dim)
        writer.add_scalar('train_action/index',action_index,self.total_it)
        return action_index
    else:
        action_index, a_prob, dist = self.actor.forward(state)
        a_prob = a_prob.cpu().data.numpy().flatten()
        action_index = np.argmax(a_prob)
        #q1,q2 = self.critic.forward_all(state)
        #q1 = q1.cpu().data.numpy().flatten()
        #action_index = np.argmax(q1)
        return action_index

  def update_critic(self, critic_loss):
    # optimize the critic
    self.critic_optimizer.zero_grad()
    critic_loss.backward()
    self.critic_optimizer.step()

  def lr_scheduler(self, optimizer, lr):
    for param_group in optimizer.param_groups:
      param_group['lr'] = lr
    return optimizer

  def train(self, replay_buffer, batch_size=100, writer=None, reward_range=20.0):
    self.total_it += 1
    log_it = (self.total_it % self.log_freq == 0)

    # sample replay buffer
    state, action, next_state, reward, not_done = replay_buffer.sample(batch_size) 

    with torch.no_grad():
      # select action according to policy
      target_Q1, target_Q2 = self.critic.forward_all(next_state)
      target_Q1_max, target_Q1_max_index = torch.max(target_Q1,dim=1,keepdim=True)
      target_Q2_max, target_Q2_max_index = torch.max(target_Q1,dim=1,keepdim=True)
      target_Q = torch.min(target_Q1_max,target_Q2_max)
      #target_action_index = (target_Q > target_Q2_max)
      #print("target_Q1_max",target_Q1_max)
      #print("target_Q1_max_index",target_Q1_max_index)
      #print("target_Q2_max",target_Q2_max)
      #print("target_Q2_max_inex",target_Q2_max_index)
      #target_action = target_Q1_max_index.clone()
      #target_action[target_action_index] = target_Q2_max_index[target_action_index]
      target_Q_final = reward + not_done * self.discount * target_Q
      #target_Q1_target_action = target_Q1.gather(1, target_action)
      #target_Q2_target_action = target_Q2.gather(1, target_action)
      if log_it:
        writer.add_scalar("train_critic/target_Q1_max_index_std",torch.std(target_Q1_max_index.clone().double()),self.total_it)
    # Get current q estimation
    current_Q1,current_Q2 = self.critic(state,action)
    # compute critic_loss
    critic_loss = F.mse_loss(current_Q1, target_Q_final) + F.mse_loss(current_Q2, target_Q_final)  
    self.update_critic(critic_loss)

    current_Q1_,current_Q2_ = self.critic(state, action)
    target_Q1_, target_Q2_  = self.critic.forward_all(next_state)
    critic_loss3_p1 = F.mse_loss(current_Q1_, target_Q_final) + F.mse_loss(current_Q2_, target_Q_final)
    critic_loss3_p2 = F.mse_loss(target_Q1_, target_Q1) + F.mse_loss(target_Q2_, target_Q2) 
    critic_loss3 = critic_loss3_p1 + critic_loss3_p2
    self.update_critic(critic_loss3)
    if log_it:
      writer.add_scalar('train_critic/loss3_p1',critic_loss3_p1, self.total_it)
      writer.add_scalar('train_critic/loss3_p2',critic_loss3_p2, self.total_it)
    init_critic_loss3 = critic_loss3.clone()

    idi = 0
    cond1 = 0
    cond2 = 0

    while True:
      idi = idi + 1
      current_Q1_,current_Q2_ = self.critic(state, action)
      target_Q1_, target_Q2_  = self.critic.forward_all(next_state)
      critic_loss3_p1 = F.mse_loss(current_Q1_, target_Q_final) + F.mse_loss(current_Q2_, target_Q_final)
      critic_loss3_p2 = F.mse_loss(target_Q1_, target_Q1) + F.mse_loss(target_Q2_, target_Q2)
      critic_loss3 = critic_loss3_p1 + critic_loss3_p2
      self.update_critic(critic_loss3)
      if self.total_it < self.max_timesteps:
        bound = self.alpha_start + float(self.total_it) / float(self.max_timesteps) * (self.alpha_end - self.alpha_start)
      else:
        bound = self.alpha_end
      if critic_loss3 < init_critic_loss3 * bound:
        cond1 = 1
        break
      if idi >= self.n_repeat:
        cond2 = 1
        break

    if 1:
      writer.add_scalar('train_critic/third_loss_cond1', cond1, self.total_it)
      writer.add_scalar('train/third_loss_bound', bound, self.total_it)
      writer.add_scalar('train_critic/third_loss_num', idi, self.total_it)
      if self.total_it < 1000:
        writer.add_scalar('train/actor_lr',self.actor_lr,self.total_it)
      writer.add_scalar("losses/repeat_l1",critic_loss, self.total_it)
      writer.add_scalar("losses/repeat_l3",critic_loss3, self.total_it)     

    critic_loss = F.mse_loss(current_Q1_, target_Q_final) + F.mse_loss(current_Q2_, target_Q_final) 
    weights_actor_lr = critic_loss.detach()
    writer.add_scalar('train/weights_actor_lr',weights_actor_lr,self.total_it)

    lr_tmp = self.actor_lr / (float(weights_actor_lr)+1.0)
    self.actor_optimizer = self.lr_scheduler(self.actor_optimizer, lr_tmp)
    actor_index, a_prob, dist_entropy = self.actor.forward(state)
    if log_it:
      writer.add_scalar("train_actor/a_prob_std",torch.std(a_prob),self.total_it)
      writer.add_scalar("train_actor/a_prob_dist0",torch.mean(a_prob[:,0]),self.total_it)
      writer.add_scalar("train_actor/a_prob_dist1",torch.mean(a_prob[:,1]),self.total_it)
      writer.add_scalar("train_actor/a_prob_dist2",torch.mean(a_prob[:,2]),self.total_it)
      writer.add_scalar("train_actor/a_prob_dist3",torch.mean(a_prob[:,3]),self.total_it)
      #writer.add_scalar("train_actor/a_prob_dist4",torch.mean(a_prob[:,4]),self.total_it)
      #writer.add_scalar("train_actor/a_prob_dist5",torch.mean(a_prob[:,5]),self.total_it)
      #writer.add_scalar("train_actor/a_prob_dist6",torch.mean(a_prob[:,6]),self.total_it)
      #writer.add_scalar("train_actor/a_prob_dist7",torch.mean(a_prob[:,7]),self.total_it)
      #writer.add_scalar("train_actor/a_prob_dist8",torch.mean(a_prob[:,8]),self.total_it)
      writer.add_scalar("train_actor/a_prob_max",torch.max(a_prob),self.total_it)
      writer.add_scalar("train_actor/a_prob_min",torch.min(a_prob),self.total_it)

    current_Q1_,current_Q2_ = self.critic.forward_all(state)
    V_2 = torch.sum(a_prob * current_Q2_,axis=1,keepdim=True)
    adv = (current_Q1_ - V_2).detach()
    if log_it:
      writer.add_scalar("train_action/adv",torch.mean(adv),self.total_it)
    actor_loss = - (a_prob * adv).mean() - 0.1 * dist_entropy.mean()

    self.actor_optimizer.zero_grad()
    actor_loss.backward()
    self.actor_optimizer.step()
    self.actor_optimizer = self.lr_scheduler(self.actor_optimizer, self.actor_lr)

    if log_it:
      #current_Q1
      writer.add_scalar('train_critic/current_Q1/mean', torch.mean(current_Q1), self.total_it)
      writer.add_scalar('train_critic/current_Q1/max', current_Q1.max(), self.total_it)
      writer.add_scalar('train_critic/current_Q1/min', current_Q1.min(), self.total_it)
      writer.add_scalar('train_critic/current_Q1/std', torch.std(current_Q1), self.total_it)

      writer.add_scalar('train_critic/current_Q2/mean', torch.mean(current_Q2), self.total_it)
      writer.add_scalar('train_critic/current_Q2/max', current_Q2.max(), self.total_it)
      writer.add_scalar('train_critic/current_Q2/min', current_Q2.min(), self.total_it)
      writer.add_scalar('train_critic/current_Q2/std', torch.std(current_Q2), self.total_it)
