import numpy as np
import torch
import gym
import argparse
import os
import cv2
import imageio
from skimage.color import rgb2gray

import datetime
import utils

from torch.utils.tensorboard import SummaryWriter

def eval_policy(policy, env_name, seed, eval_episodes=10):
  eval_env = gym.make(env_name)
  #eval_env.seed(seed + 100)
   
  avg_reward = 0
  for _ in range(eval_episodes):
    state, done = eval_env.reset(), False
    while not done:
      action = policy.select_action(np.array(state), test=True)
      state, reward, done, _ = eval_env.step(action)
      avg_reward += reward
  avg_reward /= float(eval_episodes)

  print("---------------------------------------")
  print("Evaluation over {} episodes: {:.3f}".format(eval_episodes, avg_reward))
  print("---------------------------------------")
  return avg_reward


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--policy", default="GRAC")                  # Policy name (GRAC)
  parser.add_argument("--env", default="Breakout-ram-v0")          # OpenAI gym environment name
  parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
  parser.add_argument("--start_timesteps", default=1e4, type=int) # Time steps initial random policy is used
  parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
  parser.add_argument("--max_timesteps", default=2e6, type=int)   # Max time steps to run environment
  parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
  parser.add_argument("--batch_size", default=64, type=int)      # Batch size for both actor and critic
  parser.add_argument("--discount", default=0.99)                 # Discount factor
  parser.add_argument("--n_repeat", default=200, type=int)       # Frequency of delayed policy updates
  parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
  parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
  parser.add_argument('--actor_cem_clip', default=0.5)
  parser.add_argument('--use_expl_noise', action="store_true")
  parser.add_argument("--debug", action="store_true")
  parser.add_argument("--comment", default="")
  parser.add_argument("--exp_name", default="exp_logs")
  parser.add_argument("--which_cuda", default=0, type=int)

  args = parser.parse_args()

  device = torch.device('cuda:{}'.format(args.which_cuda))

  file_name = "{}_{}_{}".format(args.policy, args.env, args.seed)
  file_name += "_{}".format(args.comment) if args.comment != "" else ""
  folder_name = datetime.datetime.now().strftime('%b%d_%H-%M-%S_') + file_name
  result_folder = 'runs/{}'.format(folder_name) 
  if args.exp_name is not "":
    result_folder = '{}/{}'.format(args.exp_name, folder_name)
  if args.debug: 
    result_folder = 'debug/{}'.format(folder_name)
  if not os.path.exists('{}/models/'.format(result_folder)):
    os.makedirs('{}/models/'.format(result_folder))
  print("---------------------------------------")
  print("Policy: {}, Env: {}, Seed: {}".format(args.policy, args.env, args.seed))

  if not os.path.exists("./results"):
    os.makedirs("./results")
  if args.save_model and not os.path.exists("./models"):
    os.makedirs("./models")

  env = gym.make(args.env)
  state_dim = env.observation_space.shape[0]
  print(state_dim)
  action_dim = 4#env.action_space.shape[0]
  print(action_dim)
  print(type(action_dim))


  if args.save_model is False:
    args.save_model = True
    kwargs = {
		"env": args.env,
		"state_dim": state_dim,
		"action_dim": action_dim,
		"batch_size": args.batch_size,
		"discount": args.discount,
		"device": device,
  } 

  # Initialize policy
  if "GRAC" in args.policy:
    GRAC = __import__(args.policy)
    policy = GRAC.GRAC(**kwargs)

  replay_buffer = utils.ReplayBuffer(state_dim, device=device)
 
  evaluations = [eval_policy(policy, args.env, args.seed)] 

  #### Evaluation
  state,done = env.reset(), False
  episode_reward = 0
  episode_timesteps = 0
  episode_num = 0

  writer = SummaryWriter(log_dir=result_folder, comment=file_name)
 
  with open("{}/parameters.txt".format(result_folder), 'w') as file:
    for key, value in vars(args).items():
      file.write("{} = {}\n".format(key, value))


  for t in range(int(args.max_timesteps)):
    episode_timesteps += 1
  
    # select action randomly or according to policy
    if t < args.start_timesteps:
      action = np.random.randint(action_dim)
    else:
      action = policy.select_action(np.array(state),writer=writer)

    #Performa action
    next_state, reward, done, _ = env.step(action)
    writer.add_scalar('train/reward',reward,t+1)
    #img = np.copy(state)
    #img_g = cv2.resize(img,(128,128))
    #print("img_g",img_g.shape)
    #print(img_g)
    #print("state",state.shape)
    #cv2.imshow('image',img_g)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    # Store data in replay buffer
    replay_buffer.add(state,action,next_state,reward,done) 
    state = next_state
    episode_reward += reward
    if t >= args.start_timesteps and (t + 1) % 20 == 0:
      policy.train(replay_buffer, args.batch_size, writer, 20.0)
    if done:
      print("Total T {} Episode Num:{} Episode T:{} Reward: {:.3f}".format(t+1, episode_num+1, episode_timesteps, episode_reward))
 
      # reset environment
      state, done = env.reset(), False
      episode_reward = 0
      episode_timesteps = 0
      episode_num += 1

    #Evaluate episode 
    if t >= args.start_timesteps and (t + 1) % args.eval_freq == 0:
      evaluation = eval_policy(policy,args.env, args.seed)
      evaluations.append(evaluation)
      writer.add_scalar('test/avg_return', evaluation, t+1)
      #np.save("{}/evaluations".format(result_folder), evaluations)
