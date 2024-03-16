
%%capture
!apt update
!apt install xvfb -y
!pip install 'swig'
!pip install 'pyglet==1.5.27'
!pip install 'gym[box2d]==0.20.0'
!pip install 'pyvirtualdisplay==3.0'

import gym
import random
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import sys
from pyvirtualdisplay import Display
from IPython import display as disp
%matplotlib inline

display = Display(visible=0,size=(600,600))
display.start()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
plot_interval = 10 # update the plot every N episodes
video_every = 100 # videos can take a very long time to render so only do it every N episodes


class ReplayBuffer():#Our replay buffer class
  def __init__(self,max_size,input_shape,act_dim):
    self.mem_size=max_size
    self.mem_cntr=0
    self.state_memory=np.zeros((self.mem_size,*input_shape))
    self.new_state_memory=np.zeros((self.mem_size,*input_shape))
    self.action_memory=np.zeros((self.mem_size,act_dim))
    self.reward_memory=np.zeros(self.mem_size)
    self.terminal_memory=np.zeros(self.mem_size,dtype=bool)#Memory of done flags
  def store_transition(self,state,action,reward,state_,done):
    index = self.mem_cntr % self.mem_size
    self.state_memory[index] = state
    self.new_state_memory[index] = state_
    self.terminal_memory[index] = done
    self.reward_memory[index] = reward
    self.action_memory[index] = action
    self.mem_cntr += 1
  def sample_buffer(self,batch_size):
    max_mem=min(self.mem_cntr,self.mem_size)
    #Get random sample of walks
    batch=np.random.choice(max_mem,batch_size)
    states=self.state_memory[batch]
    states2=self.new_state_memory[batch]
    actions=self.action_memory[batch]
    rewards=self.reward_memory[batch]
    dones=self.terminal_memory[batch]
    #Return not as tensors yet
    return states,actions,rewards,states2,dones


class Critic(nn.Module):#Our critic network class
  def __init__(self,lr,obs_dim,act_dim):
    super(Critic,self).__init__()
    #Save objects space, action space
    self.obs_dim=obs_dim
    self.act_dim=act_dim

    #Create network layers here
    self.layer1=nn.Linear(self.obs_dim[0]+act_dim,400)
    self.layer2=nn.Linear(400,300)
    self.q=nn.Linear(300,1)

    #Initialise optimizer Adam
    self.optimizer=optim.Adam(self.parameters(),lr=lr)
    self.to(device)#Send to the CPU

  def forward(self,state,action):
    #Pass through our critic network here
    q=F.relu(self.layer1(torch.cat([state,action],dim=1)))
    q=F.relu(self.layer2(q))
    q=self.q(q)#What is this--------------------------------------
    return q


class Actor(nn.Module):#Our actor network class
  def __init__(self,lr,obs_dim,act_dim):
    super(Actor,self).__init__()
    #Save objects space, action space 
    self.obs_dim=obs_dim
    self.act_dim=act_dim

    #Create network layers here
    self.layer1=nn.Linear(*self.obs_dim,400)#The * does tuple unpacking
    self.layer2=nn.Linear(400,300)
    self.output=nn.Linear(300,self.act_dim)

    #Initialise optimizer Adam
    self.optimizer=optim.Adam(self.parameters(),lr=lr)#Learning Rate
    self.to(device)#Send to the CPU

  def forward(self,s):
    #Pass through our actor network here
    p=F.relu(self.layer1(s))
    p=F.relu(self.layer2(p))
    output=torch.tanh(self.output(p))#Apply tanh here
    return output


#Agent ties everything together, make actor and critics objects seperate
class Agent():#Removed nn.Module here as used above and I didn't need it
    def __init__(self,lr,gamma,tau,batch_size,warmup,exploration_noise,policy_noise,update_interval,obs_dim,act_dim):#Noise for exploring
      super(Agent, self).__init__()
      self.gamma=gamma
      self.tau=tau
      self.max_action=env.action_space.high
      self.min_action=env.action_space.low
      self.memory=ReplayBuffer(1000000,obs_dim,act_dim)
      self.batch_size=batch_size
      self.learn_step_cntr=0
      self.time_step=0
      self.warmup=warmup
      self.act_dim=act_dim
      self.update_interval=update_interval
      self.exploration_noise=exploration_noise
      self.policy_noise=policy_noise

      #Initialising actors and critics
      self.actor=Actor(lr,obs_dim,act_dim)
      self.target_actor=copy.deepcopy(self.actor)
      self.critic1=Critic(lr,obs_dim,act_dim)
      self.critic2=copy.deepcopy(self.critic1)
      self.target_critic1=copy.deepcopy(self.critic1)
      self.target_critic2=copy.deepcopy(self.critic1)
  
      #Update agents parameters
      self.update_parameters(tau=1)

    def sample_action(self, s):
      if self.time_step<self.warmup:#Delayed start
        pi=torch.tensor(np.random.normal(scale=self.exploration_noise,size=(self.act_dim,)))
      else:
        state=torch.tensor(s,dtype=torch.float).to(device)
        pi=self.actor.forward(state).to(device)#Has Noise
      pi_prime=pi+torch.tensor(np.random.normal(scale=self.exploration_noise),dtype=torch.float).to(device)
      pi_prime=torch.clamp(pi_prime,self.min_action[0],self.max_action[0])
      #Must get back in limits
      self.time_step+=1
      #Dereference, pass a numpy array for action
      return pi_prime.cpu().detach().numpy()
  
    def remember(self,state,action,reward,new_state,done):
      self.memory.store_transition(state,action,reward,new_state,done)
      #Messy but need interface between objects

    def train(self,iter):
      #Fill to batch size then start
      if self.memory.mem_cntr<self.batch_size:
        return
      for i in range(iter):
        state,action,reward,new_state,done=self.memory.sample_buffer(self.batch_size)
        #Convert np arrays to tensors
        reward=torch.tensor(reward,dtype=torch.float).to(device)
        done2=torch.tensor(done).to(device)#MUST CALCULATE DIFFERENT DONES HERE
        done=torch.FloatTensor(1-done).to(device)
        state2=torch.tensor(new_state,dtype=torch.float).to(device)
        state=torch.tensor(state,dtype=torch.float).to(device)
        action=torch.tensor(action,dtype=torch.float).to(device)
        #pytorch specific about these datatypes going to devices

        #Update here
        target_actions=self.target_actor.forward(state2)
        target_actions=target_actions+torch.clamp(torch.tensor(np.random.normal(scale=self.policy_noise)),-0.5,0.5)
        target_actions=torch.clamp(target_actions,self.min_action[0],self.max_action[0])

        #Critic values
        q1_=self.target_critic1.forward(state2,target_actions)
        q2_=self.target_critic2.forward(state2,target_actions)

        #Action values for states/actions actually according to critic
        q1=self.critic1.forward(state,action)
        q2=self.critic2.forward(state,action)

        q1_[done2]=0.0
        q2_[done2]=0.0

        q1_ = q1_.view(-1)
        q2_ = q2_.view(-1)
        #Double q learning update rule
        critic_value=torch.min(q1_,q2_)
        target = reward + (done*self.gamma*critic_value).detach()#Can remove done here
        #Add batch dimension 
        target=target.view(self.batch_size,1)
        #Gradient Descent
        #Loss functions
        q1_loss=F.mse_loss(q1,target)
        q2_loss=F.mse_loss(q2,target)
        #Zero optimizers first
        self.critic1.optimizer.zero_grad()
        self.critic2.optimizer.zero_grad()
        #Backpropagate
        critic_loss=q1_loss+q2_loss
        critic_loss.backward()
        self.critic1.optimizer.step()
        self.critic2.optimizer.step()

        self.learn_step_cntr+=1
        if i%self.update_interval==0:
          #Update actor here
          self.actor.optimizer.zero_grad()
          actor_q1_loss=self.critic1.forward(state,self.actor.forward(state))
          actor_loss=-torch.mean(actor_q1_loss)
          actor_loss.backward()#Maybe move backward step after
          self.actor.optimizer.step()
          #Update agent parameters
          self.update_parameters()

    def update_parameters(self,tau=None):
      #Corner case, want to initialise target networks
      if tau is None:
        tau=self.tau
      actor_params=self.actor.named_parameters()
      critic1_params=self.critic1.named_parameters()
      critic2_params=self.critic2.named_parameters()
      target_actor_params=self.target_actor.named_parameters()
      target_critic1_params=self.target_critic1.named_parameters()
      target_critic2_params=self.target_critic2.named_parameters()  

      critic1=dict(critic1_params)
      critic2=dict(critic2_params)
      actor=dict(actor_params)
      target_actor=dict(target_actor_params)
      target_critic1=dict(target_critic1_params)
      target_critic2=dict(target_critic2_params)

      for name in critic1:
        critic1[name]=tau*critic1[name].clone()+(1-tau)*target_critic1[name].clone()
      for name in critic2:
        critic2[name]=tau*critic2[name].clone()+(1-tau)*target_critic2[name].clone()
      for name in actor:
        actor[name]=tau*actor[name].clone()+(1-tau)*target_actor[name].clone()

      self.target_critic1.load_state_dict(critic1)
      self.target_critic2.load_state_dict(critic2)
      self.target_actor.load_state_dict(actor)

"""**Prepare the environment and wrap it to capture videos**"""

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# env = gym.make("BipedalWalker-v3")
# # env = gym.make("BipedalWalkerHardcore-v3") # only attempt this when your agent has solved BipedalWalker-v3
# env = gym.wrappers.Monitor(env, "./video", video_callable=lambda ep_id: ep_id%video_every == 0, force=True)
# 
# obs_dim = env.observation_space.shape
# act_dim = env.action_space.shape[0]

print('The environment has {} observations and the agent can take {} actions'.format(obs_dim, act_dim))
print('The device is: {}'.format(device))

if device.type != 'cpu': print('It\'s recommended to train on the cpu for this')

# in the submission please use seed 42 for verification
seed = 42
torch.manual_seed(seed)
env.seed(seed)
random.seed(seed)
np.random.seed(seed)
env.action_space.seed(seed)

# logging variables
ep_reward = 0
reward_list = []
plot_data = []
log_f = open("agent-log.txt","w+")

#HYPERPARAMETERS
lr=0.001
tau=0.005 #Target policy update paramater is 1-tau
batch_size=100
exploration_noise=0.1
policy_noise=0.2#Smoothing policy
warmup=1000#not using anymore
update_interval=2
gamma=0.95

#Runtime Parameters
max_episodes = 1000
max_timesteps = 2000


# initialise agent
agent = Agent(lr,gamma,tau,batch_size,warmup,exploration_noise,policy_noise,update_interval,obs_dim,act_dim)
# training procedure:
for episode in range(1, max_episodes+1):
    state = env.reset()
    done=False
    t=0
    temp_buffer=[]
    expcount=0
    for t in range(max_timesteps):#Change from original only update actor after each episode
      # select the agent action
      action = agent.sample_action(state)
      # take action in environment and get r and s'
      next_state, reward, done, info = env.step(action)
      ep_reward+=reward

      if reward==-100:#recommended Reward scaling
        reward=-5
      else:
        reward=5*reward

      agent.remember(state,action,reward,next_state,done)
      state = next_state
      if done or t==(max_timesteps-1):
        agent.train(t)
        break

    # append the episode reward to the reward list
    reward_list.append(ep_reward)#Still add the true reward

    # do NOT change this logging code - it is used for automated marking!
    log_f.write('episode: {}, reward: {}\n'.format(episode, ep_reward))
    log_f.flush()
    ep_reward = 0
    
    # print reward data every so often - add a graph like this in your report
    if episode % plot_interval == 0:
        plot_data.append([episode, np.array(reward_list).mean(), np.array(reward_list).std()])
        reward_list = []
        # plt.rcParams['figure.dpi'] = 100
        plt.plot([x[0] for x in plot_data], [x[1] for x in plot_data], '-', color='tab:grey')
        plt.fill_between([x[0] for x in plot_data], [x[1]-x[2] for x in plot_data], [x[1]+x[2] for x in plot_data], alpha=0.2, color='tab:grey')
        plt.xlabel('Episode number')
        plt.ylabel('Episode reward')
        plt.show()
        disp.clear_output(wait=True)
