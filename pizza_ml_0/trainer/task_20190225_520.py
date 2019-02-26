import argparse
import os
import tensorflow as tf      # Deep Learning library
import numpy as np           # Handle matrices
from builtins import input

#from trainer.helpers import discount_rewards, prepro
#from agents.tools.wrappers import AutoReset, FrameHistory
from collections import deque# Ordered collection with ends
from trainer.game import Game
import random

#import matplotlib.pyplot as plt # Display graphs

#import gym                   #can remove (with call to gym)

# Game H
ACTIONS = ["down", "right", "next"]  # (CONVERTION TO ONE HOT)
R = 6 #200 # WONT WORK ON CNN! if < 8
C = 7 #250
OBSERVATION_DIM = R * C * 2 + 4 # give ingredient map (RxC), map of slices (RxC) ....cursor position (2x1) x slice_mode (1)  and your constraints L, H (2)

MEMORY_CAPACITY = 100000        #(OK)
ROLLOUT_SIZE = 50 #10000        #(OK)

# MEMORY stores tuples:
# (observation, label, reward)
MEMORY = deque([], maxlen=MEMORY_CAPACITY)   #NEEDS TO BE CHECKED!!


### MODEL HYPERPARAMETERS
#state_size = [110, 84, 4]      # NEEDS TO BE CHECKED !!! Our input is a stack of 4 frames hence 110x84x4 (Width, height, channels)
state_size = [OBSERVATION_DIM] #[R, C, 2]  # NEEDS TO BE CHECKED !!! ([1, OBSERVATION_DIM]) ? [None, OBSERVATION_DIM]
action_size = len(ACTIONS)    #(OK) 5 possible actions
learning_rate =  0.00025      #(OK) Alpha (aka learning rate)

### TRAINING HYPERPARAMETERS
total_episodes = 60000            #(OK) Total episodes for training (6000 EPOCHS)
max_steps = 50                 #(OK) Max possible steps in an episode [ROLLOUT_SIZE]
batch_size = 64     #instead of 64           #10000                # NEEDS TO BE CHECKED !!! Batch size 64?

# Exploration parameters for epsilon greedy strategy
explore_start = 1.0            # (OK)exploration probability at start
explore_stop = 0.01            # (OK)minimum exploration probability
decay_rate = 0.00001           # (OK)exponential decay rate for exploration prob

# Q learning hyperparameters
gamma = 0.9                    # (OK)Discounting rate

### MEMORY HYPERPARAMETERS
pretrain_length = batch_size   # NEEDS TO BE CHECKED !!! Number of experiences stored in the Memory when initialized for the first time
memory_size = MEMORY_CAPACITY  # (OK)Number of experiences the Memory can keep [MEMORY_CAPACITY]

### PREPROCESSING HYPERPARAMETERS
#stack_size = 4                 # NEEDS TO BE CHECKED Number of frames stacked

### MODIFY THIS TO FALSE IF YOU JUST WANT TO SEE THE TRAINED AGENT
training = True

## TURN THIS TO TRUE IF YOU WANT TO RENDER THE ENVIRONMENT
episode_render = True           #(OK) use args.render (default=False)

def preprocess(state_dict):
    state = np.concatenate((
        np.array(state_dict['ingredients_map']).ravel(),
        np.array(state_dict['slices_map']).ravel(),
        np.array(state_dict['cursor_position']).ravel(),
        [state_dict['min_each_ingredient_per_slice'],
        state_dict['max_ingredients_per_slice']],
    ))
    return state.astype(np.float).ravel()

possible_actions = np.array(np.identity(action_size,dtype=int).tolist()) #(OK)
print("The action size is : ", action_size) #(OK)
print(possible_actions)                #(OK)

class DQNetwork:
    def __init__(self, state_size, action_size, learning_rate, name='DQNetwork'):
        self.state_size = state_size         #(OK)
        self.action_size = action_size       #(OK)
        self.learning_rate = learning_rate   #(OK) args.

        with tf.variable_scope(name):
            # We create the placeholders
            # *state_size means that we take each elements of state_size in tuple hence is like if we wrote
        	### Replacing [None, *state_size] by [1, batch_size, *state_size] NOPE needs [None, *state_size for predict_action (1 value)]
            self.inputs_ = tf.placeholder(tf.float32, [None, *state_size], name="inputs")
            #self.inputs_ = tf.placeholder(tf.float32, [1, batch_size, *state_size], name="inputs")
            self.actions_ = tf.placeholder(tf.float32, [None, self.action_size], name="actions_")

            # Remember that target_Q is the R(s,a) + ymax Qhat(s', a')
            self.target_Q = tf.placeholder(tf.float32, [None], name="target")


            ### INIT self.flatten to our flatten state!!! (no CNN for now)
            self.flatten = self.inputs_
            #print("==========Yo==========")
            #print(self.flatten)

            # append 5 node features at the end (cursor 2x1, L, H)
            self.fc = tf.layers.dense(inputs = self.flatten,
                                  units = 512,
                                  activation = tf.nn.elu,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name="fc1")
            self.fc2 = tf.layers.dense(inputs = self.fc,
                                  units = 512,
                                  activation = tf.nn.elu,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name="fc2")

            self.output = tf.layers.dense(inputs = self.fc2,
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          units = self.action_size,
                                        activation=None)



            # Q is our predicted Q value.
            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_))

            # The loss is the difference between our predicted Q_values and the Q_target
            # Sum(Qtarget - Q)^2
            self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))

            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)


# Reset the graph
tf.reset_default_graph()

# Instantiate the DQNetwork
DQNetwork = DQNetwork(state_size, action_size, learning_rate)



class Memory():
    def __init__(self, max_size):
        self.buffer = deque(maxlen = max_size)
    #MEMORY = deque([], maxlen=MEMORY_CAPACITY)   #NEEDS TO BE CHECKED!!
    #MEMORY is buffer....
    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size),
                                size = batch_size,
                                replace = False)

        return [self.buffer[i] for i in index] # similar to the generator....



# Instantiate memory
memory = Memory(max_size = memory_size)
for i in range(pretrain_length):
    # If it's the first step
    if i == 0:

        game = Game({'max_steps':max_steps}) # initialize game from game.py // not 10000
        h = 6 #random.randint(1, R * C + 1)
        l = 1 #random.randint(1, h // 2 + 1)
        pizza_lines = [''.join([random.choice("MT") for _ in range(C)]) for _ in range(R)]
        pizza_lines = ["TMMMTTT","MMMMTMM", "TTMTTMT", "TMMTMMM", "TTTTTTM", "TTTTTTM"]
        pizza_config = { 'pizza_lines': pizza_lines, 'r': R, 'c': C, 'l': l, 'h': h }
        _state = preprocess(game.init(pizza_config)[0])  #np.zeros(OBSERVATION_DIM) #get only first value of tuple

    # Get the next_state, the rewards, done by taking a random action
    choice = random.randint(1,len(possible_actions))-1
    action = possible_actions[choice] #as one-hot
    #translate _action into 1 to 5 action for the game...
    _action = ACTIONS[np.argmax(action)]
    next_state, _reward, _done, _ = game.step(_action)  #next_state is _state in Game agent
    _next_state = preprocess(next_state)

    if episode_render and i % 20 == 0:        # NEEDS TO BE CHECKED  args.render:
        game.render()


    # If the episode is finished (we maxed out the number of frames)
    if _done:
        # We finished the episode
        _next_state = np.zeros(_state.shape) # _state is flattened with cursor, L and H appended

        # Add experience to memory (push action one-hot encoded instead of _action label e.g.'right')
        memory.add((_state, action, _reward, _next_state, _done))

        # Start a new episode
        game = Game({'max_steps':max_steps}) # initialize game from game.py not
        h = 6 #random.randint(1, R * C + 1)
        l = 1 #random.randint(1, h // 2 + 1)
        pizza_lines = [''.join([random.choice("MT") for _ in range(C)]) for _ in range(R)]
        pizza_lines = ["TMMMTTT","MMMMTMM", "TTMTTMT", "TMMTMMM", "TTTTTTM", "TTTTTTM"]
        pizza_config = { 'pizza_lines': pizza_lines, 'r': R, 'c': C, 'l': l, 'h': h }
        _state = preprocess(game.init(pizza_config)[0])  #np.zeros(OBSERVATION_DIM) #get only first value of tuple
        ### Watch-out, _observation will be flattened and won't conserve Rc needed for CNN

    else:
        # Add experience to memory (push action one-hot encoded instead of _action label e.g.'right')
        memory.add((_state, action, _reward, _next_state, _done))

        # Our new state is now the next_state
        _state = _next_state


# Setup TensorBoard Writer  #NEEDS TO BE CHECKED
#summary_path = os.path.join(args.output_dir, 'summary')
summary_path = os.path.join('gs://pizza-game/', 'summary')

writer = tf.summary.FileWriter(summary_path)
#writer = tf.summary.FileWriter("./tensorboard/dqn/1")

for var in tf.trainable_variables():
    tf.summary.histogram(var.op.name, var)
    tf.summary.scalar('{}_max'.format(var.op.name), tf.reduce_max(var))
    tf.summary.scalar('{}_min'.format(var.op.name), tf.reduce_min(var))

#tf.summary.scalar('rollout_reward', rollout_reward)
tf.summary.scalar('loss', DQNetwork.loss)
#tf.summary.scalar('reward', _total_reward)


write_op = tf.summary.merge_all()

def predict_action(explore_start, explore_stop, decay_rate, decay_step, state, actions):
    ## EPSILON GREEDY STRATEGY
    # Choose action a from state s using epsilon greedy.
    ## First we randomize a number
    exp_exp_tradeoff = np.random.rand()

    # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
    explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)

    if (explore_probability > exp_exp_tradeoff):
        # Make a random action (exploration)
        choice = random.randint(1,len(possible_actions))-1
        action = possible_actions[choice]

    else:
        # Get action from Q-network (exploitation)
        # Estimate the Qs values state
        Qs = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs_: state.reshape((1, *state.shape))})

        # Take the biggest Q value (= the best action)
        choice = np.argmax(Qs)
        action = possible_actions[choice]

    return action, explore_probability

# Saver will help us to save our model
saver = tf.train.Saver()

if training == True:
    with tf.Session() as sess:
        # Initialize the variables
        sess.run(tf.global_variables_initializer())

        # Initialize the decay rate (that will use to reduce epsilon)
        decay_step = 0
        rewards_list = []
        for episode in range(total_episodes):
            # Set step to 0
            step = 0

            # Initialize the rewards of the episode
            episode_rewards = []
            episode_actions = []
            # Make a new episode and observe the first state
            # Start a new episode
            game = Game({'max_steps':50}) # initialize game from game.py
            h = 6 #random.randint(1, R * C + 1)
            l = 1 #random.randint(1, h // 2 + 1)
            pizza_lines = [''.join([random.choice("MT") for _ in range(C)]) for _ in range(R)]
            pizza_lines = ["TMMMTTT","MMMMTMM", "TTMTTMT", "TMMTMMM", "TTTTTTM", "TTTTTTM"]
            pizza_config = { 'pizza_lines': pizza_lines, 'r': R, 'c': C, 'l': l, 'h': h }
            _state = preprocess(game.init(pizza_config)[0])

            while step < max_steps:
                step += 1

                #Increase decay_step
                decay_step +=1

                # Predict the action to take and take it
                action, explore_probability = predict_action(explore_start, explore_stop, decay_rate, decay_step, _state, possible_actions)
                #action is one-hot... so translate _action into 1 to 5 action for the game...
                _action = ACTIONS[np.argmax(action)]

                #Perform the action and get the next_state, reward, and done information
                next_state, _reward, _done, _ = game.step(_action)  #next_state is _state in Game agent
                _next_state = preprocess(next_state)

                # Add the reward to total reward
                episode_rewards.append(_reward)
                episode_actions.append(_action)

                # If the game is finished
                if _done:
                    # The episode ends so no next state
                    _next_state = np.zeros(_state.shape) # _state is flattened with cursor, L and H appended

                    # Set step = max_steps to end the episode
                    step = max_steps

                    # Get the total reward of the episode
                    total_reward = np.sum(episode_rewards)
                    if (episode % 100 == 0 and episode < 500) or (episode % 1000 == 0):
                        print('Episode: {}'.format(episode),
                                      'Total reward: {}'.format(total_reward),
                                      'Explore P: {:.4f}'.format(explore_probability),
                                    'Training Loss {:.4f}'.format(loss))
                        print(episode_actions)
                        print(episode_rewards)
                    rewards_list.append((episode, total_reward))

                    # Store transition <st,at,rt+1,st+1> in memory D
                    memory.add((_state, action, _reward, _next_state, _done))

                    if episode_render and ((episode % 100 == 0 and episode < 500) or (episode % 1000 == 0)):
                        game.render()
                else:
                    # Add experience to memory
                    memory.add((_state, action, _reward, _next_state, _done))

                    # st+1 is now our current state
                    _state = _next_state


                ### LEARNING PART
                # Obtain random mini-batch from memory
                batch = memory.sample(batch_size)
                # reshaping states by using squeeze....
                states_mb = np.array([each[0] for each in batch], ndmin=3)
                states_mb = np.squeeze(states_mb, axis=0)
                actions_mb = np.array([each[1] for each in batch])

                rewards_mb = np.array([each[2] for each in batch])
                # reshaping next_states by using squeeze....
                next_states_mb = np.array([each[3] for each in batch], ndmin=3)
                next_states_mb = np.squeeze(next_states_mb, axis=0)

                dones_mb = np.array([each[4] for each in batch])
                target_Qs_batch = []

                # Get Q values for next_state /!\ --- Why shape of DQNetwork.inputs_ = (1, 64, 89??)
                Qs_next_state = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs_: next_states_mb})

                # Set Q_target = r if the episode ends at s+1, otherwise set Q_target = r + gamma*maxQ(s', a')
                for i in range(0, len(batch)):
                    terminal = dones_mb[i]

                    # If we are in a terminal state, only equals reward
                    if terminal:
                        target_Qs_batch.append(rewards_mb[i])

                    else:
                        target = rewards_mb[i] + gamma * np.max(Qs_next_state[i])
                        target_Qs_batch.append(target)


                targets_mb = np.array([each for each in target_Qs_batch])
                loss, _ = sess.run([DQNetwork.loss, DQNetwork.optimizer],
                                        feed_dict={DQNetwork.inputs_: states_mb,
                                                   DQNetwork.target_Q: targets_mb,
                                                   DQNetwork.actions_: actions_mb})

                # Write TF Summaries
                summary = sess.run(write_op, feed_dict={DQNetwork.inputs_: states_mb,
                                                       DQNetwork.target_Q: targets_mb,
                                                       DQNetwork.actions_: actions_mb})
                #_total_reward = tf.placeholder(tf.float32, (), name="tot_reward")
                writer.add_summary(summary, episode)
                writer.flush()

            # Save model every 5 episodes
            if episode % 1000 == 0:
                save_path = saver.save(sess, "./models/model.ckpt")
                print("Model Saved")

with tf.Session() as sess:
    total_test_rewards = []

    # Load the model
    saver.restore(sess, "./models/model.ckpt")

    for episode in range(1):
        total_rewards = 0

        game = Game({'max_steps':200}) # initialize game from game.py
        h = 6 #random.randint(1, R * C + 1)
        l = 1 #random.randint(1, h // 2 + 1)
        pizza_lines = [''.join([random.choice("MT") for _ in range(C)]) for _ in range(R)]
        pizza_lines = ["TMMMTTT","MMMMTMM", "TTMTTMT", "TMMTMMM", "TTTTTTM", "TTTTTTM"]
        pizza_config = { 'pizza_lines': pizza_lines, 'r': R, 'c': C, 'l': l, 'h': h }
        _state = preprocess(game.init(pizza_config)[0])  #np.zeros(OBSERVATION_DIM) #get only first value of tuple

        print("****************************************************")
        print("EPISODE ", episode)

        while True:
            _state = _state.reshape((1, *state_size))
            # Get action from Q-network
            # Estimate the Qs values state
            Qs = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs_: _state})

            # Take the biggest Q value (= the best action)
            choice = np.argmax(Qs)
            action = possible_actions[choice] #as one-hot
            #translate _action into 1 to 5 action for the game...
            _action = ACTIONS[np.argmax(action)]
            print(_action)

            #Perform the action and get the next_state, reward, and done information
            next_state, _reward, _done, _ = game.step(_action)  #next_state is _state in Game agent
            _next_state = preprocess(next_state)
            game.render()

            total_rewards += _reward

            if _done:
                print ("Score", total_rewards)
                total_test_rewards.append(total_rewards)
                break

            _state = _next_state
