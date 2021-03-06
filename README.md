# MetaQNN_ImageGenerationGAN_DiscriminatorFixed_PyTorch
Implementation of MetaQNN (https://arxiv.org/abs/1611.02167, https://github.com/bowenbaker/metaqnn.git) with Additions and Modifications in PyTorch for Image Generation with GAN where the Discriminator network is fixed and same as that in the infoGAN paper (https://arxiv.org/abs/1606.03657)       
      
# Basic Search Space Specs:
 i) Minimum no. of Conv./Wrn layers   
 ii) Maximum no. of Conv./Wrn layers   
 iii) No FC layer 
 iv) Search over latent space

# Additions/Modifications:
 i) Optional Greedy version of Q-learning update rule added for shorter search schedules (Greedy version being used because init_utility for initiliazation of utility value for all transitions difficult to be set in case of VCAE since there is no generic measure for VCAE performance)   
```python
    def update_q_value_sequence(self, states, termination_reward, latent_size, flag):
        states[-1].fc_size = latent_size
        self._update_q_value(states[-2], states[-1], termination_reward, flag)
        for i in reversed(range(len(states) - 2)): 
            # TODO: q-learning update (set proper q-learning rate in cmdparser.py)
            # self._update_q_value(states[i], states[i+1], 0, flag)

            # TODO: modified update for shorter search schedules (doesn't use q-learning rate in computation)
            self._update_q_value(states[i], states[i+1], termination_reward, flag)

    def _update_q_value(self, start_state, to_state, reward, flag):
        if flag == 0:
            if start_state.as_tuple() not in self.qstore_enc.q:
                self.enum.enumerate_state(start_state, self.qstore_enc.q)
            if to_state.as_tuple() not in self.qstore_enc.q:
                self.enum.enumerate_state(to_state, self.qstore_enc.q)

            actions = self.qstore_enc.q[start_state.as_tuple()]['actions']
            values = self.qstore_enc.q[start_state.as_tuple()]['utilities']

            max_over_next_states = max(self.qstore_enc.q[to_state.as_tuple()]['utilities']) if to_state.terminate != 1 else 0

            action_between_states = self.enum.transition_to_action(start_state, to_state).as_tuple()

            # TODO: q-learning update (set proper q-learning rate in cmdparser.py)
            # values[actions.index(action_between_states)] = values[actions.index(action_between_states)] + \
            #                                       self.state_space_parameters.learning_rate * (reward + \
            #                                       self.state_space_parameters.discount_factor * max_over_next_states\
            #                                        - values[actions.index(action_between_states)])

            # TODO: modified update for shorter search schedules (doesn't use q-learning rate in computation)
            values[actions.index(action_between_states)] = values[actions.index(action_between_states)] + \
                                                        (max(reward, values[actions.index(action_between_states)]) -
                                                         values[actions.index(action_between_states)])

            self.qstore_enc.q[start_state.as_tuple()] = {'actions': actions, 'utilities': values}

        elif flag == 1:
            if start_state.as_tuple() not in self.qstore_dec.q:
                self.enum.enumerate_state(start_state, self.qstore_dec.q)
            if to_state.as_tuple() not in self.qstore_dec.q:
                self.enum.enumerate_state(to_state, self.qstore_dec.q)

            actions = self.qstore_dec.q[start_state.as_tuple()]['actions']
            values = self.qstore_dec.q[start_state.as_tuple()]['utilities']

            max_over_next_states = max(self.qstore_dec.q[to_state.as_tuple()]['utilities']) if to_state.terminate != 1 else 0

            action_between_states = self.enum.transition_to_action(start_state, to_state).as_tuple()

            # TODO: q-learning update (set proper q-learning rate in cmdparser.py)
            # values[actions.index(action_between_states)] = values[actions.index(action_between_states)] + \
            #                                       self.state_space_parameters.learning_rate * (reward + \
            #                                       self.state_space_parameters.discount_factor * max_over_next_states\
            #                                        - values[actions.index(action_between_states)])

            # TODO: modified update for shorter search schedules (doesn't use q-learning rate in computation)
            values[actions.index(action_between_states)] = values[actions.index(action_between_states)] + \
                                                        (max(reward, values[actions.index(action_between_states)]) -
                                                         values[actions.index(action_between_states)])
            self.qstore_dec.q[start_state.as_tuple()] = {'actions': actions, 'utilities': values}
```
 ii) Skip connections with WideResNet blocks, minimum and maximum conv layer limit and made some other search space changes for better performace    
 iii) Continuing from the previous Q-learning iteration if code crashes  while running      
 iv) Running over single or multiple GPUs    
 iv) Automatic calculation of available GPU space and skipping of architecture if it doesn't fit      
             
# NOTE:    
code for MNIST, CIFAR10 and STL; for other datasets dataloader has to be added to _lib/Datasets/datasets.py_    
# Intalling Code Dependencies -     
```sh
pip install -r requirements.txt
```     
# Running Search:       
Use _python 2.7_ and _torch 0.4.0_      
Look at __lib/cmdparser.py__ for the available command line options or just run 
```sh
$ python main.py --help
```
      
Finally, run __main.py__
       
       
# Running script to plot rolling mean from replay dictionary   
Look at __plot/plotReplayDictRollingMean.py__ for the available command line options or just run 
```sh
$ python plot/plotReplayDictRollingMean.py --help
```    
    
Finally, run __plot/plotReplayDictRollingMean.py__ with the necessary command line arguments
