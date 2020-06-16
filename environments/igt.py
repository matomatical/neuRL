#!/usr/bin/env python
# coding: utf-8

# TODO: Add references

import numpy as np

# #### Scheme 1
# 
# > Rewards and losses from 40 choices of each deck as used in the traditional
# payoff scheme with variable loss in deck C (see Bechara et al.[1]; classified
# here as payoff scheme 1). Within each deck, the presented payoff sequence is
# repeated after participants have made 40 choices from the corresponding deck.

class RewardScheme1:
    def __init__(self, random_seed=None):
        self.acounts = np.zeros(4, dtype=int)
        self.pos = np.array([100, 100, 50, 50,])
        self.neg = np.array([
            [    0,     0,  -150,     0,  -300,     0,  -200,     0,  -250,  -350,
                 0,  -350,     0,  -250,  -200,     0,  -300,  -150,     0,     0,
                 0,  -300,     0,  -350,     0,  -200,  -250,  -150,     0,     0,
              -350,  -200,  -250,     0,     0,     0,  -150,  -300,     0,     0,
            ],
            [    0,     0,     0,     0,     0,     0,     0,     0, -1250,     0,
                 0,     0,     0, -1250,     0,     0,     0,     0,     0,     0,
             -1250,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                 0, -1250,     0,     0,     0,     0,     0,     0,     0,     0,
            ],
            [    0,     0,   -50,     0,   -50,     0,   -50,     0,   -50,   -50,
                 0,   -25,   -75,     0,     0,     0,   -25,   -75,     0,   -50,
                 0,     0,     0,   -50,   -25,   -50,     0,     0,   -75,   -50,
                 0,     0,     0,   -25,   -25,     0,   -75,     0,   -50,   -75,
            ],
            [    0,     0,     0,     0,     0,     0,     0,     0,     0,  -250,
                 0,     0,     0,     0,     0,     0,     0,     0,     0,  -250,
                 0,     0,     0,     0,     0,     0,     0,     0,  -250,     0,
                 0,     0,     0,     0,  -250,     0,     0,     0,     0,     0
            ],
        ])
    def reset(self):
        self.acounts = np.zeros(4, dtype=int)
    def rewards(self, action):
        a = "ABCD".index(action)
        r = (self.pos[a], self.neg[a, self.acounts[a] % 40])
        self.acounts[a] += 1
        return r
    def __str__(self):
        return "Scheme 1"


# #### Scheme 2
# 
# > A possible sequence of rewards and losses from 10 choices of each deck
# based on the traditional payoff scheme with constant loss in deck C (see
# Bechara et al.[1]; classified here as payoff scheme 2). A payoff sequence
# with the presented characteristics is randomly generated for each block of 10
# trials.

class RewardScheme2:
    def __init__(self, random_seed=None):
        self.n = 0
        self.pos = np.array([100, 100, 50, 50,])
        self.neg = np.array([
            [    0,  -300,  -150,     0,  -350,     0,     0,  -250,     0,  -200],
            [    0,     0,     0,     0,     0, -1250,     0,     0,     0,     0],
            [    0,     0,   -50,     0,   -50,     0,   -50,     0,   -50,   -50],
            [    0,     0,     0,     0,     0,     0,     0,     0,  -250,     0],
        ])
        self.rng = np.random.default_rng(seed=random_seed)
    def reset(self, random_state=None):
        if random_state: self.rng.set_state(random_state)
        self.n = 0
    def rewards(self, action):
        # reorder the schemes every 10 rounds
        if self.n % 10 == 0:
            for a in range(4):
                self.rng.shuffle(self.neg[a])
        self.n += 1
        # draw a reward from the scheme
        a = "ABCD".index(action)
        r = (self.pos[a], self.neg[a, self.n % 10])
        return r
    def __str__(self):
        return "Scheme 2"


# #### Scheme 3
# 
# > Rewards and losses from 60 choices of each deck as used in the payoff
# scheme introduced by Bechara & Damasio ([2]; classified here as payoff scheme
# 3).

# TODO


# #### Scheme R
# 
# > Table 4: Schemes of Iowa Gambling Task

class RewardSchemeR:
    def __init__(self, random_seed=None, variable_C=True):
        self.pos = np.array([100, 100, 50, 50])
        self.neg = np.array([
            [0, -150, -200, -250, -300, -350],
            [0, -1250],
            [0, -25, -50, -75] if variable_C else [0, -50],
            [0, -250],
        ])
        self.prs = np.array([
            [0.5, 0.1, 0.1, 0.1, 0.1, 0.1],
            [0.9, 0.1],
            [0.5, 0.1, 0.3, 0.1] if variable_C else [0.5, 0.5],
            [0.9, 0.1],
        ])
        self.rng = np.random.default_rng(seed=random_seed)
    def reset(self):
        pass
    def rewards(self, action):
        a = "ABCD".index(action)
        return (self.pos[a], self.rng.choice(self.neg[a], p=self.prs[a]))
    def __str__(self):
        return "Scheme R"

def reward_scheme(scheme, random_seed):
    if scheme == "1" or scheme == 1:
        return RewardScheme1(random_seed=random_seed)
    elif scheme == "2" or scheme == 2:
        return RewardScheme2() # no seed
    elif scheme == "3" or scheme == 3:
        raise Exception("Sorry, scheme 3 implementation missing.")
    elif scheme == "R":
        return RewardSchemeR(random_seed=random_seed)
    else:
        return scheme # better be a reward scheme!


class IowaGamblingTask:
    def __init__(self, scheme="R", max_nrounds=100, split_rewards=False, random_seed=None):
        self.max_nrounds = max_nrounds
        self.scheme = reward_scheme(scheme, random_seed)
        self.split_rewards = split_rewards
        self.action_space = np.arange(4)
        self.reset()
    def reset(self):
        self.scheme.reset()
        self.nrounds = 0
        return self.nrounds
    def step(self, action):
        self.nrounds += 1
        reward = self.scheme.rewards("ABCD"[action])
        if not self.split_rewards:
            reward = sum(reward)
        done = (self.nrounds == self.max_nrounds)
        return self.nrounds, reward, done, {}
    def render(self, end="\r", **kwargs):
        print("Iowa Gambling Task ({}).".format(self.scheme), 
              "{:3d} / {:3d} rounds complete.".format(self.nrounds, self.max_nrounds),
              end=end, **kwargs)
    def close(self, **kwargs):
        print(**kwargs)


class TwoStepIowaGamblingTask:
    def __init__(self, scheme="R", n_steps=2, random_seed=None):
        self.n_steps = n_steps
        self.scheme = reward_scheme(scheme, random_seed)
        self.action_spaces = [np.arange(4)] + [np.arange(1)]*4*(self.n_steps-1)
        self.state = 0
    def reset(self):
        self.scheme.reset()
        self.state = 0
        return self.state
    def step(self, action):
        if self.state == 0:
            self.state = action+1
            return self.state, 0, False, {}
        else:
            # ignore action and sample reward
            self.state += 4
            done = False
            reward = 0
            if self.state > 4 * (self.n_steps - 1):
                # end of task!
                done = True
                reward = sum(self.scheme.rewards("ABCD"[(self.state-1) % 4]))
                self.state = 0
            return self.state, reward, done, {}
    def render(self, end="\r", **kwargs):
        print("Two-step Iowa Gambling Task ({}) state {}.".format(self.scheme, self.state),
              end=end, **kwargs)
    def close(self, **kwargs):
        print(**kwargs)

if __name__ == '__main__':
    env = IowaGamblingTask(split_rewards=True, scheme="R")
    done = False
    s = env.reset()
    env.render()
    while not done:
        s, r, done, _ = env.step('A')
        env.render()
    env.close()

    env = TwoStepIowaGamblingTask(split_rewards=True, scheme="R")
    done = False
    s = env.reset()
    env.render()
    while not done:
        s, r, done, _ = env.step('A')
        env.render()
    env.close()
