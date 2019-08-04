import collections
from enum import Enum

import gym
import numpy as np
import re
import logging
from tensorboardX import SummaryWriter
from gym import envs

from core import TabularPolicy, EpisodeResult, EpsilonSoftTabularPolicy, StateActionValueTable, CustomPolicy

logger = logging.getLogger(__file__)


class NStepAlgorithm(Enum):
    N_STEP_TD_PREDICTION = "N_STEP_TD_PREDICTION"


class NStepAgent(object):

    def __init__(self):
        self.v_table = {}
        self.q_table = StateActionValueTable()
        self.policy = None
