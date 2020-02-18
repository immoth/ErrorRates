import numpy as np
from copy import deepcopy
import itertools
from fractions import Fraction as F

from models import dual_ising
from mps.mps import iMPS
from algorithms import simulation
from algorithms.linalg import np_conserved as npc
from tools.string import joinstr, to_mathematica_lists

print(np.pi)
