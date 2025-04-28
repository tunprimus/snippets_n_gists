#!/usr/bin/env python3
from matplotlib import rcParams
from matplotlib.cm import rainbow
from auto_pip_finder import PipFinder
from dbscan_pp import DBSCANPP


## Some Constants
##*********************##
RANDOM_SEED = 42
RANDOM_SAMPLE_SIZE = 13
NUM_DEC_PLACES = 4
GOLDEN_RATIO = 1.618033989
FIG_WIDTH = 20
FIG_HEIGHT = FIG_WIDTH / GOLDEN_RATIO
FIG_SIZE = (FIG_WIDTH, FIG_HEIGHT)
FIG_DPI = 72

