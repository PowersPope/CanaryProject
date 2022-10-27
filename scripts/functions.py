# Import modules
import numpy as np
import pandas as pd


def duration_to_str(line: str) -> str:
    """
    Convert some given duration data into Markov Model readable tokens.
    Int duration -> Str approximation added onto phase token.
    Durations will be broken up into 3 categories based up on the range of the data. 1/3 splits.
    Low,Medium,High
    ---
    E.g.
    str -> GRJKM
    duration -> 4,10,12,4,50
    ---
    For a Low data point a # will be added
    G#RJK#M
    ---
    For a Medium data point a * will be added
    GR*J*KM
    ---
    For a High data point a ^ will be added
    GRJKM^
    ---
    End Result:
    GRJKM -> G#R*J*K#M^
    """
    # break the line up on the comma
    token_str: str
    duration: str
    token_str, duration = line.split(',')
    ####
    # convert the duration into a list of int items
    dur_list: list = [eval(x) for x in duration.rstrip('*').split('*')]
    # convert list into numpy array
    dur_array: np.array = np.array(dur_list)
    ####
    # Determine the range and then therefore the quantile cuts for the range of this data
    data_range: np.ndarray = np.ptp(dur_array)
    cuts: np.ndarray = np.floor(data_range/3)
    min_data: int = np.min(dur_array)
    low: int = min_data + cuts
    med: int = min_data + cuts + cuts
    high: int = np.max(dur_array)
    ####
    # cycle elements in token_str and add duration characters
    n: str
    i: int
    mutant_str: str = ''
    for i,n in enumerate(list(token_str)):
        num: int = dur_array[i]
        if num <= low:
            new_n: str = n + '#'
            mutant_str += new_n
        elif num <= med:
            new_n: str = n + '*'
            mutant_str += new_n
        else:
            new_n: str = n + '^'
            mutant_str += new_n
    return mutant_str








