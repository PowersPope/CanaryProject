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


def nstep(N: int, duration: bool) -> int:
    """
    Given N return the correct steps/kmer size of the Markov Model.

    For example 2-Step Markov Model needs to have a kmer of 2 which can be
    written with an i,k of 0,2.
    This way if you have a word of size M
    If N <= M then you can take a kmer step within it.
    """
    # If duration is added
    if duration:
        # mutiply the nstep by 2 to get the correct tokens with their duration token
        M: int = N * 2
    else:
        # return what the step is to add to the base step
        M: int = N

    return M

def split_token(nstep: int, duration: bool) -> tuple[int, int]:
    """
    Return the splitting token size based upon if the duration is True and
    what nstep is passed into the data.
    E.g.
    nstep = 3
    #### duration == True
    K = (3 * 2) + 1
    J = (3 * 2)
    #### duration == False
    K = 3 + 1
    J = 3
    """
    # Only special case where * 2 is not appropriate
    if nstep == 1:
        N: int = 2
    else:
        N: int = nstep
    # based on if duration is true then we need to multiply by 2 and add 1
    if duration:
        K: int = (N * 2) + 1
        J: int = (N * 2)
    else:
        K: int =  N + 1
        J: int =  N
    # return the touple of K and J values
    return (K, J)








