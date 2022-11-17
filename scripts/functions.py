# Import modules
import numpy as np
import pandas as pd
import os
import sys



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
    # Don't need this code anymore as I have the hard coded cutoffs now
    # data_range: np.ndarray = np.ptp(dur_array)
    # cuts: np.ndarray = np.floor(data_range/3)
    # min_data: int = np.min(dur_array)
    low: int = 104 # min_data + cuts
    med: int = 169 # min_data + cuts + cuts
    # high: int = np.max(dur_array)
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
        M: int = N * 4
    else:
        # return what the step is to add to the base step
        M: int = N * 2

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
    # if nstep == 1:
        # N: int = 2
    # else:
    N: int = nstep
    # based on if duration is true then we need to multiply by 2 and add 1
    if duration:
        K: int = ((N * 2) + 1)
        J: int = N * 2
    else:
        K: int =  N + 1
        J: int =  N
    # return the touple of K and J values
    return (K, J)


def kmerize_string(element: str, kmer_size: int, duration: bool) -> list[str]:
    """
    Take the string and kmerize it based upon the desired kmer size
    """
    n: int = 0
    k: int = kmer_size
    kmer_str_list: list = list()
    while k <= len(element):
        kmer_str_list.append(element[n:k])
        if duration:
            n+= 2
            k+= 2
        else:
            n+= 1
            k+= 1
    return kmer_str_list




##### find the distribution of the data
def get_distribution()-> tuple[int]:
    """
    Take in the durations for each data set and figure out what the low, medium, and high
    thresholds should be
    """
    # grab files
    path = "../CANARY_CSV_DATA/Laura_CSV/"
    files: list = os.listdir(path)
    # Specify the list that will hold all of the numbers
    duration_total: list = list()
    # Filter through files
    file: str
    for file in files:
        # We only want clean files
        if r'clean_' not in file: continue
        # Grab clean file and put it into a variable for pandas to read from
        file_path: str = os.path.join(path,file)
        df: pd.DataFrame = pd.read_csv(file_path)
        # iter through the duration and separate the strs into ints
        idx: int
        row: pd.Series
        for idx, row in df.iterrows():
            dur_row: list = row.duration.rstrip("*").split("*")
            nums: list = [int(x) for x in dur_row]
            duration_total.extend(nums)
    # Total pass done
    duration_array: np.array = np.array(duration_total)
    bins: np.array = np.split(np.sort(duration_array), 3)
    low, med, high = np.max(bins[0]), np.max(bins[1]), np.max(bins[2])
    return (low, med, high)

