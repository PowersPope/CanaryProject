import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt



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
        # print(i, n)
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


#### New Markov Model Approach
def generate_initial_occurences(df: pd.DataFrame, alphabet_dict: dict,
                                nsteps: int, duration: bool) -> dict:
    """
    Take in a dataframe of strings with or without duration and separate it by the nsteps
    E.g.:
    Nsteps: 2
    !ATGCTABE-
    When it is the beginning you include the ! at the beginning and don't let it affect
    the Nstep
    {
    !AT: {A:0,T:0,G:1,...,-:0}
    }
    """
    # Init dict
    freq_dict: dict = dict()
    # iter through dataframe
    i: int
    row: pd.Series
    for i, row in df.iterrows():
        # grab the sequence
        seq: str = row.string + '-'
        if duration:
            if len(seq) <= nsteps*2+1: continue
            # parameters for taking steps
            if nsteps == 1:
                k: int = 2
            else:
                k: int = nsteps * 2
            # walk along the word
            n: int = 0
            next_char: str = ""
            while next_char != "-":
            # for n in range(0, len(seq), 2):
                # Generate the nstep token given that X0,..Xn
                if n == 0: # if the first iteration then add a ! at the beginning
                    token: str = "!" + seq[n:k]
                else: # if not first then do a normal
                    token: str = seq[n:k]
                # The next value which will be incremented in the value
                next_char: str = seq[k:k+2]
                # Now check if token is in the freq_dict
                if token in freq_dict:
                    freq_dict[token][next_char] += 1
                else:
                    freq_dict[token] = dict(zip(alphabet_dict, np.zeros(len(alphabet_dict),dtype=np.int32)))
                    # niche case that the token hasn't been seen before and the whole thing ends right away
                    freq_dict[token][next_char] += 1
                # increment k to keep walking along the sequence and generate tokens
                n += 2
                k += 2
                # if next_char == "-": break
        else:
            if len(seq) <= nsteps+1: continue
            # parameters for taking steps
            k: int = nsteps
            # walk along the word
            n: int = 0
            next_char: str = ""
            while next_char != "-":
                # Generate the nstep token given that X0,..Xn
                if n == 0: # if the first iteration then add a ! at the beginning
                    token: str = "!" + seq[n:k]
                else: # if not first then do a normal
                    token: str = seq[n:k]
                # The next value which will be incremented in the value
                next_char: str = seq[k]
                # Now check if token is in the freq_dict
                if token in freq_dict:
                    freq_dict[token][next_char] += 1
                else:
                    freq_dict[token] = dict(zip(alphabet_dict, np.zeros(len(alphabet_dict),dtype=np.int32)))
                    # niche case that the token hasn't been seen before and the whole thing ends right away
                    freq_dict[token][next_char] += 1
                # increment k to keep walking along the sequence and generate tokens
                k += 1
                n += 1
                # print('Before Break Check:', next_char)
                # if next_char == "-":
                    # break
    return freq_dict

def gen_alphabet(df: pd.DataFrame, duration: bool) -> dict:
    """
    Take in some dataframe and spit out all of the unique characters that make it up.
    This will be no duration and including duration. There will be a bool character to split itup.
    E.g.
    ---
    No Duration:
    ATGBAT -> {'A': 0, 'T': 0, 'G': 0, 'B': 0}
    ---
    Duration
    A*T#G#B*A*T^ -> {'A*': 0, 'T#': 0, 'T^': 0, 'G#': 0, 'B*': 0}
    """
    # Init dict
    alphabet_dict: dict = dict()
    row: pd.Series
    # iter through all entries
    for _, row in df.iterrows():
        if duration:
            for elem in list(map(''.join, zip(*[iter(row.string)]*2))):
                # if the elem isn't in the alphabet_dict keys
                if elem not in alphabet_dict:
                    # Init the key and value
                    alphabet_dict[elem] = 0
        else:
            # iter through each char in the string
            for elem in row.string:
                # if the elem isn't in the alphabet_dict keys
                if elem not in alphabet_dict:
                    # Init the key and value
                    alphabet_dict[elem] = 0
    alphabet_dict['-'] = 0
    return list(alphabet_dict.keys())


def calc_loglikelihood(markov: dict, samples: pd.DataFrame, nsteps: int,
                       duration: bool = False) -> np.array:
    """
    Take a sequence and generate the log likelihood of a sequence
    E.g.
    Log(P(s1))+log(P(s2|s1))+log(p(s3|s2))+log(p(s4|s3))
    """
    print(len(samples))
    if duration:
        samples = samples.drop(samples[samples.string.map(len)+2 <= nsteps*2+2].index).reset_index()
    else:
        samples = samples.drop(samples[samples.string.map(len)+2 <= nsteps+2].index).reset_index()
    # init the array to house the data
    output: np.array = np.zeros(shape=(len(samples),))
    # iter through the generated data
    row_idx: int
    row: pd.Series
    for row_idx, row in samples.iterrows():
        # grab mcmc sample
        sample: str = "!"+row.string+"-"
        likelihood: int = 0
        # if we are looking at duration
        if duration:
            # calculate the entire step we are taking
            fullstep: int = (nsteps+1)*2
            # iter through values
            for i in range(0, len(sample), 2):
                # decide the kmer based on if it is the beginng or end of the kmer
                kmer: str
                if i == 0: kmer = sample[i:i+fullstep+1]
                elif i >= len(sample)-fullstep: kmer = sample[i+1:i+fullstep+2]
                else: kmer = sample[i+1: i+fullstep+1]
                # now that we have the kmer we have to calculate the log likelihood
                if r'-' in kmer:
                    first: str = kmer[:-1]
                    prediction: str = kmer[-1]
                    # print(markov[first])
                    likelihood += np.log((markov[first][prediction]/np.array(list(markov[first].values())).sum()))
                    break
                else:
                    first: str = kmer[:-2]
                    prediction: str = kmer[-2:]
                likelihood += np.log((markov[first][prediction]/np.array(list(markov[first].values())).sum()))
        else:
            for i in range(0, len(sample), 1):
                kmer: str
                if i == 0: kmer = sample[i:i+nsteps+2]
                elif i >= len(sample)-nsteps: kmer = sample[i+1:i+nsteps+2]
                else: kmer = sample[i+1: i+nsteps+2]
                # now that we have the kmer we have to calculate the log likelihood
                if r'-' in kmer:
                    first: str = kmer[:-1]
                    prediction: str = kmer[-1]
                    likelihood += np.log((markov[first][prediction]/np.array(list(markov[first].values())).sum()))
                    break
                else:
                    first: str = kmer[:-1]
                    prediction: str = kmer[-1]
                likelihood += np.log((markov[first][prediction]/np.array(list(markov[first].values())).sum()))

        output[row_idx] = likelihood
    output[output == float('-inf')] = -1
    # plt.hist(output, bins=25)
    # plt.title(f'The Mean Log Likelihood is {np.mean(output)}')
    # plt.show()
    return output


def shuffle_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Take in a dataframe and generate a 80/20 split
    for train, test data
    """
    # shuffle data
    shuffled: pd.DataFrame = df.sample(frac=1, random_state=32).reset_index()
    # get lenght of Data
    N: int = len(shuffled)
    # get 80 twenty
    train_num: int = int(np.floor(N*.8))
    return (shuffled.loc[:train_num,['string']].reset_index(), shuffled.loc[train_num:, ['string']].reset_index())


