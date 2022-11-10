###############################
# Script made by Andrew Powers
# October 2022
###############################


#import funcitons
import numpy as np
import pandas as pd
import argparse
import os
from functions import duration_to_str, split_token, nstep

def arg_parse() -> object:
    """ Pass in arguments into the script"""
    parser = argparse.ArgumentParser(description="In order for this file to work a file dir path is \
            needed. It will calculate the transition matrix for them.")
    parser.add_argument('-p', '--path', help='Path to directory with .csv_clean files. Must be path',
                        type=str, required=True)
    parser.add_argument('-d', '--duration', help='Include Duration? True/False', type=bool,
                        default=False)
    parser.add_argument('--nsteps', type=int, default=1, help="Specify the Markov Step \
            default is 1")
    return parser.parse_args()

# get arguments
args = arg_parse()



def generate_occurence_dict(data: pd.DataFrame, duration_included: bool, nsteps: int) -> tuple[dict,set,int]:
    """
    Input a DataFrame with columns [Song(String), Duration(Int)]
    This function will generate an occurence dict for every unique token that is
    found within the Song column.
    Also, append prefix start and postfix end tokens to each string
    Ouptut dict with counts
    """

    # Init occurence dict
    occ_dict: dict = {}

    # create set
    set_obj: set = set()

    # Pull out string column
    string_token: pd.Series = data.loc[:, "string"]

    # Apply pre and post tokens to strings in vector
    # string_token: pd.Series = string_column.apply(lambda x: f"!{x}-")

    # total number of Tokens
    N: int = 0
    M: int = nstep(N=nsteps, duration=duration_included)
    _, num_splits = split_token(nstep=nsteps, duration=duration_included)
    # Loop through elements in vector
    i: int
    for i in string_token.index:
        #row string
        element: str = string_token[i]
        # check to make sure that the lines are the correct length if not discard them
        if len(element) > nsteps+2:
            if nsteps == 1:
                element = '!' + element + '-'
                add_set: list = list(element)
            else:
                add_set: list = list(map(''.join, zip(*[iter(element)]*num_splits)))
                add_set[0] = '!' + add_set[0]
                add_set[-1] = add_set[-1] + '-'
            # add to set
            set_obj.update(add_set)
            #kmerize the read
            if args.duration:
                skips: int = nsteps + 1
            else:
                skips: int = 1
            # run through the string skipping by skips
            n: int
            for n in range(0, len(element),skips):
                # running total of tokens
                N += 1
                # generate Kmer of size 4 to account for duration added char
                k: int = n + M
                # if duration_included:
                    # k: int = n+4
                # else:
                    # # gen Kmer size of 2
                    # k: int = n+2
                if (n==0) and (args.duration):
                    token: str = '!'+element[n:k]
                elif (k == len(element)) and (args.duration):
                    token: str = element[n:k]+'-'
                else:
                    token: str = element[n:k]
                # check to see if the token is in the dict and is the last token in string
                if k == len(element) and token in occ_dict.keys():
                    occ_dict[token] +=1
                    break
                # check to see if the token is not in dict and the last token in string
                elif k == len(element) and token not in occ_dict.keys():
                    occ_dict[token] = 1
                    break
                # check to see if the token is in the dict
                elif token in occ_dict.keys():
                    occ_dict[token] +=1
                # token is not in the dict so add it
                else:
                    occ_dict[token] = 1
        else:
            break

    return occ_dict, set_obj, N


def transition_matrix(occ_dict: dict, num_tokens: int, set_alphabet: set, duration: bool,
                      step: int) -> np.array:
    """
    Take in the the output from generate_occurence_dict and use that to create a transition matrix.
    We need make a matrix that is MxM M is the lenght of the set_alphabet varialbe.
    We can then input the probabilites into the cell that correspondes with the components the
    token makes up.
    """
    # First lets get the occurences for each Token in an array
    occ_array: np.array = np.array(list(occ_dict.values()))

    # generate MxM matrix of zeros
    M: int = len(set_alphabet)
    transition_mat: np.array = np.zeros(shape=(M,M))

    # create list of the set to give it an index
    alphabet_list: list = list(set_alphabet)

    dict_keys: list = list(occ_dict.keys())


    # go through each element and now add in the probabilities
    num: int
    for num in range(0,len(occ_dict),1):
        # grab the token
        tok: str = dict_keys[num]
        # grab the occurence associated with that token
        occurence: int = occ_array[num]
        K, J = split_token(nstep=step, duration=duration)
        if duration:
            if r'!' in tok:
                split_tok: list[str] = [tok[0:K],tok[K:]]
            elif r'-' in tok:
                split_tok: list[str] = [tok[0:K-1],tok[K-1:]]
            else:
                # split the token
                split_tok: list[str] = list(map(''.join, zip(*[iter(tok)]*J)))
            if r'' in split_tok:
                break
        else:
            split_tok: list[str] = [tok[0], tok[1]]
        # grab the index of each of the chars in the token
        row_index: int = alphabet_list.index(split_tok[0])
        col_index: int = alphabet_list.index(split_tok[1])
        # add occurence to the transition matrix
        transition_mat[row_index,col_index]: int = occurence

    # we want to suppress the warning as we know some may have nans and be invalid
    np.seterr(invalid='ignore')
    # now lets get the percentage for each row and column
    row_sums: np.array = transition_mat.sum(axis=1)
    transition_mat_prob: np.array = transition_mat / row_sums[:, np.newaxis]
    # if any rows have nothing then get rid of the nans and convert to 0
    transition_mat_prob[np.isnan(transition_mat_prob)] = 0


    return transition_mat_prob, alphabet_list

def main() -> int:
    """
    Run processes to generate a transition matrix that will be used for a
    One Step Markov Model for song-string generation
    """

    # get list of files in path
    files: list = os.listdir(args.path)
    files_clean: list = [x for x in files if r'clean_' in x]

    # test right now 
    file_one = os.path.join(args.path, files_clean[0])

    #######
    ### I need to add functionality to load in all files into one df, however for testing I am only
    ### focused on one file currently
    #####

    if args.duration:
        # init dict to hold new dur strings
        string_dict: dict = dict()
        with open(file_one, 'r') as f:
            f.readline()
            for line in f:
                line = line.rstrip('\n')
                out: str = duration_to_str(line)
                if out in string_dict:
                    string_dict[out] += 1
                else:
                    string_dict[out] = 1

        # df_data
        df_data: pd.DataFrame = pd.DataFrame.from_dict(string_dict, orient='index')
        df_data.reset_index(inplace=True)
        df_data = df_data.rename(columns={'index': 'string', 0:'count'})
    else:
        # load in data
        df_data: pd.DataFrame = pd.read_csv(file_one)

    #convert to 

    # generate the frequency dict and alphabet
    count_dict, alphabet, N_token = generate_occurence_dict(df_data,
                                                            duration_included=args.duration,
                                                            nsteps=args.nsteps)

    # print(count_dict)
    # print(alphabet)
    # print(N_token)

    # generate the transition matrix
    trans_mat, alph_list = transition_matrix(count_dict, N_token, alphabet, duration=args.duration,
                                             step=args.nsteps)

    print(trans_mat)
    print(alph_list)
    # occ_dict1: dict[str, int] = {'!A': 120, '!F': 100, '!S': 30, 'SA':30, 'FA': 20, 'FS':100,
                                 # 'S-': 100, 'F-': 20, 'AF': 30, 'AS': 30, 'AA':10}
    # alphabet1: dict[str] = set(['!', '-', 'A', 'F', 'S'])


    # print(transition_matrix(occ_dict1, len(alphabet1), alphabet1, duration=False))


    return 0

#### Run main function
main()

