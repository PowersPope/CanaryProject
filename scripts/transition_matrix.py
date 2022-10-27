###############################
# Script made by Andrew Powers
# October 2022
###############################


#import funcitons
import numpy as np
import pandas as pd
import argparse
import os,sys

def arg_parse() -> object:
    """ Pass in arguments into the script"""
    parser = argparser.ArgumentParser(description="In order for this file to work a file dir path is \
            needed. It will calculate the transition matrix for them.")
    parser.add_argument('-p', '--path', help='Path to directory with .csv_clean files. Must be path',
                        type=str, required=True)
    return parser.parse_args()

# get arguments
args = arg_parse()



def generate_occurence_dict(data: pd.DataFrame) -> tuple[dict,set,int]:
    """
    Input a DataFrame with columns [Song(String), Duration(Int)]
    This function will generate an occurence dict for every unique token that is
    found within the Song column.
    Also, append prefix start and postfix end tokens to each string
    Ouptut dict with counts
    """

    # Init occurence dict
    occ_dict: dict = dict()

    # create set
    set_obj: set = set()

    # Pull out string column
    string_column: pd.Series = data.loc[:, "string"]

    # Apply pre and post tokens to strings in vector
    string_token: pd.Series = string_column.apply(lambda x: f"!{x}-")

    # total number of Tokens
    N: int = 0
    # Loop through elements in vector
    i: int
    for i in string_token.index:
        #row string
        element: str = string_token[i]
        # add to set
        set_obj.update(list(element))
        #kmerize the read
        n: int
        for n in range(0, len(element)):
            # running total of token
            N += 1
            # generate Kmer of size 3
            k: int = n+3
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

    return occ_dict, set_obj, N


def transition_matrix(occ_dict: dict, num_tokens: int, set_alphabet: set) -> np.array:
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
        # split the token
        split_tok: list = list(tok)
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
    files: list = os.dirlist(args.path)
    files_clean: list = [x for x in files if r'_clean' in x]

    # test right now 
    file_one = os.path.join(args.path, files_clean[0])

    # load in data
    # df_data: pd.DataFrame = pd.read_csv(file_one)
    

    # generate the frequency dict and alphabet
    count_dict, alphabet, N_token = generate_occurence_dict(df_data)

    # generate the transition matrix
    trans_mat, alph_list = transition_matrix(count_dict, N_token, alphabet)

    print(trans_mat)
    print(alph_list)


    return 0

#### Run main function
main()

