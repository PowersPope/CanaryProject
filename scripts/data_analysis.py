import os
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
from transition_matrix import transition_matrix, generate_occurence_dict
from functions import duration_to_str


def combine_all_data_to_one_df(files: list, path: str) -> pd.DataFrame:
    """
    Take all of the files and put them into one dataframe.
    We do not care about duration. As all we are looking for is Character Frequency
    It may be important to add duration. Though that can be easily done from the functions.py
    script
    """
    # Create a housing df
    df: pd.DataFrame = pd.DataFrame()
    #Iter through the files from the list of files
    for file in files:
        # if it doesn't have clean_ at the beginning, then it isn't in the right format
        if r'clean_' not in file: continue
        # Make the path for the file
        file_path: str = os.path.join(path, file)
        # Generate a df from the file
        temp_df: pd.DataFrame = pd.read_csv(file_path)
        # concat the strings together
        df: pd.DataFrame = pd.concat([df,temp_df.string],ignore_index=True)
    df.rename(columns={0:'string'}, inplace=True)
    # if r'!' not in df.iloc[0,0]: df['string'] = df['string'].apply(lambda x: '!'+x+'-')
    return df

def gen_occ(data: pd.DataFrame) -> tuple[dict, dict, int]:
    """
    Go through the list of data and pull out each element and add it to a dictionary
    incrementing the amount of times that the element is seen.
    Also, generate a length occurence dictionary for comparison as well.
    """
    # init dicts to house the function returning data
    output_occ_freq: dict = dict()
    output_len_freq: dict = dict()
    # iter through the data
    row: pd.Series
    i: int
    for i, row in data.iterrows():
        # split string into a list of chars
        split_phrase: list = list(row.string)
        # Iter through the elements in the list
        ele: str
        for ele in split_phrase:
            # If in the dict already. Increment
            if ele in output_occ_freq:
                output_occ_freq[ele] += 1
            # If not in the dict. Initialize
            else:
                output_occ_freq[ele] = 1
        # grab the length of the phrase
        len_split_phrase: int = len(split_phrase)
        # If the length has been seen then. Increment
        if len_split_phrase in output_len_freq:
            output_len_freq[len_split_phrase] += 1
        # If the length hasn't been seen. Initialize
        else:
            output_len_freq[len_split_phrase] = 1
    return (output_occ_freq, output_len_freq, i)

def graph_data(occ_dict: dict, length_dict: dict,
               gen_occ_dict: dict = None, gen_length_dict: dict = None,
               final: bool = False, total1: int = 0, total2: int = 0,
               duration: bool = False) -> None:
    """
    Display both the char_freq dict as a bar chart and the length as a bar chart
    """
    if final:
        # pre procees values for dataset
        occ_dict['!'] = total1
        occ_dict['-'] = total1
        occ_values: np.array = np.array(list(occ_dict.values())) / total1
        length_values: np.array = np.array(list(length_dict.values())) / total1
        # pre process values for generated dataset
        if duration:
            gen_length_keys: np.array = np.array(list(gen_length_dict.keys())) / 2
        else:
            gen_length_keys: np.array = np.array(list(gen_length_dict.keys()))
        gen_occ_values: np.array = np.array(list(gen_occ_dict.values())) / total2
        gen_length_values: np.array = np.array(list(gen_length_dict.values())) / total2
        # graph everything
        fig,ax = plt.subplots(3,2)
        ax[0][0].bar(x=list(occ_dict.keys()), height=occ_values)
        ax[2][0].set_xlabel('Chars in Phrases')
        ax[1][0].set_ylabel('Percentage of Occurences')
        ax[0][1].bar(x=list(length_dict.keys()), height=length_values)
        ax[2][1].set_xlabel('Length of Phrases')
        ax[1][0].bar(x=list(gen_occ_dict.keys()), height=gen_occ_values)
        ax[1][1].bar(x=gen_length_keys, height=gen_length_values)
        # Third row
        ax[2][0].bar(x=list(occ_dict.keys()), height=occ_values, color='green', alpha=0.5,
                     label="Natural Data")
        ax[2][0].bar(x=list(gen_occ_dict.keys()), height=gen_occ_values, color="red", alpha=0.5,
                     label="Generated Data")
        ax[2][1].bar(x=list(length_dict.keys()), height=length_values, color="green", alpha=0.5,
                     label="Natural Data")
        ax[2][1].bar(x=gen_length_keys, height=gen_length_values, color="red", alpha=0.5,
                     label="Generated Data")
        fig.suptitle(f'Actual Dataset vs. Generated Dataset {total2} sequences sampled\nN-step of 3')
        plt.legend(loc='upper right', labels=['Natural Data', 'Generated Data'])
        plt.tight_layout()
        plt.show()
    else:
        # pre procees values
        occ_values: np.array = np.array(list(occ_dict.values())) / total1
        length_values: np.array = np.array(list(length_dict.values())) / total1
        fig,ax = plt.subplots(1,2)
        ax[0].bar(x=list(occ_dict.keys()), height=occ_values)
        ax[0].set_xlabel('Chars in Phrases')
        ax[0].set_ylabel('Percentage of Occurences')
        ax[1].bar(x=list(length_dict.keys()), height=length_values)
        ax[1].set_xlabel('Length of Phrases')
        fig.suptitle('Dataset Attributes')
        plt.tight_layout()
        plt.show()
    return None

def generate_seqs(transition_mat: np.array, alphabet_list: set, N: int, nsteps: int) -> pd.DataFrame:
    """
    Generate a library of generated samples from the transition matrix.
    These will be used to test
    """
    # house the strings in a list
    gen_data: list = list()
    print(alphabet_list)
    # While loop to generate a bunch of strings until N numbers are reached
    i: int = 0
    while i < N:
        # create sample
        #### IF USING NSTEP == 1 and Duration then this needs to be off
        #### Update this later
        if nsteps == 1:
            sample: str = "!"
            current_char: str = sample
        else:
            starting_token: list = [x for x in alphabet_list if r'!' in x]
            random_index: int = np.random.randint(0,len(starting_token)-1,1)[0]
            sample: str = starting_token[random_index]
            current_char: str = sample
        while r"-" not in current_char:
            idx_item: int = alphabet_list.index(current_char)
            # print(alphabet_list.index(current_char))
            # print(alphabet_list[idx_item])
            # print(transition_mat[idx_item,:].sum())
            try:
                next_char: str = np.random.choice(alphabet_list,
                                                  p=transition_mat[idx_item])
            except:
                next_char: str = "-"
            sample: str = sample + next_char
            current_char: str = next_char
            # print(current_char)
            # print(sample)
        gen_data.append(sample)
        i += 1
    gen_df: pd.DataFrame = pd.DataFrame(gen_data, columns=["string"])
    print(len(gen_df))
    return gen_df


def main()->int:
    """
    Take in some data.
    Generate a distribution of Letters that is seen within the data.
    Then generate 1000 sequences using our transition matrix and see if the data graphs are similar.
    """

    parser = argparse.ArgumentParser(description="Provide a path to the set of files you want to \
                                        generate comparisons for")
    parser.add_argument("--path", type=str, help="Path to data folder")
    # parser.add_argument("-d", "--duration", action="store_true",
                        # help="Include duration")
    parser.add_argument("--duration", default=False, type=bool, help="Include duration")
    parser.add_argument("-n", "--nstruct", type=int, default=1000, help="Numer of samples you want generated \
                        (default=1000)")
    parser.add_argument("--nsteps", type=int, default=1, help="Number of steps for the Markov Model \
                        (default=1)")
    args = parser.parse_args()

    files = os.listdir(args.path)
    # Load in all of the data
    actual_df: pd.DataFrame = combine_all_data_to_one_df(files, path=args.path)

    # get occurences of the Letters within the data
    char_freq, length_freq, N1 = gen_occ(actual_df)

    # graph the occurences (% of characters that show up) could also graph length of words
    # graph_data(char_freq, length_freq)

    # if duration == True
    if args.duration:
        file_clean: str
        for file_clean in files:
            if r'clean_' not in file_clean: continue
            file: str = os.path.join(args.path, file_clean)
            # print('New File:', file)
            # init dict to hold new dur strings
            # string_dict: dict = dict()
            string_dict: list = list()
            # with open(file_one, 'r') as f:
            with open(file, 'r') as f:
                f.readline()
                for line in f:
                    line = line.rstrip('\n')
                    # print(line)
                    out: str = duration_to_str(line)
                    # print(out)
                    string_dict.append(out)
        actual_df: pd.DataFrame = pd.DataFrame(string_dict, columns=['string'])

        # Generate transition matrix
        init_dict: dict = dict()
        init_set: set = set()
        occ_dict, alphabet_set, N_token = generate_occurence_dict(actual_df,
                                                                  duration_included=args.duration,
                                                                  nsteps=args.nsteps,
                                                                  occ_dict=init_dict,
                                                                  set_obj=init_set)
        trans_mat, alphabet = transition_matrix(occ_dict, N_token, alphabet_set, duration=args.duration,
                                      step=args.nsteps)
    else:
        # Generate transition matrix
        init_dict: dict = dict()
        init_set: set = set()
        occ_dict, alphabet_set, N_token = generate_occurence_dict(actual_df,
                                                                  duration_included=args.duration,
                                                                  nsteps=args.nsteps,
                                                                  occ_dict=init_dict,
                                                                  set_obj=init_set)
        trans_mat, alphabet = transition_matrix(occ_dict, N_token, alphabet_set, duration=args.duration,
                                      step=args.nsteps)

    # sample 1000+ sequences from this
    num_samples: int = 10000
    sample_df: pd.DataFrame = generate_seqs(trans_mat, alphabet, num_samples, args.nsteps) # args.nstruct taken out

    # get occurences of the letters within the simulated data
    test_char_freq, test_length_freq, _ = gen_occ(sample_df)
    if args.duration:
        del test_char_freq["*"]
        del test_char_freq["#"]
        del test_char_freq["^"]

    # graph the occurences (could also graph length of words)
    graph_data(char_freq, length_freq, test_char_freq, test_length_freq, final=True,
               total1=N1, total2=num_samples, duration=args.duration) # args.nsturct taken out

    return 0

main()
