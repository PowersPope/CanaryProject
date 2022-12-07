import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from functions import duration_to_str, gen_alphabet, generate_initial_occurences, shuffle_data, calc_loglikelihood

def main() -> int:
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
    parser.add_argument("--test", type=bool, default=False, help="Running a test. Set to True.")
    args = parser.parse_args()

    # load files into a list
    files: list = os.listdir(args.path)

    # init log_dict to house likelihoods
    log_dict: dict = dict()

    test_conditions: list = [1, 2, 3, 4, 5]
    duration: list = [True, False]

    # itr through files
    file: str
    for d in duration:
        print(f'==> Duration of {d} being tested now')
        for nstep in test_conditions:
            print(f'==> Nstep of {nstep} being tested')
            for file in files:
                if r'clean_' not in file: continue
                # init the holding string dict for all data
                string_dict: list = list()
                # read the file into a and augment it
                print(f'==> FILE {file} computing')
                with open(file, 'r') as f:
                    line: str
                    f.readline()
                    for line in f:
                        # strip the newline off the end
                        line = line.rstrip('\n')
                        out: str
                        if d:
                            # convert the string to a duration format
                            out = duration_to_str(line)
                        else:
                            out = line
                        string_dict.append(out)
                actual_df: pd.DataFrame = pd.DataFrame(string_dict, columns=['string'])

                # train test split
                train_df, test_df = shuffle_data(actual_df)

                # calc necessary alphabet and occurence dictionary
                alphabet_dict: dict = gen_alphabet(actual_df, d)
                occ_dict: dict = generate_initial_occurences(actual_df, alphabet_dict,
                                                             nstep,
                                                             d)

                # calc loglikelihood
                loglikelihood: np.array = calc_loglikelihood(occ_dict, test_df.loc[:,['string']], nstep,
                                                          d)

                log_dict[file.rstrip('.csv').strip('clean_')+'-'+str(nstep)+'-'+str(d)] = loglikelihood

    x = list()
    y = list()
    c = list()
    colors = {1:'red', 2:'blue', 3:'green', 4: 'purple', 5: 'orange'}
    for k, v in log_dict.items():
        x.append(k)
        y.append(v.mean())
        data = k.split('-')
        nstep = int(data[1])
        duration = data[2]
        c.append(colors[nstep])

    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='red', lw=2),
                    Line2D([0], [0], color='blue', lw=2),
                    Line2D([0], [0], color='green', lw=2),
                    Line2D([0], [0], color='purple', lw=2),
                    Line2D([0], [0], color='orange', lw=2)]


    fig, ax = plt.subplots()
    ax.barh(x,y,color=c)
    ax.set_xlabel('Negative LogLikelihood')
    ax.set_title('Iterative Models run on Canary BirdSong')
    ax.tick_params(axis='y', which='minor', labelsize=1)
    plt.legend(custom_lines, ['Nstep1', 'Nstep2', 'Nstep3', 'Nstep4', 'Nstep5'])
    plt.show()












    return 0

main()
