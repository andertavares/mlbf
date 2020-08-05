

import os
import glob

def clean_finished(dir):
    """
    Utility to clean finished .cnf files so that the experiment can
    continue with the remaiining. By 'finished' I mean with the dataset (.pkl.gz) present
    :param dir:
    :return:
    """

    for f in os.listdir(dir):
        if len(glob.glob(f'{dir}/{f}*.pkl.gz')) > 0:
            print(f, 'exists, removing')
            os.unlink(f'{dir}/{f}')