import fire

import dataset


def find_bad_datasets(*files):
    """
    Finds which datasets are causing error on loading
    :param files: list of (.pkl.gz) files to check
    """
    print(f'Scanning {len(files)} files.')
    for f in files:
        try:
            d = dataset.get_dataset(f, None)
        except AttributeError:
            print(f'{f} triggered an exception')


if __name__ == '__main__':
    fire.Fire(find_bad_datasets)
