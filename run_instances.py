import os
import fire
import main
import tarfile


def run(instances, output_file='out.csv', extraction_point='/tmp/satinstances'):
    """
    Extracts all files in a tar.gz file and runs the experiment for each one of them
    :param instances: .tar.gz file containing the .cnf instances
    :param output_file: path to write results to (csv format)
    :param extraction_point: point to extract the cnf instances
    :return:
    """
    # creates the extraction point if it does not exist
    os.makedirs(extraction_point, exist_ok=True)

    # extracts the instances
    print(f'Extracting contents of {instances} to {extraction_point}')
    with tarfile.open(instances) as tf:
        tf.extractall(extraction_point)

    # run each instance in the extraction point
    for root, dirs, files in os.walk(extraction_point):
        print(f'{len(files)} files are at {extraction_point}.')

        for f in files:
            print(f'Running {f}...')
            main.main(os.path.join(root, f), output_file)
            print()  # just a newline
        print("Finished all instances.")


if __name__ == '__main__':
    fire.Fire(run)
