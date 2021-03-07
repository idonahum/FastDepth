import argparse
import io
import os
from tempfile import gettempdir
import urllib.request


from tqdm import tqdm
import tarfile


DATASET_URL = 'http://datasets.lids.mit.edu/fastdepth/data/nyudepthv2.tar.gz'


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url, output_filepath, display_progressbar=False):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1],
                             disable=not display_progressbar) as t:
        urllib.request.urlretrieve(url,
                                   filename=output_filepath,
                                   reporthook=t.update_to)



if __name__ == '__main__':
    # argument parser
    parser = argparse.ArgumentParser(
        description='Prepare NYUv2 dataset for depth estimation.')
    parser.add_argument('output_path', type=str,
                        help='path where to store dataset')
    args = parser.parse_args()

    # preprocess args and expand user
    output_path = os.path.expanduser(args.output_path)
    tar_filepath = os.path.join(gettempdir(), 'nyudepthv2.tar.gz')

    # download mat file if mat_filepath does not exist
    if not os.path.exists(tar_filepath):
        print(f"Downloading tar to: `{tar_filepath}`")
        download_file(DATASET_URL, tar_filepath, display_progressbar=True)

    # create output path if not exist
    os.makedirs(output_path, exist_ok=True)

    # load mat file and extract images
    print(f"Loading tar file: `{tar_filepath}`")
    my_tar = tarfile.open(tar_filepath)
    my_tar.extractall(args.output_path) # specify which folder to extract to
    my_tar.close()
    
    # remove downloaded file
    print(f"Removing downloaded mat file: `{tar_filepath}`")
    os.remove(tar_filepath)
