''' 
Usage: to download dataset GSO
python download_webdataset.py --dataset GSO
Usage: (for debug) to download dataset GSO with shards 0 and 1
python download_webdataset.py --dataset GSO --shards 0 1
'''
import os
import subprocess
import tarfile
import argparse
import shutil

assert os.getcwd().split('/')[-1] == 'Data', "Current working directory is not 'Data'"

failures = []

def is_correct_tar(file_path):
    try:
        with tarfile.open(file_path) as file:
            return True
    except:
        return False

def download_url_tar(url, save_path, extract=False):
    # Check if the file already exists and is a valid tar file
    if os.path.exists(save_path) and is_correct_tar(save_path):
        print(f"{save_path} already exists and is a valid tar file.")
    else:
        # Informing the user about the download initiation
        print("Starting download. This may take a while due to the file size and download restrictions.")
        print("The file will be saved at: ", os.path.abspath(save_path))
        # Removed the -c option as the server does not support resuming downloads
        command = f"wget '{url}' -O '{save_path}'"
        os.system(command)
    if extract:
        save_folder = save_path[:-4]
        if os.path.exists(save_folder):
            print(f"Deleting existing folder: {save_folder}")
            shutil.rmtree(save_folder)
        print("Extracting the downloaded file.")
        os.mkdir(save_folder)
        try:
            subprocess.run(f"tar -xvf {save_path} -C {save_folder}", shell=True, check=True, timeout=300)
        except subprocess.CalledProcessError as e:
            print(f"Extraction failed: {e}")
            failures.append(save_path)
        # Delete the original tar file
        os.remove(save_path)
        print(f"Original tar file {save_path} deleted.")

# Create an argument parser
parser = argparse.ArgumentParser(description='Download webdataset files.')

# Add an argument for the dataset type
parser.add_argument('--dataset', default='GSO', type=str, choices=['GSO', 'shapenet'], help='Type of dataset to download')
parser.add_argument('--shards', nargs='+', default=[] , type=int, help='List of shard IDs to download')
parser.add_argument('--extract', action='store_true', help='Extract the downloaded files')

# Parse the command line arguments
args = parser.parse_args()

SHARD_SIZE = 1040
if args.dataset == "GSO":
    base_url = 'https://huggingface.co/datasets/bop-benchmark/datasets/resolve/main/MegaPose-GSO/shard-{SHARD_ID:06d}.tar'
    base_path = 'MegaPose-Training-Data/MegaPose-GSO/train_pbr_web/shard-{SHARD_ID:06d}.tar'
elif args.dataset == "shapenet":
    base_url = 'https://huggingface.co/datasets/bop-benchmark/datasets/resolve/main/MegaPose-ShapeNetCore/shard-{SHARD_ID:06d}.tar'
    base_path = 'MegaPose-Training-Data/MegaPose-ShapeNetCore/train_pbr_web/shard-{SHARD_ID:06d}.tar'

if len(args.shards) == 0:
    SHARD_LIST = range(SHARD_SIZE)
else:
    SHARD_LIST = args.shards

for SHARD_ID in SHARD_LIST:
    url = base_url.format(SHARD_ID=SHARD_ID)
    save_path = base_path.format(SHARD_ID=SHARD_ID)
    download_url_tar(url, save_path, args.extract)

print("failures:", failures)
# Note that 2024-7-25 the GSO [101, 102, 108, 114] and shapenet [701] are not correct tar files.