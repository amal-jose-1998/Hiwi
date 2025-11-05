import pandas as pd
from typing import List, Tuple
import numpy as np
from glob import glob
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

base_path = r"/mnt/nas/uncompressed_data/real_01/Keyence-Messung/"

def process_metadata(path) -> dict:
    """ Process the path to get the corresponding metadata.  """

    metadata={'BF':-1, 'fine':None, "medium":None, "coarse":None, "id":None}

    raw_metadata = Path(path).name.split('_')
    raw_metadata = list(filter(None, raw_metadata))

    metadata["part"] = None
    
    for key in metadata:
        if key == "part":
            continue
        try:
            idx = raw_metadata.index(key)

            if key == "BF":
                metadata[key] = raw_metadata[idx + 1]
            else:
                metadata[key] = 1
        except ValueError:
            metadata[key] = 0
        except IndexError:
            print(f"Error while parsing: {path}")
    return metadata


def quick_get_pairs(main_folder_path: str) -> List[Tuple[str, str]]:
    """Compact version for quick access to pairs"""
    p = Path(main_folder_path)
    dirs = sorted(p.iterdir(), key=lambda x: int(x.name.split('_')[0]))
    return [(str(dirs[i]/"OP_10"), str(dirs[i+2]/"OP_20")) 
            for i in range(len(dirs)-1) 
            if (dirs[i]/"OP_10").exists() and (dirs[i+2]/"OP_20").exists()]

def main():

    metadata_df = pd.DataFrame()
    folders = [f for f in sorted(glob(base_path + "*/"))]

    for f in tqdm(folders):
        metadata = process_metadata(f)
        # The Operation 10 and the corresponding Operation 20 are not in the exact same directory, therefore this mappong occurs
        pairs = quick_get_pairs(f)

        for f_10, f_20 in pairs:
            print(f_10) # Operation 10 directory
            print(f_20) # Operation 20 directory

            # Display for OP10
            z_data = np.load(f"{f_10}/z_data.npy")
            lumi_data = np.load(f"{f_10}/lumi_data.npy")
            print(f"\nShape of z data (OP10): {z_data.shape}\nShape of lumi data (OP10): {lumi_data.shape}")

            z_data = np.load(f"{f_20}/z_data.npy")
            lumi_data = np.load(f"{f_20}/lumi_data.npy")
            print(f"\nShape of z data (OP20): {z_data.shape}\nShape of lumi data (OP20): {lumi_data.shape}")
            break 
        metadata_df = pd.concat([metadata_df, pd.DataFrame([metadata])], ignore_index=True)
        break

    print("\nMetadata csv:")
    print(metadata_df.head())
    metadata_df.to_csv("metadata_laser.csv", index=False)

if __name__ == "__main__":
    main()
    