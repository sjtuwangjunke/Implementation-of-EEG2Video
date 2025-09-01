import numpy as np
import os
from tqdm import tqdm

# segment a EEG data numpy array with the shape of (7 * 62 * 520s*fre) into 2-sec EEG segments
# segment it into a new array with the shape of (7 * 40 * 5 * 62 * 2s*fre), 
# meaning 7 blocks, 40 concepts, 5 video clips, 62 channels, and 2s*fre time-points.

fre = 200

def get_files_names_in_directory(directory):
    files_names = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            files_names.append(filename)
    return files_names

sub_list = get_files_names_in_directory("/home/drink/SEED-DV/EEG/")

for subname in sub_list:
    npydata = np.load('/home/drink/SEED-DV/EEG/' + subname)

    save_data = np.empty((0, 40, 5, 62, 2*fre))

    for block_id in range(7):
        print("block: ", block_id)
        now_data = npydata[block_id]
        l = 0
        block_data = np.empty((0, 5, 62, 2*fre))
        for class_id in tqdm(range(40)):
            l += (3 * fre)
            class_data = np.empty((0, 62, 2*fre))
            for i in range(5):
                class_data = np.concatenate((class_data, now_data[:, l : l + 2*fre].reshape(1, 62, 2*fre)))
                l += (2 * fre)
            block_data = np.concatenate((block_data, class_data.reshape(1, 5, 62, 2*fre)))
        save_data = np.concatenate((save_data, block_data.reshape(1, 40, 5, 62, 2*fre)))

    np.save('/home/drink/SEED-DV/Segmented_Rawf_200Hz_2s/' + subname, save_data)