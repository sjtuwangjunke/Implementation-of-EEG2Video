import numpy as np
from DE_PSD import DE_PSD
from tqdm import tqdm
import os
# Extract DE or PSD features with a 2-second window, that is, for each 2-second EEG segment, we extract a DE or PSD feature.
# Input the shape of (7 * 40 * 5 * 62 * 2s*fre), meaning 7 blocks, 40 concepts, 5 video clips, 62 channels, and 2s*fre time-points.
# Output the DE or PSD feature with (7 * 40 * 5 * 62 * 5), the last 5 indicates the frequency bands' number.

fre = 200

def get_files_names_in_directory(directory):
    files_names = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            files_names.append(filename)
    return files_names

sub_list = get_files_names_in_directory("SEED-DV/Segmented_Rawf_200Hz_2s/")

for subname in sub_list:

    loaded_data = np.load('SEED-DV/Segmented_Rawf_200Hz_2s/' + subname)
    # (7 * 40 * 5 * 62 * 2*fre)

    print("Successfully loaded .npy file.")
    print("Loaded data:")

    DE_data = np.empty((0, 40, 5, 62, 5))
    PSD_data = np.empty((0, 40, 5, 62, 5))

    for block_id in range(7):
        print("block: ", block_id)
        now_data = loaded_data[block_id]
        de_block_data = np.empty((0, 5, 62, 5))
        psd_block_data = np.empty((0, 5, 62, 5))
        for class_id in tqdm(range(40)):
            de_class_data = np.empty((0, 62, 5))
            psd_class_data = np.empty((0, 62, 5))
            for i in range(5):
                de, psd = DE_PSD(now_data[class_id, i, :, :].reshape(62, 2*fre), fre, 2)
                de_class_data = np.concatenate((de_class_data, de.reshape(1, 62, 5)))
                psd_class_data = np.concatenate((psd_class_data, psd.reshape(1, 62, 5)))
            de_block_data = np.concatenate((de_block_data, de_class_data.reshape(1, 5, 62, 5)))
            psd_block_data = np.concatenate((psd_block_data, psd_class_data.reshape(1, 5, 62, 5)))
        DE_data = np.concatenate((DE_data, de_block_data.reshape(1, 40, 5, 62, 5)))
        PSD_data = np.concatenate((PSD_data, psd_block_data.reshape(1, 40, 5, 62, 5)))

    np.save("SEED-DV/DE_1per2s/" + subname, DE_data)
    np.save("SEED-DV/PSD_1per2s/" + subname, PSD_data)
