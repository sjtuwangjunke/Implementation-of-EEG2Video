import numpy as np
import os
from DE_PSD import DE_PSD
from tqdm import tqdm

fre = 200

# Extract DE or PSD features with a 1-second window, that is, for each 1-second EEG segment, we extract a DE or PSD feature.
# Input the shape of (7 * 40 * 5 * 62 * 2s*fre), meaning 7 blocks, 40 concepts, 5 video clips, 62 channels, and 2s*fre time-points.
# Output the DE or PSD feature with (7 * 40 * 5 * 2 * 62 * 5), the last 5 indicates the frequency bands' number.


def get_files_names_in_directory(directory):
    files_names = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            files_names.append(filename)
    return files_names

sub_list = get_files_names_in_directory("/home/drink/SEED-DV/Segmented_Rawf_200Hz_2s/")

for subname in sub_list:

    loaded_data = np.load('/home/drink/SEED-DV/Segmented_Rawf_200Hz_2s/' + subname)
    # (7 * 40 * 5 * 62 * 2*fre)

    print("Successfully loaded .npy file.")
    print("Loaded data:")

    DE_data = np.empty((0, 40, 5, 2, 62, 5))
    PSD_data = np.empty((0, 40, 5, 2, 62, 5))

    for block_id in range(7):
        print("block: ", block_id)
        now_data = loaded_data[block_id]
        de_block_data = np.empty((0, 5, 2, 62, 5))
        psd_block_data = np.empty((0, 5, 2, 62, 5))
        for class_id in tqdm(range(40)):
            de_class_data = np.empty((0, 2, 62, 5))
            psd_class_data = np.empty((0, 2, 62, 5))
            for i in range(5):
                de1, psd1 = DE_PSD(now_data[class_id, i, :, :200].reshape(62, fre), fre, 1)
                de2, psd2 = DE_PSD(now_data[class_id, i, :, 200:].reshape(62, fre), fre, 1)
                de_class_data = np.concatenate((de_class_data, np.concatenate((de1.reshape(1, 62, 5), de2.reshape(1, 62, 5))).reshape(1, 2, 62, 5)))
                psd_class_data = np.concatenate((psd_class_data, np.concatenate((psd1.reshape(1, 62, 5), psd2.reshape(1, 62, 5))).reshape(1, 2, 62, 5)))
            de_block_data = np.concatenate((de_block_data, de_class_data.reshape(1, 5, 2, 62, 5)))
            psd_block_data = np.concatenate((psd_block_data, psd_class_data.reshape(1, 5, 2, 62, 5)))
        DE_data = np.concatenate((DE_data, de_block_data.reshape(1, 40, 5, 2, 62, 5)))
        PSD_data = np.concatenate((PSD_data, psd_block_data.reshape(1, 40, 5, 2, 62, 5)))

    np.save("/home/drink/SEED-DV/DE_1per1s/" + subname, DE_data)
    np.save("/home/drink/SEED-DV/PSD_1per1s/" + subname, PSD_data)