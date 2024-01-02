import numpy as np


def visualize_keypoints(keypoints, image_size):
    # Visualize keypoints on a blank image
    # keypoints: (num_keypoints, 3), normalized to [-1, 1]
    # image_size: height and width of the image
    image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    intensities = keypoints[:, 2] / np.max(keypoints[:, 2] + 1e-6)
    for i in range(keypoints.shape[0]):
        x = int((keypoints[i, 0] + 1) / 2 * (image_size-1))
        y = int((keypoints[i, 1] + 1) / 2 * (image_size-1))
        image[-y, x, :] = [int(intensities[i] * 255), 0, 0]
    return image


def visualize_keypoint_sequence(keypoints, image_size):
    # Visualize a sequence of keypoints on a blank image
    # keypoints: (seq_len, num_keypoints, 3), normalized to [-1, 1]
    # image_size: height and width of the image
    seq_len = keypoints.shape[0]
    image = np.zeros((seq_len, image_size, image_size, 3), dtype=np.uint8)
    for t in range(seq_len):
        image[t] = visualize_keypoints(keypoints[t], image_size)
    return image
