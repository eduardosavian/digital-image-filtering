import numpy as np
import matplotlib.pyplot as plt


import seaborn as sns

from utils import read_image, write_plot_image

def img_8bit_val_count_percent(img: np.ndarray):
    counts = np.zeros(256, np.uint64)

    h, w = img.shape
    for row in range(h):
        for col in range(w):
            idx = img[row, col]
            counts[idx] += 1

    total = img.size
    counts = counts / total

    return counts


def make_histogram(image_name: str, title: str = ''):
    img = read_image(image_name, "manipulations/mono")

    if title == '':
        title = image_name

    try:
        plt.xlabel('Pixel Value [0, 255]')
        plt.ylabel('% of Pixels')
        vals = img_8bit_val_count_percent(img)
        plt.bar(range(256), vals, width=1, align='edge')  # Adjusted the range and width
        plt.xlim(0, 255)  # Set x-axis limit
        plt.ylim(0)  # Set y-axis limit to start from 0, but don't specify an upper limit
        write_plot_image(image_name, "histogram", plt)
    finally:
        plt.clf()

def plot_histogram(image: np.ndarray, image_noise: np.ndarray, image_mask: np.ndarray, image_name: str) -> np.ndarray:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))  # Create two subplots side by side
# Use seaborn color palette for better colors
    colors = sns.color_palette("bright")

    # Plot a histogram of image without noise (excluding 255 and 0) in ax1
    ax1.hist(image.ravel(), bins=256, range=(0, 255), color=colors[0], alpha=0.7, label='Without Noise')
    ax1.hist(image_noise.ravel(), bins=256, range=(0, 255), color=colors[3], alpha=0.7, label='With Noise')

    # Set labels and legend for ax1
    ax1.set_xlabel('Pixel Value')
    ax1.set_ylabel('Frequency')
    ax1.legend(loc='upper right')

    # Plot a histogram of image_mask in ax2
    ax2.hist(image_noise.ravel(), bins=255, range=(0, 255), color=colors[3], alpha=0.7, label='With Noise')
    ax2.hist(image_mask.ravel(), bins=255, range=(0, 255), color=colors[2], alpha=0.7, label='With Mask')

    # Set labels and legend for ax2
    ax2.set_xlabel('Pixel Value')
    ax2.set_ylabel('Frequency')
    ax2.legend(loc='upper right')

    # Save the plot

    write_plot_image(image_name, "histogram", plt)

