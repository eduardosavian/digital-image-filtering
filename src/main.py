import sys

from noise import put_noise
from calculations import calculate_mse, calculate_psnr
from histogram import make_histogram, plot_histogram
from blur import blur


from utils import read_image, write_image, normalize_image, denormalize_image, write_plot_image


def main():
    print("Main")

    images_name = ["barb.tif", "tank.tiff", "lena2.tif"]

    std_dev = float(sys.argv[1])

    for image_name in images_name:
        print(image_name)

        image = read_image(image_name, "repository")
        image_normalized = normalize_image(image)

        image_noise = put_noise(image_normalized, std_dev)
        image_noise_desnormalized = denormalize_image(image_noise)
        write_image(image_name, "noise", image_noise_desnormalized)

        mse = calculate_mse(image, image_noise_desnormalized)
        psnr = calculate_psnr(image, mse)

        print("\nMSE")
        print(mse)
        print("\nPSNR")
        print(psnr)

        image_blur = blur(image_noise)
        image_blur_desnormalized = denormalize_image(image_blur)
        write_image(image_name, "mask", image_noise_desnormalized)

        plot_histogram(image, image_noise_desnormalized, image_blur_desnormalized, image_name)



if __name__ == "__main__":
    main()
