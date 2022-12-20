import numpy as np
import scipy
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2


def gaussian_kernel(ksize, sigma):
    x, y = np.meshgrid(np.linspace(-1, 1, ksize), np.linspace(-1, 1, ksize))
    return (1 / (2.0 * np.pi * sigma**2)) * np.exp(-((x**2 + y**2) / (2.0 * sigma**2)))


def convolve(img, kernel):
    return scipy.signal.convolve2d(img, kernel)


def gradient_magnitude(img):
    sobelx = np.asarray([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    sobely = np.asarray([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], np.float32)

    gx = convolve(img, sobelx)
    gy = convolve(img, sobely)

    grad_mag = np.sqrt(gx**2 + gy**2)
    return grad_mag


def threshold_algo(gm):
    # 1
    prev_taui = np.mean(gm)
    taui = np.mean(gm) + 1
    # 2
    i = 0
    while not np.isclose(taui, prev_taui):
        lower_class = gm[gm < prev_taui]
        upper_class = gm[gm >= prev_taui]
        # 3
        mL = np.mean(lower_class)
        mH = np.mean(upper_class)
        # 4
        i = i + 1
        prev_taui = taui
        taui = (mL + mH) / 2
    # 5
    tau = taui
    mask = gm >= tau
    edge_mapped = mask * 255
    return edge_mapped


def q6_step1(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g1 = gaussian_kernel(5, 1)
    g3 = gaussian_kernel(15, 3)

    b1 = convolve(gray, g1)
    b3 = convolve(gray, g3)

    _, axs = plt.subplots(2, 2)
    axs[0][0].imshow(g1, cmap='gray')
    axs[0][0].set_title('sigma=1')
    axs[0][1].imshow(g3, cmap='gray')
    axs[0][1].set_title('sigma=3')
    axs[1][0].imshow(b1, cmap='gray')
    axs[1][1].imshow(b3, cmap='gray')
    plt.savefig('A1_images/q6_step1.jpg')
    plt.clf()


def q6_test(img):
    _, axs = plt.subplots(2, 2)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gauss = gaussian_kernel(5, 1)
    blur = convolve(gray, gauss)

    axs[0][0].imshow(img)
    axs[0][0].set_title('original')
    axs[0][1].imshow(blur, cmap='gray')
    axs[0][1].set_title('blur gray')
    gm = gradient_magnitude(blur)
    axs[1][0].imshow(gm, cmap='gray')
    axs[1][0].set_title('gradient magnitude')
    threshold = threshold_algo(gm)
    axs[1][1].imshow(threshold, cmap='gray')
    axs[1][1].set_title('threshold')
    plt.savefig('A1_images/q6_test.jpg')
    plt.clf()


if __name__ == '__main__':
    img = cv2.imread('A1_images/image1.jpg')[..., ::-1]  # read as BGR
    q6_step1(img)

    q6_test(img)



