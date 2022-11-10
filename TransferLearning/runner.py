"""
Main Driver
"""
import tensorflow as tf

import matplotlib.pyplot as plt

from TransferLearning import StyleTransfer as TL


def Run(content, style, verbose):
    if verbose is None:
        verbose=False

    # Train with Style Transfer Model
    content_path =\
        tf.keras.utils.get_file(content[0], content[1])
    style_path =\
        tf.keras.utils.get_file(style[0], style[1])

    content_image = TL.LoadImage(content_path)
    style_image = TL.LoadImage(style_path)

    if verbose:
        plt.subplot(1, 2, 1)
        TL.ShowImage(content_image, 'Content Image')

        plt.subplot(1, 2, 2)
        TL.ShowImage(style_image, 'Style Image')
        plt.show()

    if verbose:
        print("Beginning Model")

    TL.VGG19Model(content_image, style_image, gif_output=style[0] + '.gif')

    if verbose:
        print("Finished Model")
