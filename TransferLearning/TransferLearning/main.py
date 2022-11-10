import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib as mpl

import StyleTransfer as ST


mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False


if __name__ == '__main__':
    content_path =\
        tf.keras.utils.get_file('YellowLabradorLooking_new.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg')
    style_path =\
        tf.keras.utils.get_file('kandinsky5.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg')

    content_image = ST.LoadImage(content_path)
    style_image = ST.LoadImage(style_path)

    plt.subplot(1, 2, 1)
    ST.ShowImage(content_image, 'Content Image')

    plt.subplot(1, 2, 2)
    ST.ShowImage(style_image, 'Style Image')
    plt.show()

    ST.RunTensorHubModel(content_image, style_image)

    ST.VGG19Model(content_image, style_image)

