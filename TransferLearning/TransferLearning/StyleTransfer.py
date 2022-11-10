"""
Style Transfer from a base model.
"""
import os
import time

import matplotlib.pyplot as plt
import matplotlib.pylab as plt
import matplotlib as mpl

import PIL.Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

import TransferLearning.StyleModel as SM

# Load compressed models from tensorflow_hub
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'

# Matplotlib characterstics
mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False


def TensortoImage(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)


def LoadImage(path_to_image):
    max_dim = 512
    img = tf.io.read_file(path_to_image)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]

    return img


def ShowImage(image, title=None):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)

    plt.imshow(image)
    if title:
        plt.title(title)


def RunTensorHubModel(content_image, style_image):
    hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
    stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]

    im = TensortoImage(stylized_image)
    im.show()

    return


# Build the Model
def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

def style_content_loss(outputs, params):
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-params['style_targets'][name])**2)
                        for name in style_outputs.keys()])
    style_loss *= params['style_weight'] / params['num_style_layers']

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-params['content_targets'][name])**2)
                            for name in content_outputs.keys()])
    content_loss *= params['content_weight'] / params['num_content_layers']
    loss = style_loss + content_loss

    return loss


@tf.function()
def train_step(image, params):
    with tf.GradientTape() as tape:
        outputs = params['extractor'](image)
        loss = style_content_loss(outputs, params)

    grad = tape.gradient(loss, image)
    params['opt'].apply_gradients([(grad, image)])
    image.assign(clip_0_1(image))


def VGG19Model(content_image, style_image, gif_output):
    # Define content and style representations
    x = tf.keras.applications.vgg19.preprocess_input(content_image*255)
    x = tf.image.resize(x, (224, 224))
    vgg = tf.keras.applications.VGG19(include_top=True, weights='imagenet')
    prediction_probabilities = vgg(x)
    print(prediction_probabilities.shape)

    predicted_top_5 = tf.keras.applications.vgg19.decode_predictions(prediction_probabilities.numpy())[0]
    [print(class_name, prob) for (number, class_name, prob) in predicted_top_5]

    # Without the classification head
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')

    print()
    for layer in vgg.layers:
        print(layer.name)

    content_layers = ['block5_conv2']
    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1',
                    'block4_conv1',
                    'block5_conv1']

    num_content_layers = len(content_layers)
    num_style_layers = len(style_layers)

    style_extractor = SM.vgg_layers(style_layers)
    style_outputs = style_extractor(style_image*255)

    #Look at the statistics of each layer's output
    for name, output in zip(style_layers, style_outputs):
        print(name)
        print("  shape: ", output.numpy().shape)
        print("  min: ", output.numpy().min())
        print("  max: ", output.numpy().max())
        print("  mean: ", output.numpy().mean())
        print()

    extractor = SM.StyleContentModel(style_layers, content_layers)
    results = extractor(tf.constant(content_image))

    print('Styles:')
    for name, output in sorted(results['style'].items()):
        print("  ", name)
        print("    shape: ", output.numpy().shape)
        print("    min: ", output.numpy().min())
        print("    max: ", output.numpy().max())
        print("    mean: ", output.numpy().mean())
        print()

    print("Contents:")
    for name, output in sorted(results['content'].items()):
        print("  ", name)
        print("    shape: ", output.numpy().shape)
        print("    min: ", output.numpy().min())
        print("    max: ", output.numpy().max())
        print("    mean: ", output.numpy().mean())

    # Gradient Descent
    style_targets = extractor(style_image)['style']
    content_targets = extractor(content_image)['content']
    image = tf.Variable(content_image)

    opt = tf.keras.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
    style_weight=1e-2
    content_weight=1e4

    params = {
        'extractor': extractor,
        'opt': opt,
        'style_weight': style_weight,
        'style_targets': style_targets,
        'content_weight': content_weight,
        'content_targets': content_targets,
        'num_content_layers': num_content_layers,
        'num_style_layers': num_style_layers
    }

    print("Performing Longer Optimization")
    start = time.time()
    gif = []

    epochs = 10
    steps_per_epoch = 50
    step = 0
    for n in range(epochs):
        for m in range(steps_per_epoch):
            step += 1
            train_step(image, params)
            print(".", end='', flush=True)
            gif.append(TensortoImage(image))

        # gif.append(TensortoImage(image))
        print("Train step: {}".format(step))

    end = time.time()
    print("Total time: {:.1f}".format(end-start))

    gif[0].save(gif_output, save_all=True, optimize=False,
        append_images=gif[1:], loop=0)
