# DeepLearning

# Transfer Learning Example

Developed a command line tool to perform transfer learning and recreated from Tutorial on TensorFlow (https://www.tensorflow.org/tutorials/generative/style_transfer) with refactoring.

## Installation
- Anaconda from (https://www.anaconda.com/products/distribution)
- If using amd gpu check (https://learn.microsoft.com/en-us/windows/ai/directml/gpu-tensorflow-plugin)
    - Increases training significantly (2 hours with cpu down to less than 10 minutes)
- Editor: VSCode

## Create an Environment based on requirements
From the conda prompt:
- conda create --name tensorflow-dml python=3.9
- pip install -r requirements.txt

## Example Usage
- python command_line.py --content name url --style name url --verbose True

- Content [Name of image, url of image]
 - Style [Name of Style, url of style]

- python command_line.py --content YellowLabradorLooking_new https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg --style kandinsky5 https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg --verbose True

- python command_line.py --content Nietzsche https://upload.wikimedia.org/wikipedia/commons/c/cd/Nietzsche-21.jpg --style MonaLisa https://upload.wikimedia.org/wikipedia/commons/6/6a/Mona_Lisa.jpg --verbose True

## Example Output
Creates a gif of the process for transfer learning with name: style:name.gif
