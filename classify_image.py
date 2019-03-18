"""A demo to classify image."""
from edgetpu.classification.engine import ClassificationEngine
from PIL import Image
import glob

# Folder where our images to be classified are stored
image_folder = 'images/*.jpg'
# Uninitialized array of the images
image_list = []
# Object classifaction model - courtesy of tensorflow.org
model = 'mobilenet_v1_1.0_224_quant.tflite'
# Labels for the objects we classify
labelfile = 'labels_mobilenet_quant_v1_224.txt'

# Function to read labels from text files.
def ReadLabelFile(file_path):
  i = 0
  with open(file_path, 'r') as f:
    lines = f.readlines()
  ret = {}
  for line in lines:
    pair = [i,line]
    ret[int(pair[0])] = pair[1].strip()
    i += 1
  return ret


def main():

  # Prepare labels.
  labels = ReadLabelFile(labelfile)
  # Initialize engine.
  engine = ClassificationEngine(model)
  # Run inference on all images in images/ folder
  for image in glob.glob(image_folder):
    img = Image.open(image)
    for result in engine.ClassifyWithImage(img, top_k=1):
      print ('---------------------------')
      print (labels[result[0]])
      print ('Score : ', result[1])
      print ('-----END-------------------')

if __name__ == '__main__':
  main()
