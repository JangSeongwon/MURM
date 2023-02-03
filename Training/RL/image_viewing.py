
import numpy as np
from PIL import Image

img_array = np.load('/media/jang/jang/0ubuntu/image_dataset/Images_produced_for_goals/4.npy')

im = Image.fromarray(img_array.astype(np.uint8))

im.show()
