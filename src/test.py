from PIL import Image
import numpy as np
single_image_name="../DRIVE/train/images/23_training.tif"
#single_image_name="../DRIVE/val/masks/21_training_mask.gif"
img_as_img = Image.open(single_image_name)
# img_as_img.show()
img_as_np = np.asarray(img_as_img)
images_v=np.pad(img_as_np[:,:,1],((4,4),(5,6)),"constant")
print(images_v.shape)
print(np.unique(images_v))