

import numpy as np
import pydicom
from PIL import Image

def dicom2jpeg(src_file, dest_file):
    try:
        ds = pydicom.dcmread(src_file)
        shape = ds.pixel_array.shape

        # Convert to float to avoid overflow or underflow losses.
        image_2d = ds.pixel_array.astype(float)

        # Rescaling grey scale between 0-255
        image_2d_scaled = (np.maximum(image_2d,0) / image_2d.max()) * 255.0

        # Convert to uint
        image_2d_scaled = np.uint8(image_2d_scaled)

            # Create PIL Image and save as JPEG
        image = Image.fromarray(image_2d_scaled)
        image.save(dest_file, 'JPEG', quality=95)
    except:
        print('Could not convert: ', src_file)

