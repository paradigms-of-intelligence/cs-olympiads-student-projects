import gzip
import numpy as np
images = gzip.open('train-images-idx3-ubyte.gz','r')
labels = gzip.open('train-labels-idx1-ubyte.gz','r')
labels.read(8)
images.read(16)

image_size = 28
buf = images.read()

num_images = int(len(buf)/(image_size**2))
print(f"Number of images: {num_images}")

data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
data = data.reshape(num_images, image_size, image_size, 1)

imgid = 0

for image in data:
    with open(f"decompiled/img_{imgid}.txt", "w") as file:
        imgid += 1
        for row in image:
            for p in row:
                file.write("0" if p < 128 else "1")
            pass
        buf = labels.read(1)
        file.write(f"\n{str(np.frombuffer(buf, dtype=np.uint8).astype(np.int64))[1]}") 
