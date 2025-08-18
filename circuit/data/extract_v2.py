import gzip
import numpy as np
import matplotlib.pyplot as plt

dotest = input("Decompile test data instead? y/N : ").lower() == 'y'
showimg = input("Show images? y/N: ").lower() == 'y'

images = gzip.open('t10k-images-idx3-ubyte.gz' if dotest else 'train-images-idx3-ubyte.gz','r')
labels = gzip.open('t10k-labels-idx1-ubyte.gz' if dotest else 'train-labels-idx1-ubyte.gz','r')
labels.read(8)
images.read(16)

image_size = 28
buf = images.read()

num_images = int(len(buf)/(image_size**2))
print(f"Number of images: {num_images}")

data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
data = data.reshape(num_images, image_size, image_size, 1)

output: list[str] = []
idd = 0

with open(f'testdata.txt' if dotest else 'training.txt', "w") as file:
    for image in data:
        buf = labels.read(1)
        if showimg:
            imge = np.asarray(image).squeeze()
            plt.imshow(imge)
            print(f"{buf[0]}") 
            plt.show()
        for row in image:
            for p in row:
                output.append("0" if p < 64 else "1")
        output.append(f"\n{buf[0]}\n")
        idd += 1
        if(idd%100 == 0): print(f"Zero: {idd}")
    file.write("".join(output).strip())
