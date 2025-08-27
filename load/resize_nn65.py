import os
from PIL import Image
from tqdm import tqdm

for size in [48]:
    inp = './data1024x1024'
    print(size)
    os.mkdir(str(size)+'_nn65')
    filenames = os.listdir(inp)
    for filename in tqdm(filenames):
        Image.open(os.path.join(inp, filename)) \
            .resize((size, size), Image.NEAREST) \
            .save(os.path.join('.', str(size)+'_nn65', filename), quality=65)
"""
for size in [256, 128, 64, 48, 32, 16]:
    inp = './data1024x1024'
    print(size)
    os.mkdir(str(size)+'_nn65')
    filenames = os.listdir(inp)
    for filename in tqdm(filenames):
        Image.open(os.path.join(inp, filename)) \
            .resize((size, size), Image.NEAREST) \
            .save(os.path.join('.', str(size)+'_nn65', filename), quality=65)
"""