
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import glob
import imageio
import math
import numpy as np

def generate_and_save_images(model, epoch, test_input, image_dir):
    predictions = model(test_input, training=False)
    size = math.sqrt(predictions.shape[0])
    fig = plt.figure(figsize=(size,size))
    assert size**2 == len(test_input), 'need to be square value'
    size = int(size)
    for i in range(predictions.shape[0]):
        plt.subplot(size,size,(i+1))
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig(image_dir+'/image_at_epoch_{:04d}.png'.format(epoch))
    plt.close()

def generate_and_save_gif(image_dir, anim_file):
    with imageio.get_writer(anim_file, mode='I') as writer:
        filenames = glob.glob(image_dir+'/image*.png')
        filenames = sorted(filenames)
        last = -1
        for i,filename in enumerate(filenames):
            frame = 2*(i**0.5)
            if round(frame) > round(last):
                last = frame
            else:
                continue
            image = imageio.imread(filename)
            writer.append_data(image)
        image = imageio.imread(filename)
        writer.append_data(image)