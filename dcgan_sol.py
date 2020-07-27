####### 1. Import packages #######

import os
import time
import argparse
import glob
import numpy as np
import tqdm 
import tensorflow as tf
from tensorflow.keras import layers
from utils import generate_and_save_images, generate_and_save_gif

# assign specific gpu 
os.environ["CUDA_VISIBLE_DEVICES"]="5" 

# identifier of the code
EXPERIMENT = 'dcgan_sol'

####### 2. Define Argparser  #######
parser = argparse.ArgumentParser()
# dataset
parser.add_argument('--dataset', required=True, 
		 help='mnist | fashion ')

# directory
parser.add_argument('--image_dir', type=str, default='./images', 
		 help='directory for output images')
parser.add_argument('--log_dir', type=str, default='./logs', 
		 help='directory for logging losses using tensorboard')
parser.add_argument('--ckpt_dir', type=str, default='./training_checkpoints', 
		 help='directory for checkpoints')

# training parameters
parser.add_argument('--epochs', type=int, default=100, 
         help='training epochs')
parser.add_argument('--batch_size', type=int, default=256, 
         help='input batch size')
parser.add_argument('--latent_size', type=int, default=100, 
         help='size of the latent z vector')
parser.add_argument('--lr', type=float, default=1e-4, 
         help='learning rate, default=0.0002')
parser.add_argument('--n_samples', type=int, default=16, 
         help='number of images to generate')

args = parser.parse_args()

# assign hyper-parameters
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
LATENT_SIZE = args.latent_size
N_SAMPLES = args.n_samples
LR = args.lr
IMAGE_SIZE = 28
IMAGE_CHANNEL = 1

# assign file/folder location and make folders that does not exist in the directory 
IMAGE_DIR = os.path.join(args.image_dir, EXPERIMENT+'_'+ args.dataset)
CKPT_DIR = os.path.join(args.ckpt_dir,EXPERIMENT+'_'+ args.dataset)
LOG_DIR = os.path.join(args.log_dir, EXPERIMENT+'_'+ args.dataset)
GIF_FILE = EXPERIMENT+'_'+ args.dataset +'.gif'

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


####### 3. Load dataset #######
if args.dataset == 'mnist':
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

elif args.dataset == 'fashion':
    (train_images, train_labels), (_, _) = tf.keras.datasets.fashion_mnist.load_data()

else:
    assert False, 'invalid dataset (choose from [ mnist | fashion ])'

train_images = train_images.reshape(
                        train_images.shape[0], 
                        IMAGE_SIZE, IMAGE_SIZE, 
                        IMAGE_CHANNEL).astype('float32')
train_images = (train_images - 127.5) / 127.5 # normalizing the image in between [-1,1]
train_dataset = tf.data.Dataset.from_tensor_slices(train_images)\
                        .shuffle(train_images.shape[0], reshuffle_each_iteration=True)\
                        .batch(BATCH_SIZE)

####### 4. Define Generator & Discriminator #######

def generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(LATENT_SIZE,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256) # Batch size is None to handle all case

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(IMAGE_CHANNEL, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, IMAGE_CHANNEL)
    
    return model


def discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
   
    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model


####### 5. Define Loss for Generator & Discriminator #######

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    # (1) Loss for D network: maximize log(D(x)) + log(1 - D(G(z)))
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    # (2) Loss for G network: maximize log(D(G(z)))
    return cross_entropy(tf.ones_like(fake_output), fake_output)



####### 6. Define Optimizer & Checkpoint #######

# declare models
generator = generator_model()
discriminator = discriminator_model()

# declare optimizers
generator_optimizer = tf.keras.optimizers.Adam(LR)
discriminator_optimizer = tf.keras.optimizers.Adam(LR)

 
checkpoint_prefix = os.path.join(CKPT_DIR, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)



####### 7. Train  #######

# for visualization
seed_z = tf.random.normal([N_SAMPLES, LATENT_SIZE])

# define metrics that track losses
train_gen_loss = tf.keras.metrics.Mean('gen_loss', dtype=tf.float32)
train_disc_loss = tf.keras.metrics.Mean('disc_loss', dtype=tf.float32)
train_summary_writer = tf.summary.create_file_writer(LOG_DIR)

@tf.function
def train_step(images):
    # sample noise z from normal distribution
    noise_z = tf.random.normal([BATCH_SIZE, LATENT_SIZE])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

        # generator generates fake image
        generated_images = generator(noise_z, training=True)

        # discriminator discriminates real image
        real_output = discriminator(images, training=True)

        # discriminator discriminates fake image
        fake_output = discriminator(generated_images, training=True)

        # compute discriminator loss
        disc_loss = discriminator_loss(real_output, fake_output)
        
        # compute generator loss
        gen_loss = generator_loss(fake_output)

    # compuate gradients for generator & discriminator
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    # update gradients for generator & discriminator
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    # keep tracking the losses
    train_gen_loss(gen_loss)
    train_disc_loss(disc_loss)


for epoch in range(EPOCHS):
    start = time.time()

    for image_batch in train_dataset:
        train_step(image_batch)
    
    with train_summary_writer.as_default():
        tf.summary.scalar('gen_loss', train_gen_loss.result(), step=epoch)
        tf.summary.scalar('disc_loss', train_disc_loss.result(), step=epoch)
    
    generate_and_save_images(generator, epoch + 1, seed_z, IMAGE_DIR)

    if (epoch + 1) % 15 == 0:
        checkpoint.save(file_prefix = checkpoint_prefix)
    
    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

generate_and_save_images(generator, EPOCHS, seed_z, IMAGE_DIR)
generate_and_save_gif(IMAGE_DIR, GIF_FILE)