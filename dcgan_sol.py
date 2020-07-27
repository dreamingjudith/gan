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

def Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.dense0 = layers.Dense(7*7*256, use_bias=False, input_shape=(LATENT_SIZE,))
        self.bn0 = layers.BatchNormalization()
        self.leaky_relu0 = layers.LeakyReLU()
        self.reshape0 = layers.Reshape((7, 7, 256))

        self.conv1 = layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False)
        self.bn1= layers.BatchNormalization()
        self.leaky_relu1 = layers.LeakyReLU()

        self.conv2= layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)
        self.bn2 = layers.BatchNormalization())
        self.leaky_relu2 = layers.LeakyReLU()

        self.conv3= layers.Conv2DTranspose(IMAGE_CHANNEL, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    
    def call(self, x, training=True):
        x = self.dense0(x)
        x = self.bn0(x)
        x = self.leaky_relu0(x)
        x = self.reshape0(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leaky_relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leaky_relu2(x)
        return self.conv3(x)

def Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv0 = layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL])
        self.leaky_relu0 = layers.LeakyReLU()
        self.dropout0 = layers.Dropout(0.3)

        self.conv1 = layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same')
        self.leaky_relu1 = layers.LeakyReLU()
        self.dropout1 = layers.Dropout(0.3)
   
        self.flatten2 = layers.Flatten()
        self.dense2 = layers.Dense(1)
    
    def call(self, x, training=True):
        x = self.conv0(x)
        x = self.leaky_relu0(x)
        if training:
            x = self.dropout0(x, training=training)
        x = self.conv1(x)
        x = self.leaky_relu2(x)
        if training:
            x = self.dropout1(x, training=training)
        x = self.flatten2(x)
        return self.dense2(x)


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
generator = Generator()
discriminator = Discriminator()

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
