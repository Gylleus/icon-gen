import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers
import time
import os
from IPython import display
from PIL import Image

#data_dir="C:/Users/Karl/Downloads/IconData/Data/Abilities/GW2"
data_dir="C:/Users/Karl/Downloads/IconData/backup_data"
batch_size = 16
img_height = 64
img_width = img_height
EPOCHS = 50000
noise_dim = 100
num_examples_to_generate = 16
epoch_print_interval = 25

disc_dropout_prob = 0.4
gen_dropout_prob = 0.4
 
RGB=False
num_channels = 3 if RGB else 1

generator_learning_rate=5e-5
discriminator_learning_rate=1e-5

# Create Dataset

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.1,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)
classes = train_ds.class_names
num_classes = len(classes)

#AUTOTUNE = tf.data.experimental.AUTOTUNE
#train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Rescale input images
train_ds = train_ds.map(lambda x, y: (tf.image.rgb_to_grayscale(x, name=None), y))
normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(scale=1./127.5, offset=-1)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))

def make_generator_model():
    image_size = img_height
    mid_size = image_size //2
    start_size = image_size //16
    model = tf.keras.Sequential()
    model.add(layers.Dense(start_size*start_size*256, input_shape=(noise_dim,)))
    model.add(layers.BatchNormalization(momentum=0.9))
    model.add(layers.Activation('relu'))
    model.add(layers.Reshape((start_size, start_size, 256)))

    model.add(layers.Dropout(gen_dropout_prob))


    model.add(layers.UpSampling2D())
    model.add(layers.UpSampling2D())
    model.add(layers.Conv2DTranspose(128, 5, padding='same'))
    model.add(layers.BatchNormalization(momentum=0.9))
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(gen_dropout_prob))

    model.add(layers.UpSampling2D())
    model.add(layers.Conv2DTranspose(128, 5, padding='same'))
    model.add(layers.BatchNormalization(momentum=0.9))
    model.add(layers.Activation('relu'))

    model.add(layers.UpSampling2D())
    model.add(layers.Conv2DTranspose(64, 5, padding='same'))
    model.add(layers.BatchNormalization(momentum=0.9))
    model.add(layers.Activation('relu'))

    model.add(layers.Conv2DTranspose(32, 5, padding='same'))
    model.add(layers.BatchNormalization(momentum=0.9))
    model.add(layers.Activation('relu'))
    
    model.add(layers.Conv2DTranspose(num_channels, 5,padding='same'))
    model.add(layers.Activation('tanh'))
    print( model.output_shape)

    return model


def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[img_height, img_height, num_channels]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(disc_dropout_prob))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(disc_dropout_prob))

    model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(disc_dropout_prob))

    model.add(layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())

    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    model.add(layers.Activation('tanh'))


    return model

generator = make_generator_model()
discriminator = make_discriminator_model()
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(generator_learning_rate)
discriminator_optimizer = tf.keras.optimizers.Adam(discriminator_learning_rate)


# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    noise = tf.random.normal([batch_size, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
        tf.keras.backend.print_tensor(gen_loss, "GEN")
        tf.keras.backend.print_tensor(disc_loss, "DISC")
    

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))



def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training=False)
  predictions = (predictions+1)/2
  fig = plt.figure(figsize=(4,4))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, :], cmap=None if RGB else 'gray')
      plt.axis('off')

  plt.savefig('image_at_epoch_{}.png'.format(epoch))
  #plt.show()

kek_noise = tf.random.normal([num_examples_to_generate, noise_dim])
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
def train(dataset, epochs):
  for epoch in range(epochs):
    start = time.time()
    for image_batch in dataset:
        train_step(image_batch[0])

    # Produce images for the GIF as we go
    
    if epoch % epoch_print_interval == 0:
        display.clear_output(wait=True)
        checkpoint.save(file_prefix = f"{checkpoint_prefix}_{epoch}")
        generate_and_save_images(generator,
                             epoch,
                             kek_noise)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    # Generate after the final epoch

train(train_ds, EPOCHS)
generator.save('saved_model')
display.clear_output(wait=True)
generate_and_save_images(generator,
                        epochs,
                        kek_noise)
