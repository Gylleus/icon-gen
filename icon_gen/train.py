import logging
import os

import matplotlib.pyplot as plt
import yaml
import click
import tensorflow as tf
import tensorflowjs as tfjs
from IPython import display


import generator
import discriminator
import time

logger = logging.Logger(__name__)


class Trainer:
    def __init__(self, generator: generator.Generator, discriminator, noise_dim: int, rgb: bool):
        self.generator = generator
        self.discriminator = discriminator
        self.noise_dim = noise_dim
        self.rgb = rgb
        self.image_dir = "./train_images"
        os.makedirs(self.image_dir, exist_ok=True)
        self.static_noise = tf.random.normal([16, noise_dim])
        self.checkpoint_prefix = os.path.join("./training_checkpoints", "ckpt")
        self.checkpoint = tf.train.Checkpoint(
            generator_optimizer=generator.optimizer,
            discriminator_optimizer=discriminator.optimizer,
            generator=generator.model,
            discriminator=discriminator.model,
        )

    def train(self, dataset, epochs: int, save_interval: int):
        for epoch in range(epochs):
            start = time.time()
            for image_batch in dataset:
                image_batch = image_batch[0]
                self._step(image_batch, batch_size=image_batch.shape[0])

            if epoch % save_interval == 0:
                self._save_checkpoint(epoch)
                self.generate_and_save_images(epoch)

            print("Time for epoch {} is {} sec".format(epoch + 1, time.time() - start))

    def _save_checkpoint(self, epoch: int):
        self.checkpoint.save(file_prefix=f"{self.checkpoint_prefix}_{epoch}")

    @tf.function
    def _step(self, images: tf.Tensor, batch_size: int ):
        noise = tf.random.normal([batch_size, self.noise_dim])
        generator = self.generator.model
        discriminator = self.discriminator.model

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = generator(noise, training=True)

            real_output = discriminator(images, training=True)
            fake_output = discriminator(generated_images, training=True)

            gen_loss = self.generator.loss(fake_output)
            disc_loss = self.discriminator.loss(real_output, fake_output)
            tf.keras.backend.print_tensor(gen_loss, "GEN")
            tf.keras.backend.print_tensor(disc_loss, "DISC")

        gradients_of_generator = gen_tape.gradient(
            gen_loss, generator.trainable_variables
        )
        gradients_of_discriminator = disc_tape.gradient(
            disc_loss, discriminator.trainable_variables
        )

        self.generator.optimizer.apply_gradients(
            zip(gradients_of_generator, generator.trainable_variables)
        )
        self.discriminator.optimizer.apply_gradients(
            zip(gradients_of_discriminator, discriminator.trainable_variables)
        )

    def generate_and_save_images(self, epoch: int):
        predictions = self.generator.model(self.static_noise, training=False)
        # Remap from [-1,1] to [0,1]
        predictions = (predictions + 1) / 2
        fig = plt.figure(figsize=(4, 4))

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow(predictions[i, :, :, :], cmap=None if self.rgb else "gray")
            plt.axis("off")
        filename = os.path.join(self.image_dir, "epoch_{}.png".format(epoch))
        plt.savefig(filename)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_data(data_path: str, img_size : int, batch_size: int, use_rgb: bool):
    data = tf.keras.preprocessing.image_dataset_from_directory(
        data_path,
        validation_split=0.1,
        subset="training",
        seed=123,
        image_size=(img_size, img_size),
        batch_size=batch_size,
    )
    logger.info(
        f"Data loaded successfully ~ Found {len(data.class_names)} classes."
    )
    if not use_rgb:
        data = data.map(lambda x, y: (tf.image.rgb_to_grayscale(x, name=None), y))
    # Normalize
    normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(
        scale=1.0 / 127.5, offset=-1
    )
    data = data.map(lambda x, y: (normalization_layer(x), y))
    return data


@click.command()
@click.option("--config", required=True)
def main(config: str):
    config = load_config(config)
    parameters = config['parameters']
    train_data = load_data(
        data_path=config['data']['directory'], 
        img_size=parameters['img_size'], 
        batch_size=parameters['batch_size'],
        use_rgb=parameters["rgb"],
        )


    gen = generator.Generator(
        generator.GeneratorConfig.from_config_dict(parameters)
    )
    disc = discriminator.Discriminator(
        discriminator.DiscriminatorConfig.from_config_dict(parameters)
    )
    trainer = Trainer(gen, disc, noise_dim=parameters['generator']['noise_dim'], rgb=parameters["rgb"])
    trainer.train(dataset=train_data, epochs=config['train']['epochs'], save_interval=config['train']['save_interval'])
    gen.save("saved_model")
    tfjs.converters.save_keras_model(gen, "saved_js_model")


if __name__ == "__main__":
    main()
