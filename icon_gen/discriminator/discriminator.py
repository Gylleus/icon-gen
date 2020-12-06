import tensorflow as tf
from tensorflow.keras import layers

from .config import DiscriminatorConfig


class Discriminator:
    def __init__(self, config: DiscriminatorConfig):
        self._config = config
        self._loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.optimizer = tf.keras.optimizers.Adam(config.learning_rate)
        self.model = self._create_model()

    def _create_model(self):
        img_size = self._config.img_size
        model = tf.keras.Sequential()
        model.add(
            layers.Conv2D(
                64,
                (5, 5),
                strides=(2, 2),
                padding="same",
                input_shape=[img_size, img_size, self._config.num_channels],
            )
        )
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(self._config.dropout_prob))

        model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding="same"))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(self._config.dropout_prob))

        model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding="same"))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(self._config.dropout_prob))

        model.add(layers.Conv2D(512, (5, 5), strides=(2, 2), padding="same"))
        model.add(layers.LeakyReLU())

        model.add(layers.Flatten())
        model.add(layers.Dense(1))
        model.add(layers.Activation("tanh"))
        return model

    def loss(self, real_guesses, fake_guesses):
        real_loss = self._loss_fn(tf.ones_like(real_guesses), real_guesses)
        fake_loss = self._loss_fn(tf.zeros_like(fake_guesses), fake_guesses)
        total_loss = real_loss + fake_loss
        return total_loss
