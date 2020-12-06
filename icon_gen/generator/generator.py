import tensorflow as tf
from tensorflow.keras import layers

from .config import GeneratorConfig


class Generator:
    def __init__(self, config: GeneratorConfig):
        self._config = config
        self._loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.optimizer = tf.keras.optimizers.Adam(config.learning_rate)
        self.model = self._create_model()


    def _create_model(self):
        image_size = self._config.img_size
        mid_size = image_size // 2
        start_size = image_size // 16
        model = tf.keras.Sequential()
        model.add(
            layers.Dense(
                start_size * start_size * 256, input_shape=(self._config.noise_dim,)
            )
        )
        model.add(layers.BatchNormalization(momentum=0.9))
        model.add(layers.Activation("relu"))
        model.add(layers.Reshape((start_size, start_size, 256)))
        model.add(layers.Dropout(self._config.dropout_prob))

        model.add(layers.UpSampling2D())
        model.add(layers.UpSampling2D())
        model.add(layers.Conv2DTranspose(128, 5, padding="same"))
        model.add(layers.BatchNormalization(momentum=self._config.batch_norm_momentum))
        model.add(layers.Activation("relu"))
        model.add(layers.Dropout(self._config.dropout_prob))

        model.add(layers.UpSampling2D())
        model.add(layers.Conv2DTranspose(128, 5, padding="same"))
        model.add(layers.BatchNormalization(momentum=self._config.batch_norm_momentum))
        model.add(layers.Activation("relu"))

        model.add(layers.UpSampling2D())
        model.add(layers.Conv2DTranspose(64, 5, padding="same"))
        model.add(layers.BatchNormalization(momentum=self._config.batch_norm_momentum))
        model.add(layers.Activation("relu"))

        model.add(layers.Conv2DTranspose(32, 5, padding="same"))
        model.add(layers.BatchNormalization(momentum=self._config.batch_norm_momentum))
        model.add(layers.Activation("relu"))

        model.add(layers.Conv2DTranspose(self._config.num_channels, 5, padding="same"))
        model.add(layers.Activation("tanh"))

        return model

    def loss(self, discriminator_guesses):
        return self._loss_fn(tf.ones_like(discriminator_guesses), discriminator_guesses)

    def save(self, directory : str):
        self.model.save(directory)