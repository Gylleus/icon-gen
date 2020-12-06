class DiscriminatorConfig:
    def __init__(
        self,
        img_size: int,
        learning_rate: float,
        dropout_prob: float,
        num_channels: bool,
    ):
        self.img_size = img_size
        self.dropout_prob = dropout_prob
        self.learning_rate = learning_rate
        self.num_channels = num_channels

    @staticmethod
    def from_config_dict(config: dict)  :
        return DiscriminatorConfig(
            img_size=int(config["img_size"]),
            learning_rate=float(config["discriminator"]["learning_rate"]),
            dropout_prob=float(config["discriminator"]["dropout_prob"]),
            num_channels=3 if bool(config["rgb"]) else 1,
        )
