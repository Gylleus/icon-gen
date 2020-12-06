class GeneratorConfig:
    def __init__(
        self,
        img_size: int,
        noise_dim: int,
        learning_rate: float,
        dropout_prob: float,
        batch_norm_momentum: float,
        num_channels: bool,
    ):
        self.img_size = img_size
        self.noise_dim = noise_dim
        self.dropout_prob = dropout_prob
        self.learning_rate = learning_rate
        self.batch_norm_momentum = batch_norm_momentum
        self.num_channels = num_channels

    @staticmethod
    def from_config_dict(config: dict):
        return GeneratorConfig(
            img_size=int(config["img_size"]),
            noise_dim=int(config["generator"]["noise_dim"]),
            learning_rate=float(config["generator"]["learning_rate"]),
            dropout_prob=float(config["generator"]["dropout_prob"]),
            batch_norm_momentum=float(config["generator"]["batch_norm_momentum"]),
            num_channels=3 if bool(config["rgb"]) else 1,
        )
