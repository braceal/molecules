from molecules.ml.hyperparams import Hyperparams


class AAE3dHyperparams(Hyperparams):
    def __init__(
        self,
        num_features=1,
        encoder_filters=[64, 128, 256, 256, 512],
        encoder_kernel_sizes=[1, 1, 1, 1, 1],
        generator_filters=[64, 128, 512, 1024],
        discriminator_filters=[512, 512, 128, 64],
        latent_dim=256,
        encoder_relu_slope=0.0,
        generator_relu_slope=0.0,
        discriminator_relu_slope=0.0,
        use_encoder_bias=True,
        use_generator_bias=True,
        use_discriminator_bias=True,
        noise_mu=0.0,
        noise_std=1.0,
        lambda_rec=1.0,
        lambda_gp=10.0,
    ):

        # network features
        self.num_features = num_features
        self.encoder_filters = encoder_filters
        self.encoder_kernel_sizes = encoder_kernel_sizes
        self.generator_filters = generator_filters
        self.discriminator_filters = discriminator_filters
        self.latent_dim = latent_dim
        self.encoder_relu_slope = encoder_relu_slope
        self.generator_relu_slope = generator_relu_slope
        self.discriminator_relu_slope = discriminator_relu_slope
        self.use_encoder_bias = use_encoder_bias
        self.use_generator_bias = use_generator_bias
        self.use_discriminator_bias = use_discriminator_bias

        # random features
        self.noise_mu = noise_mu
        self.noise_std = noise_std

        # loss features
        self.lambda_rec = lambda_rec
        self.lambda_gp = lambda_gp

        # Placed after member vars are declared so that base class can validate
        super().__init__()

    def validate(self):
        return
