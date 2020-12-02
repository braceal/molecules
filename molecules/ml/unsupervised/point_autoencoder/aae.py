import time
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from itertools import chain 
from collections import OrderedDict, namedtuple
from molecules.ml.unsupervised.point_autoencoder.hyperparams import AAE3dHyperparams
from molecules.ml.unsupervised.utils import _init_weights
from molecules.ml.unsupervised.point_autoencoder.losses import chamfer_loss as cl
#from molecules.ml.unsupervised.point_autoencoder.losses import earth_movers_distance as emd
from molecules.ml.hyperparams import OptimizerHyperparams, get_optimizer

__all__ = ['AAE3d']

Device = namedtuple('Device', ['encoder', 'generator', 'discriminator'])

class Generator(nn.Module):
    def __init__(self, num_points, num_features, hparams, init_weights=None):
        super().__init__()

        # copy some parameters
        self.num_points = num_points
        self.num_features = num_features
        self.z_size = hparams.latent_dim
        self.use_bias = hparams.use_generator_bias
        self.relu_slope = hparams.generator_relu_slope

        # select activation
        self.activ_args = {"inplace": True}
        if self.relu_slope > 0.:
            self.activation = nn.LeakyReLU
            self.activ_args["negative_slope"] = self.relu_slope
        else:
            self.activation = nn.ReLU
            
        # first layer
        layers = OrderedDict([('gen_linear1', nn.Linear(in_features = self.z_size,
                                                    out_features = hparams.generator_filters[0],
                                                    bias=self.use_bias)),
                              ('gen_relu1', self.activation(**self.activ_args))])
        
        # intermediate layers
        for idx in range(1, len(hparams.generator_filters)):
            layers.update({'gen_linear{}'.format(idx+1) : nn.Linear(in_features = hparams.generator_filters[idx - 1],
                                                                out_features = hparams.generator_filters[idx],
                                                                bias=self.use_bias)})
            layers.update({'gen_relu{}'.format(idx+1) : self.activation(**self.activ_args)})

        # last layer
        layers.update({'gen_linear{}'.format(idx+2) : nn.Linear(in_features = hparams.generator_filters[-1],
                                                            out_features = self.num_points * (3 + self.num_features),
                                                            bias=self.use_bias)})

        # construct model
        self.model = nn.Sequential(layers)
        
        # init weights
        self.init_weights(init_weights)
        
        #self.model = nn.Sequential(
        #    nn.Linear(in_features=self.z_size, out_features=64, bias=self.use_bias),
        #    nn.ReLU(inplace=True),
        #
        #    nn.Linear(in_features=64, out_features=128, bias=self.use_bias),
        #    nn.ReLU(inplace=True),
        #
        #    nn.Linear(in_features=128, out_features=512, bias=self.use_bias),
        #    nn.ReLU(inplace=True),
        #
        #    nn.Linear(in_features=512, out_features=1024, bias=self.use_bias),
        #    nn.ReLU(inplace=True),
        #
        #    nn.Linear(in_features=1024, out_features=2048 * 3, bias=self.use_bias),
        #)
        
    def init_weights(self, init_weights):
        if init_weights is None:
            self.model.apply(_init_weights)
        elif init_weights.endswith('.pt'):
            checkpoint = torch.load(init_weights, map_location='cpu')
            self.load_state_dict(checkpoint['generator_state_dict'])
        
    def save_weights(self, path):
        torch.save(self.state_dict(), path)
        
    def load_weights(self, path):
        self.load_state_dict(torch.load(path))

    def forward(self, input):
        #output = self.model(input)
        # TODO: test without squeeze
        output = self.model(input.squeeze())
        output = output.view(-1, (3 + self.num_features), self.num_points)
        return output


class Discriminator(nn.Module):
    def __init__(self, hparams, init_weights=None):
        super().__init__()

        self.z_size = hparams.latent_dim
        self.use_bias = hparams.use_discriminator_bias
        self.relu_slope = hparams.discriminator_relu_slope

        # select activation
        self.activ_args = {"inplace": True}
        if self.relu_slope > 0.:
            self.activation = nn.LeakyReLU
            self.activ_args["negative_slope"] = self.relu_slope
        else:
            self.activation = nn.ReLU

        # first layer
        layers = OrderedDict([('dis_linear1', nn.Linear(in_features = self.z_size,
                                                        out_features = hparams.discriminator_filters[0],
                                                        bias = self.use_bias)),
                              ('dis_relu1', self.activation(**self.activ_args))])

        # intermediate layers
        for idx in range(1, len(hparams.discriminator_filters)):
            layers.update({'dis_linear{}'.format(idx+1) : nn.Linear(in_features = hparams.discriminator_filters[idx - 1],
                                                                    out_features = hparams.discriminator_filters[idx],
                                                                    bias = self.use_bias)})
            layers.update({'dis_relu{}'.format(idx+1) : self.activation(**self.activ_args)})

        # final layer
        layers.update({'dis_linear{}'.format(idx+2) : nn.Linear(in_features = hparams.discriminator_filters[-1],
                                                                out_features = 1,
                                                                bias = self.use_bias)})

        # construct model
        self.model = nn.Sequential(layers)
        
        # init weights
        self.init_weights(init_weights)
            
        #self.model = nn.Sequential(
        #
        #    nn.Linear(self.z_size, 512, bias=True),
        #    nn.ReLU(inplace=True),
        #
        #    nn.Linear(512, 512, bias=True),
        #    nn.ReLU(inplace=True),
        #
        #    nn.Linear(512, 128, bias=True),
        #    nn.ReLU(inplace=True),
        #
        #    nn.Linear(128, 64, bias=True),
        #    nn.ReLU(inplace=True),
        #
        #    nn.Linear(64, 1, bias=True)
        #)
        
    def init_weights(self, init_weights):
        if init_weights is None:
            self.model.apply(_init_weights)
        elif init_weights.endswith('.pt'):
            checkpoint = torch.load(init_weights, map_location='cpu')
            self.load_state_dict(checkpoint['discriminator_state_dict'])
    
    def save_weights(self, path):
        torch.save(self.state_dict(), path)
        
    def load_weights(self, path):
        self.load_state_dict(torch.load(path))

    def forward(self, x):
        logit = self.model(x)
        return logit


class Encoder(nn.Module):
    def __init__(self, num_points, num_features, hparams, init_weights=None):
        super().__init__()

        # copy some parameters
        self.num_points = num_points
        self.num_features = num_features
        self.z_size = hparams.latent_dim
        self.use_bias = hparams.use_encoder_bias
        self.relu_slope = hparams.encoder_relu_slope
        
        # select activation
        self.activ_args = {"inplace": True}
        if self.relu_slope > 0.:
            self.activation = nn.LeakyReLU
            self.activ_args["negative_slope"] = self.relu_slope
        else:
            self.activation = nn.ReLU

        # first layer
        layers = OrderedDict([('enc_conv1', nn.Conv1d(in_channels = (3 + self.num_features),
                                                      out_channels = hparams.encoder_filters[0],
                                                      kernel_size = hparams.encoder_kernel_sizes[0],
                                                      bias = self.use_bias)),
                              ('enc_relu1', self.activation(**self.activ_args))])

        # intermediate layers
        for idx in range(1, len(hparams.encoder_filters)-1):
            layers.update({'enc_conv{}'.format(idx+1) : nn.Conv1d(in_channels = hparams.encoder_filters[idx - 1],
                                                                  out_channels = hparams.encoder_filters[idx],
                                                                  kernel_size = hparams.encoder_kernel_sizes[idx],
                                                                  bias = self.use_bias)})
            layers.update({'enc_relu{}'.format(idx+1) : self.activation(**self.activ_args)})
            
        # final layer
        layers.update({'enc_conv{}'.format(idx+2) : nn.Conv1d(in_channels = hparams.encoder_filters[-2],
                                                              out_channels = hparams.encoder_filters[-1],
                                                              kernel_size = hparams.encoder_kernel_sizes[-1],
                                                              bias = self.use_bias)})

        # construct model
        self.conv = nn.Sequential(layers)

        self.fc = nn.Sequential(
            nn.Linear(hparams.encoder_filters[-1],
                      hparams.encoder_filters[-2],
                      bias=True),
            self.activation(**self.activ_args)
        )

        self.mu_layer = nn.Linear(hparams.encoder_filters[-2], self.z_size, bias=True)
        self.std_layer = nn.Linear(hparams.encoder_filters[-2], self.z_size, bias=True)
        
        # init model
        self.init_weights(init_weights)
        
        #self.conv = nn.Sequential(
        #    nn.Conv1d(in_channels=3, out_channels=64, kernel_size=1,
        #              bias=self.use_bias),
        #    nn.ReLU(inplace=True),
        #
        #    nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1,
        #              bias=self.use_bias),
        #    nn.ReLU(inplace=True),
        #
        #    nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1,
        #              bias=self.use_bias),
        #    nn.ReLU(inplace=True),
        #
        #    nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1,
        #              bias=self.use_bias),
        #    nn.ReLU(inplace=True),
        #
        #    nn.Conv1d(in_channels=256, out_channels=512, kernel_size=1,
        #              bias=self.use_bias),
        #)

        #self.fc = nn.Sequential(
        #    nn.Linear(512, 256, bias=True),
        #    nn.ReLU(inplace=True)
        #)

        #self.mu_layer = nn.Linear(256, self.z_size, bias=True)
        #self.std_layer = nn.Linear(256, self.z_size, bias=True)
    
    def init_weights(self, init_weights):
        if init_weights is None:
            self.conv.apply(_init_weights)
            self.fc.apply(_init_weights)
            self.mu_layer.apply(_init_weights)
            self.std_layer.apply(_init_weights)
        elif init_weights.endswith('.pt'):
            checkpoint = torch.load(init_weights, map_location='cpu')
            self.load_state_dict(checkpoint['encoder_state_dict'])
        
    def save_weights(self, path):
        torch.save(self.state_dict(), path)
        
    def load_weights(self, path):
        self.load_state_dict(torch.load(path))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def encode(self, x):
        self.eval()
        with torch.no_grad():
            return self(x)[1]

    def forward(self, x):
        output = self.conv(x)
        output2 = output.max(dim = 2)[0]
        logit = self.fc(output2)
        mu = self.mu_layer(logit)
        logvar = self.std_layer(logit)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        self.eval()
        with torch.no_grad():
            return self(x)[0]


class AAE3dModel(nn.Module):
    def __init__(self, num_points, num_features, hparams, devices, init_weights):
        super(AAE3dModel, self).__init__()

        # instantiate encoder, generator and discriminator
        self.encoder = Encoder(num_points, num_features, hparams, init_weights).to(devices[0])
        self.generator = Generator(num_points, num_features, hparams, init_weights).to(devices[1])
        self.discriminator = Discriminator(hparams, init_weights).to(devices[2])

    def forward(self, x):
        z, mu, logvar = self.encoder(x)
        x = self.generator(z)
        return x, mu, logvar
        
    def encode(self, x):
        z, mu, logvar = self.encoder(x)
        return z, mu, logvar

    def generate(self, z):
        x = self.generator(z)
        return x
        
    def discriminate(self, z):
        p = self.discriminator(z)
        return p
    
    def save_weights(self, enc_path, gen_path, disc_path):
        self.encoder.save_weights(enc_path)
        self.generator.save_weights(gen_path)
        self.discriminator.save_weights(disc_path)

    def load_weights(self, enc_path, dec_path):
        self.encoder.load_weights(enc_path)
        self.generator.load_weights(gen_path)
        self.discriminator.load_weights(disc_path)
        
class AAE3d(object):
    """
    Provides high level interface for training, testing and saving VAE
    models. Takes arbitrary encoder/decoder models specified by the choice
    of hyperparameters. Assumes the shape of the data is square.

    Attributes
    ----------
    model : torch.nn.Module (AAE3dModel)
        Underlying Pytorch model with encoder/decoder attributes.

    optimizer : torch.optim.Optimizer
        Pytorch optimizer used to train model.

    loss_func : function
        Loss function used to train model.

    Methods
    -------
    train(train_loader, valid_loader, epochs=1, checkpoint='', callbacks=[])
        Train model

    encode(x)
        Embed data into the latent space.

    generate(embedding)
        Generate matrices from embeddings.
    
    discriminator(embedding)
        Score embedding.

    save_weights(enc_path, gen_path, disc_path)
        Save encoder/generator/discriminator weights.

    load_weights(enc_path, gen_path, disc_path)
        Load saved encoder/generator/discriminator weights.
    """
    
    def __init__(self, num_points, num_features, batch_size,
                     hparams = AAE3dHyperparams(),
                     optimizer_hparams = OptimizerHyperparams(),
                     gpu = None,
                     init_weights=None,
                     verbose = True):
        """
        Parameters
        ----------
        num_points : integer
            number of points.

        hparams : molecules.ml.hyperparams.Hyperparams
            Defines the model architecture hyperparameters. Currently implemented
            are AAE3dHyperparams.

        optimizer_hparams : molecules.ml.hyperparams.OptimizerHyperparams
            Defines the optimizer type and corresponding hyperparameters.

        loss: : function, optional
            Defines an optional loss function with inputs (recon_x, x, mu, logvar)
            and ouput torch loss.

        init_weights : str, None
            If str and ends with .pt, init_weights is a model checkpoint from
            which pretrained weights can be loaded for the encoder, generator
            and discriminator. If None, model will start with default weight
            initialization.

        gpu : int, tuple, or None
            Encoder and decoder will train on ...
            If None, cuda GPU device if it is available, otherwise CPU.
            If int, the specified GPU.
            If tuple, the first and second GPUs respectively.

        verbose : bool
            True prints training and validation loss to stdout.
        """
        hparams.validate()
        optimizer_hparams.validate()

        # these are helpers for distributed computing
        self.comm_rank = 0
        self.comm_size = 1
        if dist.is_initialized():
            self.comm_rank = dist.get_rank()
            self.comm_size = dist.get_world_size()

        # verbose level
        self.verbose = verbose

        # Tuple of encoder, decoder device
        self.devices = Device(*self._configure_device(gpu))

        # model
        self.model = AAE3dModel(num_points, num_features, hparams, self.devices, init_weights)
        
        # fixed noise vector
        self.num_points = num_points
        self.num_features = num_features
        self.batch_size = batch_size
        self.noise_mu = hparams.noise_mu
        self.noise_std = hparams.noise_std
        self.fixed_noise = torch.FloatTensor(self.batch_size, hparams.latent_dim, 1).to(self.devices[2])
        self.fixed_noise.normal_(mean = self.noise_mu, std = self.noise_std)
        self.noise = torch.FloatTensor(self.batch_size, hparams.latent_dim).to(self.devices[2])

        # TODO: consider making optimizer_hparams a member variable
        # RMSprop with lr=0.001, alpha=0.9, epsilon=1e-08, decay=0.0
        self.optimizer_d = get_optimizer(self.model.discriminator.parameters(), optimizer_hparams)
        self.optimizer_eg = get_optimizer(chain(self.model.encoder.parameters(), self.model.generator.parameters()), 
                                        optimizer_hparams)

        # loss parameters
        self.lambda_gp = hparams.lambda_gp
        self.lambda_rec = hparams.lambda_rec
        self.rec_loss = cl.ChamferLoss()
    
    def _configure_device(self, gpu):
        """
        Configures GPU/CPU device for training AAE. Allows encoder
        and decoder to be trained on seperate devices.

        Parameters
        ----------
        gpu : int, tuple, or None
            Encoder and decoder will train on ...
            If None, cuda GPU device if it is available, otherwise CPU.
            If int, the specified GPU.
            If tuple, the first and second GPUs respectively or None
            option if tuple contains None.

        Returns
        -------
        2-tuple of encoder_device, generator_device
        """

        if (gpu is None) or (isinstance(gpu, tuple) and (None in gpu)):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            return device, device, device

        if not torch.cuda.is_available():
            raise ValueError("Specified GPU training but CUDA is not available.")

        if isinstance(gpu, int):
            device = torch.device(f"cuda:{gpu}")
            return device, device, device

        if isinstance(gpu, tuple) and len(gpu) == 3:
            return torch.device(f"cuda:{gpu[0]}"), torch.device(f"cuda:{gpu[1]}"), torch.device(f"cuda:{gpu[2]}")

        raise ValueError("Specified GPU device is invalid. Should be int, 3-tuple or None.")

    def __repr__(self):
        return str(self.model)

    def _loss_fnc_d(self, noise, real_logits, codes, fake_logits):

        handle = self.model
        if isinstance(handle, torch.nn.parallel.DistributedDataParallel):
            handle = handle.module

        # classification loss (critic)
        loss = torch.mean(fake_logits) - torch.mean(real_logits)
        
        # gradient penalty
        alpha = torch.rand(self.batch_size, 1).to(self.devices[2])
        interpolates = noise + alpha * (codes - noise)
        disc_interpolates = handle.discriminate(interpolates)
        
        gradients = torch.autograd.grad(outputs=disc_interpolates,
                                        inputs=interpolates,
                                        grad_outputs=torch.ones_like(disc_interpolates).to(self.devices[2]),
                                        create_graph=True,
                                        retain_graph=True,
                                        only_inputs=True)[0]
        slopes = torch.sqrt(torch.sum(gradients ** 2, dim=1))
        gradient_penalty = ((slopes - 1) ** 2).mean()
        loss += self.lambda_gp * gradient_penalty
        
        return loss
        
    def _loss_fnc_eg(self, data, rec_batch, fake_logit):        
        # reconstruction loss: here we need input shape (batch_size, num_points, points_dim)
        loss = self.lambda_rec * torch.mean(self.rec_loss(rec_batch.permute(0, 2, 1), data.permute(0, 2, 1)))

        # add generator loss if requested
        if fake_logit is not None:
            loss += -torch.mean(fake_logit)
        
        return loss

    def train(self, train_loader, valid_loader, epochs=1, checkpoint='', callbacks=[]):
        """
        Train model

        Parameters
        ----------
        train_loader : torch.utils.data.dataloader.DataLoader
            Contains training data

        valid_loader : torch.utils.data.dataloader.DataLoader
            Contains validation data

        epochs : int
            Number of epochs to train for

        checkpoint : str
            Path to checkpoint file to load and resume training
            from the epoch when the checkpoint was saved.

        callbacks : list
            Contains molecules.utils.callback.Callback objects
            which are called during training.
        """
        
        if callbacks:
            handle = self.model
            if isinstance(handle, torch.nn.parallel.DistributedDataParallel):
                handle = handle.module
            logs = {'model': handle, 'optimizer_d': self.optimizer_d, 'optimizer_eg': self.optimizer_eg}
            if dist.is_initialized():
                logs["comm_size"] = self.comm_size
        else:
            logs = {}

        start_epoch = 1

        if checkpoint:
            start_epoch += self._load_checkpoint(checkpoint)

        for callback in callbacks:
            callback.on_train_begin(logs)
        
        for epoch in range(start_epoch, epochs + 1):

            for callback in callbacks:
                callback.on_epoch_begin(epoch, logs)

            self._train(train_loader, epoch, callbacks, logs)
            self._validate(valid_loader, epoch, callbacks, logs)

            for callback in callbacks:
                callback.on_epoch_end(epoch, logs)

        for callback in callbacks:
            callback.on_train_end(logs)
        
    def _train(self, train_loader, epoch, callbacks, logs):
        """
        Train for 1 epoch

        Parameters
        ----------
        train_loader : torch.utils.data.dataloader.DataLoader
            Contains training data

        epoch : int
            Current epoch of training

        callbacks : list
            Contains molecules.utils.callback.Callback objects
            which are called during training.

        logs : dict
            Filled with data for callbacks
        """
        handle = self.model
        if isinstance(handle, torch.nn.parallel.DistributedDataParallel):
            handle = handle.module

        handle.train()
        train_loss_d = 0.
        train_loss_eg = 0.
        times = []
        for batch_idx, token in enumerate(train_loader):
            
            if self.verbose:
                start = time.time()
            
            if callbacks:
                pass # TODO: add more to logs

            for callback in callbacks:
                callback.on_batch_begin(batch_idx, epoch, logs)

            # copy to gpu
            data, rmsd, fnc, index = token
            data = data.to(self.devices[0])
                
            # get reconstruction
            codes, mu, logvar = handle.encode(data)
            
            # get noise
            self.noise.normal_(mean = self.noise_mu, std = self.noise_std)
            
            # get logits
            real_logits = handle.discriminate(self.noise)
            fake_logits = handle.discriminate(codes)
            
            # get loss
            loss_d = self._loss_fnc_d(self.noise, real_logits, codes, fake_logits)
            
            # backward pass
            self.optimizer_d.zero_grad()
            handle.discriminator.zero_grad()
            loss_d.backward(retain_graph = True)
            
            # optimizer step
            train_loss_d += loss_d.item()
            self.optimizer_d.step()

            # eg step
            rec_batch = handle.generate(codes)
            fake_logit = handle.discriminate(codes)
            
            # get loss
            loss_eg = self._loss_fnc_eg(data, rec_batch, fake_logit)
            
            # backward pass
            self.optimizer_eg.zero_grad()
            handle.generator.zero_grad()
            handle.encoder.zero_grad()
            loss_eg.backward()
            
            # optimizer step
            train_loss_eg += loss_eg.item()
            self.optimizer_eg.step()

            # Check memory footprint
            #print("memory allocation")
            #print(torch.cuda.memory_allocated()/1024**3)
            #print(torch.cuda.memory_summary())

            # Timing
            times.append(time.time() - start)
            if len(times) == 10:
                print(f"Avg time per 10 batches: {sum(times) / len(times)}")
                times = []

            if callbacks:
                logs['train_loss_d'] = loss_d.item()
                logs['train_loss_eg'] = loss_eg.item()
                logs['global_step'] = (epoch - 1) * len(train_loader) + batch_idx
            
            if self.verbose and (self.comm_rank == 0):
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss_d: {:.6f}\tLoss_eg: {:.6f}\tTime: {:.3f}'.format(
                      epoch, (batch_idx + 1) * self.comm_size * len(data),
                      self.comm_size * len(train_loader.dataset),
                      100. * (batch_idx + 1) / len(train_loader),
                      loss_d.item(), loss_eg.item(),
                      time.time() - start))

            for callback in callbacks:
                callback.on_batch_end(batch_idx, epoch, logs)

        # running loss over epoch
        train_loss_d_ave = train_loss_d / float(batch_idx + 1)
        train_loss_eg_ave = train_loss_eg / float(batch_idx + 1)

        if callbacks:
            logs['train_loss_d_average'] = train_loss_d_ave
            logs['train_loss_eg_average'] = train_loss_eg_ave

        if self.verbose and (self.comm_rank == 0):
            print('====> Epoch: {} Average loss_d: {:.4f} loss_eg: {:.4f}'.format(epoch, train_loss_d_ave, train_loss_eg_ave))

            
    def _validate(self, valid_loader, epoch, callbacks, logs):
        """
        Test model on validation set.

        Parameters
        ----------
        valid_loader : torch.utils.data.dataloader.DataLoader
            Contains validation data

        callbacks : list
            Contains molecules.utils.callback.Callback objects
            which are called during training.

        logs : dict
            Filled with data for callbacks
        """
        handle = self.model
        if isinstance(handle, torch.nn.parallel.DistributedDataParallel):
            handle = handle.module
        
        handle.eval()
        valid_loss = 0.
        for callback in callbacks:
            callback.on_validation_begin(epoch, logs)
        
        with torch.no_grad():
            for batch_idx, token in enumerate(valid_loader):
                # copy to gpu
                data, rmsd, fnc, index = token
                data = data.to(self.devices[0])
                # get reconstruction
                codes, mu, logvar = handle.encode(data)
                # just reconstruction loss is important here
                recons_batch = handle.generate(codes)
                valid_loss += self._loss_fnc_eg(data, recons_batch, None).item()

                for callback in callbacks:
                    callback.on_validation_batch_end(epoch,
                                                     batch_idx,
                                                     logs,
                                                     input = data.detach(),
                                                     rmsd = rmsd.detach(),
                                                     fnc = fnc.detach(),
                                                     mu = mu.detach(),
                                                     logvar = logvar.detach())
        
        valid_loss /= float(batch_idx + 1)

        if callbacks:
            logs['valid_loss'] = valid_loss

        for callback in callbacks:
            callback.on_validation_end(epoch, logs)

        if self.verbose and (self.comm_rank == 0):
            print('====> Validation loss: {:.4f}'.format(valid_loss))
    
    def _load_checkpoint(self, path):
        """
        Loads checkpoint file containing optimizer state and
        encoder/decoder weights.

        Parameters
        ----------
        path : str
            Path to checkpoint file

        Returns
        -------
        Epoch of training corresponding to the saved checkpoint.
        """

        # load checkpoint
        cp = torch.load(path, map_location='cpu')

        # get model handle
        handle = self.model
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            handle = handle.module

        # load model
        handle.encoder.load_state_dict(cp['encoder_state_dict'])
        handle.generator.load_state_dict(cp['generator_state_dict'])
        handle.discriminator.load_state_dict(cp['discriminator_state_dict'])

        # optimizers
        self.optimizer_eg.load_state_dict(cp['optimizer_eg_state_dict'])
        self.optimizer_d.load_state_dict(cp['optimizer_d_state_dict'])
        
        return cp['epoch']
        
    def encode(self, x):
        """
        Embed data into the latent space.

        Parameters
        ----------
        x : torch.Tensor
            Data to encode, could be a batch of data with dimension
            (batch_size, num_points, 3)

        Returns
        -------
        torch.Tensor of embeddings of shape (batch-size, latent_dim)
        """
        handle = self.model
        if isinstance(handle, torch.nn.parallel.DistributedDataParallel):
            handle = handle.module
        return handle.encode(x)

    def decode(self, embedding):
        """
        Generate matrices from embeddings.

        Parameters
        ----------
        embedding : torch.Tensor
            Embedding data, could be a batch of data with dimension
            (batch-size, latent_dim)

        Returns
        -------
        torch.Tensor of generated matrices of shape (batch-size, num_points, 3)
        """
        handle = self.model
        if isinstance(handle, torch.nn.parallel.DistributedDataParallel):
            handle = handle.module
        return handle.generate(embedding)
        
    def save_weights(self, enc_path, gen_path, disc_path):
        """
        Save encoder/generator/discriminator weights.

        Parameters
        ----------
        enc_path : str
            Path to save the encoder weights to.

        gen_path : str
            Path to save the generator weights to.
        
        disc_path: str
            Path to save the discriminator weights to.
        """
        handle = self.model
        if isinstance(handle, torch.nn.parallel.DistributedDataParallel):
            handle = handle.module
        handle.save_weights(enc_path, gen_path, disc_path)

    def load_weights(self, enc_path, gen_path, disc_path):
        """
        Load saved encoder/generator/discriminator weights.

        Parameters
        ----------
        enc_path : str
            Path to load the encoder weights from.

        gen_path : str
            Path to load the generator weights from.
            
        disc_path: str
            Path to load the discriminator weights from.
        """
        handle = self.model
        if isinstance(handle, torch.nn.parallel.DistributedDataParallel):
            handle = handle.module
        handle.load_weights(enc_path, gen_path, disc_path)
