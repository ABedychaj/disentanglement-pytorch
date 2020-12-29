import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn

from architectures import encoders, decoders
from common.wica_loss import WICA
from models.base.base_disentangler import BaseDisentangler


class AEModel(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return torch.sigmoid(self.decoder(z))

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


class AE(BaseDisentangler):

    def __init__(self, args):
        super().__init__(args)

        # encoder and decoder
        self.encoder_name = args.encoder[0]
        self.decoder_name = args.decoder[0]
        encoder = getattr(encoders, self.encoder_name)
        decoder = getattr(decoders, self.decoder_name)

        # model and optimizer
        self.model = AEModel(encoder(self.z_dim, self.num_channels, self.image_size),
                             decoder(self.z_dim, self.num_channels, self.image_size)).to(self.device)
        self.optim_G = optim.Adam(self.model.parameters(), lr=self.lr_G, betas=(self.beta1, self.beta2))

        # nets
        self.nets = [self.model]
        self.net_dict = {
            'G': self.model
        }
        self.optim_dict = {
            'optim_G': self.optim_G,
        }

        self.setup_schedulers(args.lr_scheduler, args.lr_scheduler_args,
                              args.w_recon_scheduler, args.w_recon_scheduler_args)

        # lambda
        self.lambda_wica = args.lambda_wica
        self.wica = args.wica_loss
        self.recon_lambda = args.recon_lambda
        self.loss = WICA(args.z_dim, args.number_of_gausses)

    def loss_fn(self, **kwargs):
        x_recon = kwargs['x_recon']
        x_true = kwargs['x_true']
        bs = self.batch_size
        recon_loss = F.binary_cross_entropy(x_recon, x_true, reduction='sum') / bs * self.w_recon

        return recon_loss

    def train(self):
        while not self.training_complete():
            self.net_mode(train=True)
            freeze_first_batch = True
            for x_true1, _ in self.data_loader:
                if freeze_first_batch is True:
                    freeze_first_batch = False
                    first_batch = x_true1.to(self.device)

                x_true1 = x_true1.to(self.device)

                # dummy nested dropout
                # with torch.no_grad():
                #     t = random.randint(5, 32)
                #     self.model.encoder.main[-1].weight[t:] = torch.zeros_like(self.model.encoder.main[-1].weight[t:])

                x_recon, z_latent = self.model(x_true1)
                if self.wica:
                    _, z_latent = self.model(first_batch)
                    w_loss = self.lambda_wica * self.loss.wica_loss(z_latent.data,
                                                                    latent_normalization=True,
                                                                    how="sqrt").to(self.device)
                else:
                    w_loss = 1

                recon_loss = self.recon_lambda * self.loss_fn(x_recon=x_recon, x_true=x_true1)
                full_loss = recon_loss * w_loss
                loss_dict = {'recon': recon_loss, 'wica': w_loss}

                self.optim_G.zero_grad()
                full_loss.backward(retain_graph=True)
                self.optim_G.step()

                self.log_save(loss=loss_dict,
                              input_image=x_true1,
                              recon_image=x_recon,
                              )
            # end of epoch
        self.pbar.close()

    def test(self):
        self.net_mode(train=False)
        for x_true1, _ in self.data_loader:
            x_true1 = x_true1.to(self.device)
            x_recon, z_latent = self.model(x_true1)

            self.visualize_recon(x_true1, x_recon, test=True)
            self.visualize_traverse(limit=(self.traverse_min, self.traverse_max), spacing=self.traverse_spacing,
                                    data=(x_true1, None), test=True)

            self.iter += 1
            self.pbar.update(1)
