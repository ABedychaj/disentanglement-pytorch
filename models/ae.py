import random

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn

from architectures import encoders, decoders
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
        self.number_of_gausses = args.number_of_gausses
        self.wica = args.wica_loss

    def loss_fn(self, **kwargs):
        x_recon = kwargs['x_recon']
        x_true = kwargs['x_true']
        bs = self.batch_size
        recon_loss = F.binary_cross_entropy(x_recon, x_true, reduction='sum') / bs * self.w_recon

        return recon_loss

    def random_choice_full(self, input, n_samples):
        from torch import multinomial, ones
        if n_samples * self.number_of_gausses < input.shape[0]:
            replacement = False
        else:
            replacement = True
        idx = multinomial(ones(input.shape[0]), n_samples * self.number_of_gausses, replacement=replacement)
        sampled = input[idx].reshape(self.number_of_gausses, n_samples, -1)
        return torch.mean(sampled, axis=1)

    def wica_loss(self, z, latent_normalization=False):
        from torch.distributions import MultivariateNormal

        if latent_normalization:
            x = (z - z.mean(dim=1, keepdim=True)) / z.std(dim=1, keepdim=True)
        else:
            x = z
        dim = self.z_dim if self.z_dim is not None else x.shape[1]
        scale = (1 / dim)
        sampled_points = self.random_choice_full(x, dim)
        cov_mat = (scale * torch.eye(dim)).repeat(self.number_of_gausses, 1, 1)

        mvn = MultivariateNormal(loc=sampled_points.to(self.device),
                                 covariance_matrix=cov_mat.to(self.device))
        weight_vector = torch.exp(mvn.log_prob(x.reshape(-1, 1, dim).to(self.device)))

        sum_of_weights = torch.sum(weight_vector, axis=0)
        weight_sum = torch.sum(x * weight_vector.T.reshape(self.number_of_gausses, -1, 1), axis=1)
        weight_mean = weight_sum / sum_of_weights.reshape(-1, 1)

        xm = x - weight_mean.reshape(self.number_of_gausses, 1, -1)
        wxm = xm * weight_vector.T.reshape(self.number_of_gausses, -1, 1)

        wcov = (wxm.permute(0, 2, 1).matmul(xm)) / sum_of_weights.reshape(-1, 1, 1)

        diag = torch.diagonal(wcov ** 2, dim1=1, dim2=2)
        diag_pow_plus = diag.reshape(diag.shape[0], diag.shape[1], -1) + diag.reshape(diag.shape[0], -1, diag.shape[1])

        tmp = (2 * wcov ** 2 / diag_pow_plus)

        triu = torch.triu(tmp, diagonal=1)
        normalize = 2.0 / (dim * (dim - 1))
        cost = torch.sum(normalize * triu) / self.number_of_gausses
        return cost

    def train(self):
        while not self.training_complete():
            self.net_mode(train=True)
            for x_true1, _ in self.data_loader:
                x_true1 = x_true1.to(self.device)

                # dummy nested dropout
                # with torch.no_grad():
                #     t = random.randint(5, 32)
                #     self.model.encoder.main[-1].weight[t:] = torch.zeros_like(self.model.encoder.main[-1].weight[t:])

                x_recon, z_latent = self.model(x_true1)
                if self.wica:
                    w_loss = self.lambda_wica * self.wica_loss(z_latent.data, latent_normalization=True).to(self.device)
                else:
                    w_loss = 1

                recon_loss = self.loss_fn(x_recon=x_recon, x_true=x_true1) * w_loss
                loss_dict = {'recon': recon_loss, 'wica_loss': w_loss}

                self.optim_G.zero_grad()
                recon_loss.backward(retain_graph=True)
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
