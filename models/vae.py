import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn

from architectures import encoders, decoders
from common import constants as c
from common.ops import kl_divergence_mu0_var1, reparametrize
from common.wica_loss import WICA
from models.base.base_disentangler import BaseDisentangler


class VAEModel(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def encode(self, x, **kwargs):
        return self.encoder(x)

    def decode(self, z, **kwargs):
        return torch.sigmoid(self.decoder(z))

    def forward(self, x, **kwargs):
        mu, logvar = self.encode(x)
        z = reparametrize(mu, logvar)
        return self.decode(z)


class VAE(BaseDisentangler):
    """
    Auto-Encoding Variational Bayes
    by Kingma and Welling
    https://arxiv.org/pdf/1312.6114.pdf
    """

    def __init__(self, args):
        super().__init__(args)

        # hyper-parameters
        self.w_kld = args.w_kld
        self.max_c = torch.tensor(args.max_c, dtype=torch.float)
        self.iterations_c = torch.tensor(args.iterations_c, dtype=torch.float)

        # As a little joke
        assert self.w_kld == 1.0 or self.alg != 'VAE', 'in vanilla VAE, w_kld should be 1.0. ' \
                                                       'Please use BetaVAE if intended otherwise.'

        # encoder and decoder
        encoder_name = args.encoder[0]
        decoder_name = args.decoder[0]
        encoder = getattr(encoders, encoder_name)
        decoder = getattr(decoders, decoder_name)

        # model and optimizer
        self.model = VAEModel(encoder(self.z_dim, self.num_channels, self.image_size),
                              decoder(self.z_dim, self.num_channels, self.image_size)).to(self.device)
        self.optim_G = optim.Adam(self.model.parameters(), lr=self.lr_G, betas=(self.beta1, self.beta2))

        # nets
        self.net_dict = {
            'G': self.model
        }
        self.optim_dict = {
            'optim_G': self.optim_G,
        }

        # FactorVAE
        if c.FACTORVAE in self.loss_terms:
            from models.factorvae import factorvae_init
            self.PermD, self.optim_PermD = factorvae_init(args.discriminator[0], self.z_dim, self.num_layer_disc,
                                                          self.size_layer_disc, self.lr_D, self.beta1, self.beta2)
            self.PermD.to(self.device)
            self.net_dict.update({'PermD': self.PermD})
            self.optim_dict.update({'optim_PermD': self.optim_PermD})

        self.setup_schedulers(args.lr_scheduler, args.lr_scheduler_args,
                              args.w_recon_scheduler, args.w_recon_scheduler_args)

        # WICA
        self.lambda_wica = args.lambda_wica
        self.wica = args.wica_loss
        self.loss = WICA(args.z_dim, args.number_of_gausses)

    def encode_deterministic(self, **kwargs):
        images = kwargs['images']
        if images.dim() == 3:
            images = images.unsqueeze(0)
        mu, logvar = self.model.encode(x=images)
        return mu

    def encode_stochastic(self, **kwargs):
        images = kwargs['images']
        if images.dim() == 3:
            images = images.unsqueeze(0)
        mu, logvar = self.model.encode(x=images)
        return reparametrize(mu, logvar)

    def _kld_loss_fn(self, mu, logvar):
        if not self.controlled_capacity_increase:
            kld_loss = kl_divergence_mu0_var1(mu, logvar) * self.w_kld
        else:
            """
            Based on: Understanding disentangling in β-VAE
            https://arxiv.org/pdf/1804.03599.pdf
            """
            capacity = torch.min(self.max_c, self.max_c * torch.tensor(self.iter) / self.iterations_c)
            kld_loss = (kl_divergence_mu0_var1(mu, logvar) - capacity).abs() * self.w_kld
        return kld_loss

    def loss_fn(self, input_losses, **kwargs):
        x_recon = kwargs['x_recon']
        x_true = kwargs['x_true']
        mu = kwargs['mu']
        logvar = kwargs['logvar']
        mu_fixed = kwargs['mu_fixed']

        bs = self.batch_size
        output_losses = dict()
        output_losses[c.TOTAL_VAE] = input_losses.get(c.TOTAL_VAE, 0)

        output_losses[c.RECON] = F.binary_cross_entropy(x_recon, x_true, reduction='sum') / bs * self.w_recon
        output_losses[c.TOTAL_VAE] += 0 * output_losses[c.RECON]

        output_losses['kld'] = self._kld_loss_fn(mu, logvar)
        output_losses[c.TOTAL_VAE] += 0 * output_losses['kld']

        if self.wica:
            # only first batch
            # output_losses['wica_on_first_batch_gauss'] = self.lambda_wica * self.loss.wica_loss(mu_fixed.data,
            #                                                                                     latent_normalization=True,
            #                                                                                     how="gauss"
            #                                                                                     ).to(
            #     self.device)

            # 1/sqrt(1+x^2)
            # output_losses['wica_on_first_batch_sqrt'] = self.lambda_wica * self.loss.wica_loss(mu_fixed.data,
            #                                                                                    latent_normalization=True,
            #                                                                                    how="sqrt"
            #                                                                                    ).to(
            #     self.device)
            #
            # output_losses['wica_on_first_batch_sqrt_mean_true'] = self.lambda_wica * self.loss.wica_loss(mu_fixed.data,
            #                                                                                              latent_normalization=True,
            #                                                                                              how="sqrt",
            #                                                                                              mean=True
            #                                                                                              ).to(
            #     self.device)

            # log(1+x^2)
            # output_losses['wica_on_first_batch_log'] = self.lambda_wica * self.loss.wica_loss(mu_fixed.data,
            #                                                                                   latent_normalization=True,
            #                                                                                   how="log"
            #                                                                                   ).to(
            #     self.device)

            # TStudent
            # output_losses['wica_on_first_batch_TStudent'] = self.lambda_wica * self.loss.wica_loss(mu_fixed.data,
            #                                                                                        latent_normalization=True,
            #                                                                                        how="TStudent"
            #                                                                                        ).to(
            #     self.device)

            # Cauchy
            # output_losses['wica_on_first_batch_Cauchy'] = self.lambda_wica * self.loss.wica_loss(mu_fixed.data,
            #                                                                                      latent_normalization=True,
            #                                                                                      how="Cauchy"
            #                                                                                      ).to(
            #     self.device)

            # Gumbel
            # output_losses['wica_on_first_batch_Gumbel'] = self.lambda_wica * self.loss.wica_loss(mu_fixed.data,
            #                                                                                      latent_normalization=True,
            #                                                                                      how="Gumbel"
            #                                                                                      ).to(
            #     self.device)

            # Laplace
            # output_losses['wica_on_first_batch_Laplace'] = self.lambda_wica * self.loss.wica_loss(mu_fixed.data,
            #                                                                                       latent_normalization=True,
            #                                                                                       how="Laplace"
            #                                                                                       ).to(
            #     self.device)

            # on every batch
            # output_losses['wica_on_each_batch'] = self.lambda_wica * self.loss.wica_loss(mu.data,
            #                                                                              latent_normalization=True,
            #                                                                              how="gauss"
            #                                                                              ).to(
            #     self.device)

            # 1/sqrt(1+x^2)
            # output_losses['wica_on_each_batch_sqrt'] = self.lambda_wica * self.loss.wica_loss(mu.data,
            #                                                                                   latent_normalization=True,
            #                                                                                   how="sqrt"
            #                                                                                   ).to(
            #     self.device)

            output_losses['wica_on_each_batch_sqrt_mean_true'] = self.lambda_wica * self.loss.wica_loss(mu.data,
                                                                                                        latent_normalization=True,
                                                                                                        how="sqrt",
                                                                                                        mean=True
                                                                                                        ).to(
                self.device)

            # log(1+x^2)
            # output_losses['wica_on_each_batch_log'] = self.lambda_wica * self.loss.wica_loss(mu.data,
            #                                                                                  latent_normalization=True,
            #                                                                                  how="log"
            #                                                                                  ).to(
            #     self.device)

            # TStudent
            # output_losses['wica_on_each_batch_TStudent'] = self.lambda_wica * self.loss.wica_loss(mu.data,
            #                                                                                       latent_normalization=True,
            #                                                                                       how="TStudent"
            #                                                                                       ).to(
            #     self.device)

            # Cauchy
            # output_losses['wica_on_each_batch_Cauchy'] = self.lambda_wica * self.loss.wica_loss(mu.data,
            #                                                                                     latent_normalization=True,
            #                                                                                     how="Cauchy"
            #                                                                                     ).to(
            #     self.device)

            # Gumbel
            # output_losses['wica_on_each_batch_Gumbel'] = self.lambda_wica * self.loss.wica_loss(mu.data,
            #                                                                                     latent_normalization=True,
            #                                                                                     how="Gumbel"
            #                                                                                     ).to(
            #     self.device)

            # Laplace
            # output_losses['wica_on_each_batch_Laplace'] = self.lambda_wica * self.loss.wica_loss(mu.data,
            #                                                                                      latent_normalization=True,
            #                                                                                      how="Laplace"
            #                                                                                      ).to(
            #     self.device)

            output_losses[c.TOTAL_VAE] += output_losses['wica_on_each_batch_sqrt_mean_true']

        if c.FACTORVAE in self.loss_terms:
            from models.factorvae import factorvae_loss_fn
            output_losses['vae_tc_factor'], output_losses['discriminator_tc'] = factorvae_loss_fn(
                self.w_tc, self.model, self.PermD, self.optim_PermD, self.ones, self.zeros, **kwargs)
            # output_losses[c.TOTAL_VAE] += output_losses['vae_tc_factor']

        if c.DIPVAEI in self.loss_terms:
            from models.dipvae import dipvaei_loss_fn
            output_losses['vae_dipi'] = dipvaei_loss_fn(self.w_dipvae, self.lambda_od, self.lambda_d, **kwargs)
            output_losses[c.TOTAL_VAE] += output_losses['vae_dipi']

        if c.DIPVAEII in self.loss_terms:
            from models.dipvae import dipvaeii_loss_fn
            output_losses['vae_dipii'] = dipvaeii_loss_fn(self.w_dipvae, self.lambda_od, self.lambda_d, **kwargs)
            output_losses[c.TOTAL_VAE] += output_losses['vae_dipii']

        if c.BetaTCVAE in self.loss_terms:
            from models.betatcvae import betatcvae_loss_fn
            output_losses['vae_betatc'] = betatcvae_loss_fn(self.w_tc, **kwargs)
            output_losses[c.TOTAL_VAE] += output_losses['vae_betatc']

        if c.INFOVAE in self.loss_terms:
            from models.infovae import infovae_loss_fn
            output_losses['vae_mmd'] = infovae_loss_fn(self.w_infovae, self.z_dim, self.device, **kwargs)
            output_losses[c.TOTAL_VAE] += output_losses['vae_mmd']

        return output_losses

    def vae_base(self, losses, x_true1, x_true2, label1, label2, first_batch, first_batch_label):
        mu, logvar = self.model.encode(x=x_true1, c=label1)
        z = reparametrize(mu, logvar)
        x_recon = self.model.decode(z=z, c=label1)

        mu_fixed, logvar_fixed = self.model.encode(x=first_batch, c=first_batch_label)
        z_fixed = reparametrize(mu, logvar_fixed)

        loss_fn_args = dict(x_recon=x_recon, x_true=x_true1, mu=mu, logvar=logvar, z=z,
                            x_true2=x_true2, label2=label2, mu_fixed=mu_fixed, logvar_fixed=logvar_fixed,
                            z_fixed=z_fixed)

        losses.update(self.loss_fn(losses, **loss_fn_args))
        return losses, {'x_recon': x_recon, 'mu': mu, 'z': z, 'logvar': logvar}

    def train(self):
        torch.autograd.set_detect_anomaly(True)
        while not self.training_complete():
            self.net_mode(train=True)
            vae_loss_sum = 0
            freeze_first_batch = True
            for internal_iter, (x_true1, label1) in enumerate(self.data_loader):
                losses = dict()
                x_true1 = x_true1.to(self.device)
                label1 = label1.to(self.device)
                x_true2, label2 = next(iter(self.data_loader))
                x_true2 = x_true2.to(self.device)
                label2 = label2.to(self.device)

                if freeze_first_batch is True:
                    freeze_first_batch = False
                    first_batch = x_true1.to(self.device)
                    first_batch_label = label1.to(self.device)

                losses, params = self.vae_base(losses, x_true1, x_true2, label1, label2, first_batch, first_batch_label)

                self.optim_G.zero_grad()
                losses[c.TOTAL_VAE].backward(retain_graph=False)
                vae_loss_sum += losses[c.TOTAL_VAE]
                losses[c.TOTAL_VAE_EPOCH] = vae_loss_sum / internal_iter

                self.optim_G.step()
                self.log_save(input_image=x_true1, recon_image=params['x_recon'], loss=losses)
            # end of epoch
        self.pbar.close()

    def test(self):
        self.net_mode(train=False)
        for x_true, label in self.data_loader:
            x_true = x_true.to(self.device)
            label = label.to(self.device, dtype=torch.long)

            x_recon = self.model(x=x_true, c=label)

            self.visualize_recon(x_true, x_recon, test=True)
            self.visualize_traverse(limit=(self.traverse_min, self.traverse_max), spacing=self.traverse_spacing,
                                    data=(x_true, label), test=True)

            self.iter += 1
            self.pbar.update(1)
