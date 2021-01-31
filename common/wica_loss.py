import torch


def random_choice_full(input, n_samples, number_of_gausses, mean=None):
    from torch import multinomial, ones
    if n_samples * number_of_gausses < input.shape[0]:
        replacement = False
    else:
        replacement = True
    idx = multinomial(ones(input.shape[0]), n_samples * number_of_gausses, replacement=replacement)
    sampled = input[idx].reshape(number_of_gausses, n_samples, -1)

    if mean is None:
        return sampled
    else:
        return torch.mean(sampled, axis=1)


def provide_weights_for_x(x, how=None, device=None, n_samples=None, times=None, mean=None):
    dim = x.shape[1]

    if n_samples is None:
        n_samples = dim
    if times is None:
        times = dim

    scale = (1 / dim)

    sampled_points = random_choice_full(x, n_samples, times, mean=mean)

    if how == "gauss":
        from torch.distributions import MultivariateNormal

        cov_mat = (scale * torch.eye(dim)).repeat(dim, 1, 1)
        mvn = MultivariateNormal(loc=sampled_points.to(device), covariance_matrix=cov_mat.to(device))
        weight_vector = torch.exp(mvn.log_prob(x.reshape(-1, 1, dim).to(device)))

    elif how == "sqrt":
        weight_vector = torch.sqrt(1 + sampled_points.reshape(-1, 1, dim).to(device) ** 2) ** (-1)

    elif how == "log":
        weight_vector = torch.log(1 + sampled_points.reshape(-1, 1, dim).to(device) ** 2)

    elif how == "TStudent":
        from torch.distributions.studentT import StudentT

        mvn = StudentT(df=1, loc=x.mean(0), scale=scale)
        weight_vector = torch.exp(mvn.log_prob(x.reshape(-1, 1, dim).to(device)))
        # to trzeba poprawić ?!
    elif how == "Cauchy":
        from torch.distributions.cauchy import Cauchy

        mvn = Cauchy(loc=x.mean(0), scale=1)
        weight_vector = torch.exp(mvn.log_prob(x.reshape(-1, 1, dim).to(device)))
    elif how == "Gumbel":
        from torch.distributions.gumbel import Gumbel

        mvn = Gumbel(loc=x.mean(0), scale=1)
        weight_vector = torch.exp(mvn.log_prob(x.reshape(-1, 1, dim).to(device)))
    elif how == "Laplace":
        from torch.distributions.laplace import Laplace

        mvn = Laplace(loc=x.mean(0), scale=1)
        weight_vector = torch.exp(mvn.log_prob(x.reshape(-1, 1, dim).to(device)))
    return weight_vector


class WICA(object):
    def __init__(self, z_dim, number_of_gausses):
        self.number_of_gausses = number_of_gausses
        self.z_dim = z_dim
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def wica_loss(self, z, latent_normalization=False, how="gauss", mean=None):
        z1 = z[:len(z) // 2]
        z2 = z[len(z) // 2 + 1:]

        if latent_normalization:
            x1 = (z1 - z1.mean(dim=1, keepdim=True)) / z1.std(dim=1, keepdim=True)
            x2 = (z2 - z2.mean(dim=1, keepdim=True)) / z2.std(dim=1, keepdim=True)
        else:
            x1 = z1
            x2 = z2

        dim = self.z_dim if self.z_dim is not None else x.shape[1]

        weight_vector = provide_weights_for_x(
            x=x1,
            how=how,
            device=self.device,
            mean=mean
        )

        sum_of_weights = torch.sum(weight_vector, axis=0)

        weight_sum = torch.sum(x2.reshape(1, x2.shape[0], x2.shape[1]) * weight_vector, axis=0)

        weight_mean = weight_sum / sum_of_weights

        xm = x2 - weight_mean
        wxm = torch.sum(xm.reshape(1, xm.shape[0], xm.shape[1]) * weight_vector, axis=0)

        wcov = (wxm.reshape(1, wxm.shape[0], wxm.shape[1]).permute(0, 2, 1).matmul(xm)) / sum_of_weights

        diag = torch.diagonal(wcov ** 2, dim1=1, dim2=2)
        diag_pow_plus = diag.reshape(diag.shape[0], diag.shape[1], -1) + diag.reshape(diag.shape[0], -1, diag.shape[1])

        tmp = (2 * wcov ** 2 / diag_pow_plus)

        triu = torch.triu(tmp, diagonal=1)
        normalize = 2.0 / (dim * (dim - 1))
        cost = torch.sum(normalize * triu) / self.number_of_gausses
        return cost
