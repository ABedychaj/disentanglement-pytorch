import torch


def random_choice_full(input, n_samples, number_of_gausses):
    from torch import multinomial, ones
    if n_samples * number_of_gausses < input.shape[0]:
        replacement = False
    else:
        replacement = True
    idx = multinomial(ones(input.shape[0]), n_samples * number_of_gausses, replacement=replacement)
    sampled = input[idx].reshape(number_of_gausses, n_samples, -1)
    return torch.mean(sampled, axis=1)


def provide_weights_for_x(x, how=None, device=None):
    dim = x.shape[1]
    scale = (1 / dim)

    if how == "gauss":
        from torch.distributions import MultivariateNormal

        sampled_points = random_choice_full(x, dim, dim)
        cov_mat = (scale * torch.eye(dim)).repeat(dim, 1, 1)
        mvn = MultivariateNormal(loc=sampled_points.to(device), covariance_matrix=cov_mat.to(device))
        weight_vector = torch.exp(mvn.log_prob(x.reshape(-1, 1, dim).to(device)))

    elif how == "sqrt":
        weight_vector = 1 / torch.sqrt(1 + x.reshape(-1, 1, dim).to(device) ** 2)

    elif how == "log":
        weight_vector = torch.log(1 + x.reshape(-1, 1, dim).to(device) ** 2)

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

    def wica_loss(self, z, latent_normalization=False, how="gauss"):
        if latent_normalization:
            x = (z - z.mean(dim=1, keepdim=True)) / z.std(dim=1, keepdim=True)
        else:
            x = z
        dim = self.z_dim if self.z_dim is not None else x.shape[1]

        weight_vector = provide_weights_for_x(
            x=x,
            how=how,
            device=self.device
        )

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
