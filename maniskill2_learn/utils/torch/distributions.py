from torch.distributions import Normal, Independent, TanhTransform, TransformedDistribution  # , ComposeTransform, AffineTransform
from torch.distributions import Categorical
from torch import distributions as pyd
import torch

# import math
import torch.nn.functional as F


class TransformedNormal(TransformedDistribution):
    def _transform(self, x):
        for transform in self.transforms:
            x = transform(x)
        return x

    @property
    def mean(self):
        return self._transform(self.base_dist.mean)

    @property
    def stddev(self):
        return None


class CustomCategorical(Categorical):
    def __init__(self, *args, **kwargs):
        super(CustomCategorical, self).__init__(*args, **kwargs)

    def log_prob(self, value):
        logits_dim = self.logits.ndim
        if value.ndim == logits_dim:
            assert value.shape[-1] == 1, f"Shape error {value.shape}"
            value = value[..., 0]
        return super(CustomCategorical, self).log_prob(value)


class ScaledNormal(Normal):
    def __init__(self, mean, std, scale_prior, bias_prior):
        super(ScaledNormal, self).__init__(mean * scale_prior + bias_prior, std * scale_prior)
        self.scale_prior, self.bias_prior = scale_prior, bias_prior

    def rsample_with_log_prob(self, sample_shape=torch.Size()):
        ret = self.rsample(sample_shape)
        log_p = self.log_prob(ret)
        return ret, log_p


class ScaledTanhNormal(Normal):
    def __init__(self, mean, std, scale_prior, bias_prior, epsilon=1e-6):
        # import time
        # st = time.time()
        super(ScaledTanhNormal, self).__init__(mean, std)
        self.scale_prior, self.bias_prior = scale_prior, bias_prior
        self.epsilon = epsilon
        # print('Build Nomral', time.time() - st)

    def log_prob_with_logit(self, x):
        log_prob = super(ScaledTanhNormal, self).log_prob(x)
        """
        if log_prob.isinf().any():
            print(log_prob.shape)
            is_inf = torch.where(log_prob.isinf().any(-1))[0][0]
            print(is_inf)
            # exit(0)

            # print(is_inf.shape)
            # exit(0)
            print(x[is_inf], super(ScaledTanhNormal, self).mean[is_inf], super(ScaledTanhNormal, self).stddev[is_inf])
            print('x log_prob', x, log_prob)
            print('Log P nan', log_prob.isinf().sum())
            exit(0)
        """
        # Probability of scale * tah(x) is scale * (1 - tanh(x) ** 2) * d(x), where tanh'(x) = (1 - tanh(x) ** 2)
        log_prob -= torch.log(self.scale_prior * (1 - torch.tanh(x).pow(2)) + self.epsilon)
        return log_prob

    def log_prob(self, x):
        # Independent will not keep the last dimension
        x_ = self.un_transform(x)
        """
        isinf = x_.isinf()
        if isinf.any():
            isinf = torch.where(isinf)[0][0]
            print(isinf)
            print(x[isinf], x_[isinf])
            exit(0)
        """
        log_p = self.log_prob_with_logit(x_)
        return log_p

    def rsample(self, sample_shape=torch.Size()):
        return self.transform(super(ScaledTanhNormal, self).rsample(sample_shape))

    def sample(self, sample_shape=torch.Size()):
        return self.transform(super(ScaledTanhNormal, self).sample(sample_shape))

    @property
    def mean(self):
        return self.transform(super(ScaledTanhNormal, self).mean)

    def transform(self, x):
        return torch.tanh(x) * self.scale_prior + self.bias_prior

    def un_transform(self, x):
        return torch.atanh((x - self.bias_prior) / self.scale_prior)

    def rsample_with_log_prob(self, sample_shape=torch.Size()):
        logit = super(ScaledTanhNormal, self).rsample(sample_shape)
        log_prob = self.log_prob_with_logit(logit)
        return self.transform(logit), log_prob


class CustomIndependent(Independent):
    # def log_prob(self, value):
    # log_prob = super(CustomIndependent, self).log_prob(value)
    # return torch.unsqueeze(log_prob, self.reinterpreted_batch_ndims)

    def rsample_with_log_prob(self, sample_shape=torch.Size()):
        # import time
        # st = time.time()
        sample, log_prob = self.base_dist.rsample_with_log_prob(sample_shape)
        # print('Ind-1', time.time() - st)
        from torch.distributions.utils import _sum_rightmost

        return sample, _sum_rightmost(log_prob, self.reinterpreted_batch_ndims)
        # print('Ind-2', time.time() - st)
        # return ret

    @property
    def stddev(self):
        if hasattr(self.base_dist, "stddev"):
            return self.base_dist.stddev
        else:
            return self.base_dist.variance.sqrt()
