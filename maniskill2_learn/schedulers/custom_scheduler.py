import numpy as np
from numbers import Number
from maniskill2_learn.utils.data import is_seq_of, is_dict, is_str, is_num, auto_pad_seq, deepcopy

from maniskill2_learn.utils.meta import Registry, build_from_cfg


SCHEDULERS = Registry("scheduler of hyper-parameters")


class BaseScheduler:
    def __init__(self, init_values=None):
        self.niter = 0
        self.init_values = init_values

    def reset(self):
        self.niter = 0

    def get(self, *args, **kwargs):
        raise NotImplementedError

    def step(self, *args, **kwargs):
        self.niter += 1
        return self.get(*args, **kwargs)


@SCHEDULERS.register_module()
class FixedScheduler(BaseScheduler):
    def get(self, value=None, niter=None):
        return self.init_values if value is None else value


"""
@SCHEDULERS.register_module()
class StepScheduler:
    def __init__(self, keys, gammas, steps):
        # para = para * gamma ^ (# step)
        if keys is None:
            # For all parameters
            self.keys = None
            self.gammas = gammas
            self.steps = steps
            sorted(self.step)
            assert is_num(gammas) and is_seq_of(steps, Number)
            print('For all paramters, we will use step scheduler')
            print(f'We will multiply all parameters with {gammas} when we are at step {self.steps}!')
        else:
            if is_str(keys):
                keys = [keys, ]
            if is_seq_of(gammas, Number):
                gammas = [gammas, ]
            self.num_params = len(keys)
            self.keys = keys
            # Paded gamma to a list of parameters
            self.gammas = auto_pad_seq(keys, gammas)
            self.steps = auto_pad_seq(keys, steps)
            for i in len(self.steps):
                if is_num(self.steps[i]):
                    self.steps[i] = [self.steps[i], ]
                self.steps[i] = sorted(self.steps[i])
            assert is_seq_of(steps, (list, tuple))
            assert len(self.gamma) == len(self.steps) == len(self.keys) == self.num_params

    def get(self, kwargs, n_iter):
        assert is_dict(kwargs)
        ret = {}
        if self.keys is None:
            step_index = np.searchsorted(self.steps, n_iter, side='right')
            if is_num(kwargs):
                return kwargs * (self.gammas ** step_index)
            else:
                ret = {}
                for key in kwargs:
                    ret[key] = kwargs[key] * (self.gammas ** step_index)
                return ret
        else:
            for k, s, g in zip(self.keys, self.gammas, self.steps):
                if k not in kwargs:
                    continue
                step_index = np.searchsorted(s, n_iter, side='right')
                ret[k] = kwargs[k] * (g ** step_index)
        return ret
"""


@SCHEDULERS.register_module()
class LmbdaScheduler(BaseScheduler):
    """
    Tune the hyper-parameter by the running steps
    """

    def __init__(self, lmbda, init_values=None):
        super(LmbdaScheduler, self).__init__(init_values)
        self.lmbda = lmbda

    def get(self, init_values=None, niter=None):
        niter = self.niter if niter is None else niter
        if self.init_values is None:
            self.init_values = init_values
        return self.lmbda(init_values, niter)


@SCHEDULERS.register_module()
class StepScheduler(BaseScheduler):
    def __init__(self, steps, gamma, init_values=None):
        super(StepScheduler, self).__init__(init_values)
        self.steps = np.sort(steps)
        self.gamma = gamma
        print(self.steps, gamma)

    def get(self, init_values=None, niter=None):
        niter = self.niter if niter is None else niter
        if self.init_values is None:
            self.init_values = init_values
        init_values = self.init_values
        step_index = np.searchsorted(self.steps, niter, side="right")
        gamma = self.gamma**step_index
        if is_num(init_values):
            return init_values * gamma
        elif isinstance(init_values, (tuple, list)):
            ret = []
            for x in init_values:
                ret.append(x * gamma)
            return type(init_values)(ret)
        else:
            ret = {}
            for key in init_values:
                ret[key] = init_values[key] * gamma
            return ret


@SCHEDULERS.register_module()
class KeyStepScheduler(BaseScheduler):
    def __init__(self, keys, steps, gammas, init_values=None):
        super(KeyStepScheduler, self).__init__(init_values)
        if is_str(keys):
            keys = [
                keys,
            ]
        if is_num(gammas):
            gammas = [
                gammas,
            ]
        if is_num(steps):
            steps = [
                [
                    steps,
                ]
            ]
        elif is_seq_of(steps, Number):
            steps = [
                steps,
            ]
        self.infos = {}
        for i, key in enumerate(keys):
            gamma = gammas[min(i, len(gammas) - 1)]
            step = steps[min(i, len(steps) - 1)]
            self.infos[key] = (deepcopy(step), gamma)
        print(self.infos)

    def get(self, init_values=None, niter=None):
        niter = self.niter if niter is None else niter
        ret_values = dict() if init_values is None else init_values
        if self.init_values is None:
            assert isinstance(init_values, dict)
            self.init_values = {key: init_values[key] for key in self.infos if key in init_values}
        init_values = self.init_values
        for key in self.infos:
            steps, gamma = self.infos[key]
            step_index = np.searchsorted(steps, niter, side="right")
            # print(init_values[key], gamma, step_index)
            ret_values[key] = init_values[key] * (gamma**step_index)
        return ret_values


"""
@SCHEDULERS.register_module()
class MixedScheduler:
    def __init__(self, configs):
        assert isinstance(configs, (list, tuple))
        self.schedulers = []
        for cfg in configs:
            self.schedulers.append(build_hyper_para_scheduler(cfg))

    def get(self, kwargs, n_iter):
        assert is_dict(kwargs)
        ret = {}
        for scheduler in self.schedulers:
            ret.update(scheduler.get(n_iter, kwargs))
        return ret
"""


def build_scheduler(cfg, default_args=None):
    return build_from_cfg(cfg, SCHEDULERS, default_args)
