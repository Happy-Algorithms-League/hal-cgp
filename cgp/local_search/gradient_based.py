import numpy as np

try:
    import torch  # noqa: F401
    from torch.optim.optimizer import Optimizer  # noqa: F401

    torch_available = True
except ModuleNotFoundError:
    torch_available = False

from typing import Callable, List, Optional, Union


from ..individual import IndividualBase


def gradient_based(
    individual: IndividualBase,
    objective: Callable[[Union["torch.nn.Module", List["torch.nn.Module"]]], "torch.Tensor"],
    lr: float,
    gradient_steps: int,
    optimizer: Optional["Optimizer"] = None,
    clip_value: Optional[float] = None,
) -> None:
    """Perform a local search for numeric leaf values for an individual
    based on gradient information obtained via automatic
    differentiation.

    Parameters
    ----------
    individual : Individual
        Individual for which to perform local search.
    objective : Callable
        Objective function that is called with a differentiable graph
        and returns a differentiable loss.
    lr : float
        Learning rate for optimizer.
    gradient_steps : int
        Number of gradient steps per individual.
    optimizer : torch.optim.Optimizer, optional
        Optimizer to use for parameter updates. Defaults to
        torch.optim.SGD.
    clip_value : float, optional
        Clipping value for gradients. Clipping is skipped when set to
        np.inf. Defaults to 10% of the inverse of the learning rate so
        that the maximal update step is 0.1.

    """
    if not torch_available:
        raise ModuleNotFoundError("No module named 'torch' (extra requirement)")

    if optimizer is None:
        optimizer_class = torch.optim.SGD

    if clip_value is None:
        clip_value = 0.1 * 1.0 / lr

    f = individual.to_torch()

    if isinstance(f, list):
        params = []
        for torch_mod in f:
            params += list(torch_mod.parameters())
    else:
        params = list(f.parameters())

    if len(params) > 0:
        optimizer = optimizer_class(params, lr=lr)

        for i in range(gradient_steps):
            loss = objective(f)
            if not torch.isfinite(loss):
                continue

            if isinstance(f, list):
                for torch_mod in f:
                    torch_mod.zero_grad()
            else:
                f.zero_grad()

            loss.backward()
            if clip_value is not np.inf:
                torch.nn.utils.clip_grad.clip_grad_value_(params, clip_value)

            optimizer.step()

            assert all(torch.isfinite(t) for t in params)

        individual.update_parameters_from_torch_class(f)
