import numpy as np

try:
    import torch
except ModuleNotFoundError:
    torch = None


def gradient_based(individuals, objective, lr, gradient_steps, optimizer=None, clip_value=None):
    """Perform a local search for numeric leaf values for the list of
    individuals based on gradient information obtained via automatic
    differentiation.

    Parameters
    ----------
    individuals : List
        List of individuals for which to perform local search.
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
    if torch is None:
        raise ModuleNotFoundError("No module named 'torch' (extra requirement)")

    if optimizer is None:
        optimizer_class = torch.optim.SGD

    if clip_value is None:
        clip_value = 0.1 * 1.0 / lr

    for ind in individuals:
        f = ind.to_torch()

        if len(list(f.parameters())) > 0:
            optimizer = optimizer_class(f.parameters(), lr=lr)

            for i in range(gradient_steps):
                loss = objective(f)
                if not torch.isfinite(loss):
                    continue

                f.zero_grad()
                loss.backward()
                if clip_value is not np.inf:
                    torch.nn.utils.clip_grad.clip_grad_value_(f.parameters(), clip_value)
                optimizer.step()

                assert all(torch.isfinite(t) for t in f.parameters())

            ind.update_parameters_from_torch_class(f)
