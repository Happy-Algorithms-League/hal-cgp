import numpy as np
import matplotlib.pyplot as plt
import sys
import torch

sys.path.insert(0, '../')
import gp


def random_regression():
    params = {
        'seed': 81882,
        'n_inputs': 2,
        'n_outputs': 1,
        'n_columns': 3,
        'n_rows': 3,
        'levels_back': 2,
        'n_mutations': 3,
    }

    np.random.seed(params['seed'])

    primitives = gp.CGPPrimitives([gp.CGPAdd, gp.CGPSub, gp.CGPMul, gp.CGPConstantFloat])
    genome = gp.CGPGenome(params['n_inputs'], params['n_outputs'], params['n_columns'], params['n_rows'], primitives)
    genome.randomize(params['levels_back'])
    graph = gp.CGPGraph(genome)

    history_loss = []
    for i in range(3000):

        genome.mutate(params['n_mutations'], params['levels_back'])
        graph.parse_genome(genome)
        f = graph.compile_torch_class()

        if len(list(f.parameters())) > 0:
            optimizer = torch.optim.SGD(f.parameters(), lr=1e-1)
            criterion = torch.nn.MSELoss()

        history_loss_trial = []
        history_loss_bp = []
        for j in range(100):
            x = torch.Tensor(2).normal_()
            y = f(x)
            loss = (2.7182 + x[0] - x[1] - y[0]) ** 2
            history_loss_trial.append(loss.detach().numpy())

            if len(list(f.parameters())) > 0:
                y_target = 2.7182 + x[0] - x[1]

                loss = criterion(y[0], y_target)
                f.zero_grad()
                loss.backward()
                optimizer.step()

                history_loss_bp.append(loss.detach().numpy())

        graph.update_parameter_values(f)

        if np.mean(history_loss_trial[-10:]) < 1e-1:
            print(graph[-1].output_str)
            print(graph)

            if len(list(f.parameters())) > 0:
                plt.plot(history_loss_bp)
                plt.show()

        history_loss.append(np.sum(np.mean(history_loss_trial)))

    plt.plot(history_loss)
    plt.show()
