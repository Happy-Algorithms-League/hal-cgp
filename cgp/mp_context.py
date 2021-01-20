import multiprocessing as mp

fork_context: mp.context.ForkContext = mp.get_context("fork")
spawn_context: mp.context.SpawnContext = mp.get_context("spawn")
