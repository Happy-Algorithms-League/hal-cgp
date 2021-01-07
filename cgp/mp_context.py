import multiprocessing as mp

mp_context: mp.context.SpawnContext = mp.get_context("spawn")
