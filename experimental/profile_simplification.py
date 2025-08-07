# %%
from sympy import simplify, parse_expr
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from flash_ansr import SkeletonPool, get_path
from flash_ansr.expressions.utils import num_to_constants, numbers_to_num
from flash_ansr.eval.utils import bootstrapped_metric_ci
import time
import pickle

# %%
pool = SkeletonPool.from_config(get_path('configs', 'v7.20', 'skeleton_pool_val.yaml'))
pool.sample_strategy['max_operators'] = 15
pool.sample_strategy['max_length'] = 2 * 15 + 1
pool.sample_strategy['max_tries'] = 100
pool.simplify = False

# %%
# expression = ['+', '-', 'sin', 'sin', '-', 'x2', 'x2', 'log', 'x1', 'x3']
# assert pool.expression_space.is_valid(expression)
# pool.expression_space.simplify(expression, max_iter=5, mask_elementary_literals=True)

# %%
# print(pool.expression_space.simplify(('+', 'tan', 'x1', '-', 'log', '<num>', 'inv', 'x2')))

# %%
# print(pool.expression_space.simplify(('abs', '*', '-', 'x3', 'x2', 'sin', '-', '<num>', '<num>')))

# %%
N_SAMPLES = 10_000

# %%
simplified_skeletons = {}
simplification_times = {}

pbar = tqdm(total=N_SAMPLES, smoothing=0)
while len(simplified_skeletons) < N_SAMPLES:
    skeleton, _, _, _ = pool.sample_skeleton()

    if skeleton in simplified_skeletons:
        continue
    # pbar.set_postfix_str(f'{len(skeleton)}: {skeleton}')

    try:
        auto_time1 = time.time()
        auto_simplified = pool.expression_space.simplify(skeleton, max_iter=5)
        auto_time2 = time.time()

        simplified_skeletons[skeleton] = {
            'auto': auto_simplified,
        }
        simplification_times[skeleton] = {
            'auto': auto_time2 - auto_time1,
        }

    except (IndexError, ValueError):
        continue

    pbar.update(1)

pbar.close()