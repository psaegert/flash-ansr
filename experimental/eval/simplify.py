from sympy import simplify, parse_expr
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from flash_ansr import SkeletonPool, get_path
from flash_ansr.expressions.utils import num_to_constants, numbers_to_num
from flash_ansr.eval.utils import bootstrapped_metric_ci
import time
import pickle
READ = False
pool = SkeletonPool.from_config(get_path('configs', 'v6.0', 'skeleton_pool_val.yaml'))
pool.sample_strategy['max_operators'] = 15
pool.sample_strategy['max_length'] = 2 * 15 + 1
pool.sample_strategy['max_tries'] = 100
pool.simplify = False
N_SAMPLES = 10_000
# from multiprocessing import Process, Queue


# def timeout(seconds, action=None):
#     """Calls any function with timeout after 'seconds'.
#        If a timeout occurs, 'action' will be returned or called if
#        it is a function-like object.
#        https://www.reddit.com/r/Python/comments/8t9bk4/the_absolutely_easiest_way_to_time_out_a_function/
#     """
#     def handler(queue, func, args, kwargs):
#         queue.put(func(*args, **kwargs))

#     def decorator(func):

#         def wraps(*args, **kwargs):
#             q = Queue()
#             p = Process(target=handler, args=(q, func, args, kwargs))
#             p.start()
#             p.join(timeout=seconds)
#             if p.is_alive():
#                 p.terminate()
#                 p.join()
#                 if hasattr(action, '__call__'):
#                     return action()
#                 else:
#                     return action
#             else:
#                 return q.get()

#         return wraps

#     return decorator

# @timeout(1, action=None)
# def timeoutable_simplify(expr, ratio=None):
#     if ratio is not None:
#         return simplify(expr, ratio=ratio)
#     return simplify(expr)
import signal

def timeout_handler(signum, frame):
    raise TimeoutError()

signal.signal(signal.SIGALRM, timeout_handler)

def timeout(seconds, action=None):

    def decorator(func):

        def wraps(*args, **kwargs):
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            except TimeoutError:
                if hasattr(action, '__call__'):
                    result = action()
                else:
                    result = action
            signal.alarm(0)
            return result

        return wraps

    return decorator

@timeout(1, action=None)
def timeoutable_simplify(expr, ratio=None):
    if ratio is not None:
        return simplify(expr, ratio=ratio)
    return simplify(expr)
def sympy_simplify_wrapper(expression: list[str], ratio=None, debug=False):
    expression, constants = num_to_constants(list(expression))
    if debug: print(expression)

    expression = pool.expression_space.prefix_to_infix(expression, power='**')
    if debug: print(expression)

    for c in constants:
        expression = expression.replace(c, str(np.random.uniform(-10, 10)))
    if debug: print(expression)

    expression = parse_expr(expression)
    if debug: print(expression)

    if debug: print('Simplifying...')
    expression = timeoutable_simplify(expression, ratio)
    if expression is None:
        return None
    else:
        expression = str(expression)
    if debug: print(expression)

    translations = {
        'Abs': 'abs',
    }

    for translate_from, translate_to in translations.items():
        expression = expression.replace(translate_from, translate_to)

    if debug: print(f'Parsing {expression}')
    expression = pool.expression_space.parse_expression(expression)
    if debug: print(expression)

    expression = numbers_to_num(expression, inplace=True)
    if debug: print(expression)

    return tuple(expression)
if READ:
    with open(get_path('results', 'simplification', filename='simplification_results.pkl'), 'rb') as f:
        simplified_skeletons = pickle.load(f)

    with open(get_path('results', 'simplification', filename='simplification_times.pkl'), 'rb') as f:
        simplification_times = pickle.load(f)
else:
    simplified_skeletons = {}
    simplification_times = {}

    pbar = tqdm(total=N_SAMPLES, smoothing=0)
    while len(simplified_skeletons) < N_SAMPLES:
        skeleton, _, _ = pool.sample_skeleton()

        if skeleton in simplified_skeletons:
            continue
        pbar.set_postfix_str(f'{len(skeleton)}: {skeleton}')

        try:
            custom_time1 = time.time()
            custom_simplified = pool.expression_space.simplify(skeleton)
            custom_time2 = time.time()

            sympy_time1 = time.time()
            sympy_simplified = sympy_simplify_wrapper(skeleton)
            sympy_time2 = time.time()

            sympy_1_time1 = time.time()
            sympy_1_simplified = sympy_simplify_wrapper(skeleton, ratio=1)
            sympy_1_time2 = time.time()

            simplified_skeletons[skeleton] = {
                'custom': custom_simplified,
                'sympy': sympy_simplified,
                'sympy_1': sympy_1_simplified,
            }
            simplification_times[skeleton] = {
                'custom': custom_time2 - custom_time1,
                'sympy': sympy_time2 - sympy_time1,
                'sympy_1': sympy_1_time2 - sympy_1_time1,
            }

        except (IndexError, ValueError):
            continue

        pbar.update(1)

    pbar.close()

    with open(get_path('results', 'simplification', filename='simplification_results.pkl'), 'wb') as f:
        pickle.dump(simplified_skeletons, f)

    with open(get_path('results', 'simplification', filename='simplification_times.pkl'), 'wb') as f:
        pickle.dump(simplification_times, f)