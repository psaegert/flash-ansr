import cProfile
import pstats

from flash_ansr import SkeletonPool, get_path
from simplipy import SimpliPyEngine
from tqdm import tqdm


MODEL = 'v23.0-3M'
ENGINE = 'dev_7-3'
MAX_PATTERN_LENGTH = 7
N_SAMPLES = 1024


def main():
    pool = SkeletonPool.from_config(get_path('configs', MODEL, 'skeleton_pool_train.yaml'))
    pool.simplify = False
    pool.sample_strategy['max_tries'] = 1000

    engine = SimpliPyEngine.load(ENGINE)

    # Pre-sample skeletons so sampling time is not included in the profile
    print(f'Sampling {N_SAMPLES} skeletons...')
    skeletons = []
    for _ in tqdm(range(N_SAMPLES)):
        skeleton, _, _ = pool.sample_skeleton()
        skeletons.append(skeleton)

    # Profile simplification
    print(f'Profiling SimpliPy (max_pattern_length={MAX_PATTERN_LENGTH}) on {N_SAMPLES} skeletons...')
    profiler = cProfile.Profile()
    profiler.enable()

    for skeleton in tqdm(skeletons):
        engine.simplify(skeleton, max_pattern_length=MAX_PATTERN_LENGTH)

    profiler.disable()

    # Print results
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    print('\n=== Top 40 by cumulative time ===')
    stats.print_stats(40)

    stats.sort_stats('tottime')
    print('\n=== Top 40 by total time ===')
    stats.print_stats(40)

    # Save profile for later analysis (e.g. snakeviz)
    output_path = get_path('results', 'simplification', filename='simplipy_profile.prof')
    profiler.dump_stats(output_path)
    print(f'Profile saved to {output_path}')


if __name__ == '__main__':
    main()
