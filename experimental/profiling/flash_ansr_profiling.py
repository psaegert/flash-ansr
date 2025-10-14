# %%
from flash_ansr.utils import substitute_root_path, load_config, get_path
from flash_ansr.data import FlashANSRDataset
from flash_ansr import FlashANSR

# %%
evaluation_config = load_config(substitute_root_path(get_path('configs', 'nesymres-100M', 'evaluation.yaml')))

ansr = FlashANSR.load(get_path('models', 'ansr-models', 'v7.0'))

# %%
dataset = FlashANSRDataset.from_config(get_path('data', 'ansr-data', 'test_set', 'feynman', 'dataset.yaml'))

# %%
for batch in dataset.iterate(size=1, n_support=512, avoid_fragmentation=True, verbose=True, tqdm_kwargs={'total': 1}):
    batch = dataset.collate(batch, device='cuda')

    X = batch['x_tensors']
    y = batch['y_tensors'][:, 0]

# %%
X.shape, y.shape

# %%
nesymres_output = ansr.fit(X, y)


