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
for single_element_batch in dataset.iterate(size=1, n_support=512, avoid_fragmentation=True, verbose=True, tqdm_total=1):
    input_ids, x_tensor, y_tensor, labels, constants, skeleton_hashes = FlashANSRDataset.collate_batch(single_element_batch, device='cuda')

    X = x_tensor
    y = y_tensor[:, 0]

# %%
X.shape, y.shape

# %%
nesymres_output = ansr.fit(X, y)


