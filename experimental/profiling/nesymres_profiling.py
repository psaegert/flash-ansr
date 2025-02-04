# %%
from flash_ansr.utils import substitute_root_path, load_config, get_path
from flash_ansr.data import FlashANSRDataset
from flash_ansr import FlashANSR

# %%
evaluation_config = load_config(substitute_root_path(get_path('configs', 'nesymres-100M', 'evaluation.yaml')))

model, fitfunc = load_nesymres(
    eq_setting_path=substitute_root_path(get_path('configs', 'nesymres-100M', 'eq_config.json')),
    config_path=substitute_root_path(get_path('configs', 'nesymres-100M', 'config.yaml')),
    weights_path=substitute_root_path(get_path('models', 'nesymres', '100M.ckpt')),
    beam_size=evaluation_config['beam_width'],
    n_restarts=evaluation_config['n_restarts'],
    device=evaluation_config['device']
)

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
nesymres_output = fitfunc(X, y)


