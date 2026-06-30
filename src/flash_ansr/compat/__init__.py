# ``ParserFactory`` (the benchmark-import utility) lives in ``flash_ansr.convert_data``: it is core
# (simplipy + symbolic_data only), not a third-party model adapter. This back-compat alias keeps the
# legacy ``flash_ansr.compat.ParserFactory`` import path working.
from flash_ansr.convert_data import ParserFactory
