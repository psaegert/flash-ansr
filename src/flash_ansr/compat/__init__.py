# ``ParserFactory`` (the ``import-data`` benchmark-import utility) moved to
# ``flash_ansr.convert_data``: it is core (simplipy + flash_ansr.expressions only), not a
# third-party model adapter. This back-compat alias is kept for the pre-split window and is
# dropped when ``compat/`` is carved into ``srbf``. See REPO_SPLIT_PLAN.md §3.
from flash_ansr.convert_data import ParserFactory
