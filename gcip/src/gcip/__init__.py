"""Green Customer Insights Platform (Megathon X, PS1).

Package layout:
    gcip.io          -- small file readers (streaming ODS, checksums, parquet helpers)
    gcip.reference   -- builds the processed UK reference tables from public sources
    gcip.simulate    -- the synthetic bank: population, physical world, money, labels
"""

__version__ = "0.1.0"

import pandas as _pd

# Behave the same on pandas 2.x and 3.x: keep plain numpy object strings rather
# than Arrow-backed strings, because the simulator uses numpy string operations.
try:
    _pd.set_option("future.infer_string", False)
except (KeyError, _pd.errors.OptionError):  # option absent on older pandas
    pass
