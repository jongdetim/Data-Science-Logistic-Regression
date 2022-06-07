# %%
import sys
import pandas as pd

# %%
if not len(sys.argv) > 1:
    print("please provide path to a .csv file")
    exit(1)
data = pd.read_csv(sys.argv[1])

# %%



