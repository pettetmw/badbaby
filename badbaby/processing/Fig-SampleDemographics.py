# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import seaborn as sns
from os import path as op
import pandas as pd
import janitor
import pandas_flavor as pf
from pandas_profiling import ProfileReport
from badbaby.defaults import static


# %%
df = pd.read_excel(op.join(static, 'meg_covariates.xlsx'), sheet_name='mmn')
df.info()


# %%
covars = (
    pd.read_excel(op.join(static, 'meg_covariates.xlsx'), sheet_name='mmn')
    .clean_names()
    .select_columns(['subjid', 'ses', 'age', 'gender', 'headsize',
                     'maternaledu', 'paternaledu', 'maternalethno', 'paternalethno'])
    .rename_columns({'subjid': 'id'})
    .encode_categorical(columns=['gender', 'maternalethno', 'paternalethno'])
    .bin_numeric(
        from_column_name='maternaledu',
        to_column_name='Education (yrs)',
        num_bins=3,
        labels=['9-12', '13-17', '18-21'])
)


# %%
covars.describe()
profile = ProfileReport(covars, title="MEG covariates profile").to_widgets()

# %%
sns.set_theme(style="ticks", color_codes=True)
# plot distributions of SES amongst genders as a function of mother's years-edu
sns.catplot(x="Education (yrs)", y="ses",
            hue="gender", kind="bar", data=covars)

# %%
sns.catplot(y="maternalethno", hue="gender",
            kind="count", data=covars)  # so much for equity & diversity
# %%
sns.lmplot(y="ses", x="maternaledu", data=covars, x_estimator=np.mean)

# %%
sns.lmplot(y="age", x="headsize", data=covars, x_estimator=np.mean)

# %%
