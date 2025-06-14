#!/usr/bin/env python
# coding: utf-8

# In[33]:


# Structural model estimated by SMM with II
# Delta and Bootstrapping methods for standard errors
# Auxiliary regressions have state and year fixed effects

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import warnings
import time
import json
import seaborn as sns
import os

BASE_DIR = "/Users/robertsauer/Downloads/Keane/Python/figures"
os.makedirs(BASE_DIR, exist_ok=True)

# ------------------------- Load and prepare data -------------------------------------------------------

df_all = pd.read_stata("/Users/robertsauer/Downloads/Keane/Python/indirect_inference_robert.dta")

USE_R_CPI = False  # 🔁 Toggle to True to use R-CPI-U-RS with 1977=100

CPI_1983_BASE = {
    1977: 60.6, 1978: 65.2, 1979: 72.6, 1980: 82.4, 1981: 90.9,
    1982: 96.5, 1983: 100.0, 1984: 103.9, 1985: 107.6, 1986: 109.6,
    1987: 113.6, 1988: 118.3, 1989: 124.0, 1990: 130.7, 1991: 136.2,
    1992: 140.3, 1993: 144.5, 1994: 148.2, 1995: 152.4, 1996: 156.9,
    1997: 160.5, 1998: 163.0, 1999: 166.6, 2000: 172.2, 2001: 177.1,
    2002: 179.9, 2003: 184.0, 2004: 188.9, 2005: 195.3, 2006: 201.6,
    2007: 207.3, 2008: 215.3, 2009: 214.5, 2010: 218.1, 2011: 224.9,
    2012: 229.6
}

CPI_1977_BASE = {
    1977: 100.0, 1978: 107.8, 1979: 119.5, 1980: 132.4, 1981: 143.4,
    1982: 150.6, 1983: 156.3, 1984: 162.3, 1985: 168.1, 1986: 169.6,
    1987: 176.5, 1988: 183.4, 1989: 191.1, 1990: 202.0, 1991: 206.8,
    1992: 211.9, 1993: 216.7, 1994: 221.4, 1995: 226.4, 1996: 233.4,
    1997: 237.0, 1998: 240.3, 1999: 246.8, 2000: 255.1, 2001: 259.1,
    2002: 265.3, 2003: 270.3, 2004: 279.1, 2005: 288.6, 2006: 296.0,
    2007: 308.1, 2008: 308.3, 2009: 316.7, 2010: 321.5, 2011: 333.1,
    2012: 337.0
}

CPI_BY_YEAR = CPI_1977_BASE if USE_R_CPI else CPI_1983_BASE

# State fips code variable
df_all["stfips"] = df_all["stfips"].astype(int)

# Education variables
df_all["edu"] = df_all["edu"].astype(int)
df_all["edu_2"] = (df_all["edu"] == 2).astype(int)
df_all["edu_3"] = (df_all["edu"] == 3).astype(int)

# Employment variables
df_all["actual_nonemp"] = (df_all["employed_ptft_robert"] == 0).astype(int)
df_all["actual_part"] = (df_all["employed_ptft_robert"] == 1).astype(int)
df_all["actual_full"] = (df_all["employed_ptft_robert"] == 2).astype(int)
df_all["employed"] = (df_all["employed_ptft_robert"] > 0).astype(int)        # PT or FT vs NE
df_all["employed_pt"] = (df_all["employed_ptft_robert"] == 1).astype(int)    # PT vs NE (raw dummy, masking needed)
df_all["employed_ft"] = (df_all["employed_ptft_robert"] == 2).astype(int)    # FT vs NE (raw dummy, masking needed)

# Children variables
df_all["children"] = df_all["children"].astype(int)
df_all["dum_child"] = df_all["dum_child"].astype(int)
df_all["child_0"] = (df_all["children"] == 0).astype(int)
df_all["child_1"] = (df_all["children"] == 1).astype(int)
df_all["child_2"] = (df_all["children"] == 2).astype(int)
df_all["child_3"] = (df_all["children"] >= 3).astype(int)

# Race variables
df_all["race"] = df_all["race"].astype(int)
df_all["race_1"] = (df_all["race"] == 1).astype(int)  # White
df_all["race_2"] = (df_all["race"] == 2).astype(int)  # Black
df_all["race_3"] = (df_all["race"] == 3).astype(int)  # Other

# Age variables
df_all["age"] = df_all["age"].astype(float)
df_all["agesq"] = df_all["age"] ** 2 / 100
df_all["age_group"] = 1
df_all.loc[(df_all["age"] >= 25) & (df_all["age"] < 30), "age_group"] = 2
df_all.loc[(df_all["age"] >= 30) & (df_all["age"] < 35), "age_group"] = 3
df_all.loc[(df_all["age"] >= 35) & (df_all["age"] < 40), "age_group"] = 4
df_all.loc[(df_all["age"] >= 40) & (df_all["age"] < 45), "age_group"] = 5
df_all.loc[(df_all["age"] >= 45) & (df_all["age"] < 51), "age_group"] = 6
df_all["ageg_2"] = (df_all["age_group"] == 2).astype(int)
df_all["ageg_3"] = (df_all["age_group"] == 3).astype(int)
df_all["ageg_4"] = (df_all["age_group"] == 4).astype(int)
df_all["ageg_5"] = (df_all["age_group"] == 5).astype(int)
df_all["ageg_6"] = (df_all["age_group"] == 6).astype(int)

# Year variables
df_all["year"] = df_all["year"].astype(int)
df_all["yeart"] = df_all["year"] - 1977
df_all["yeartsq"] = df_all["yeart"] ** 2 / 100
df_all["yeartcb"] = df_all["yeart"] ** 3 / 1000
df_all["year_group"] = 1 # Carter Pre-Volker 1977-1979
df_all.loc[(df_all["year"] >= 1977) & (df_all["year"] < 1980), "year_group"] = 1
df_all.loc[(df_all["year"] >= 1980) & (df_all["year"] < 1983), "year_group"] = 2 # Volker Recession 1980-1982
df_all.loc[(df_all["year"] >= 1983) & (df_all["year"] < 1990), "year_group"] = 3 # Reagan Expansion 1983-1989
df_all.loc[(df_all["year"] >= 1990) & (df_all["year"] < 1995), "year_group"] = 4 # Early 90s Recession + Jobless Recovery 1990-1994
df_all.loc[(df_all["year"] >= 1995) & (df_all["year"] < 2001), "year_group"] = 5 # Late 90s Boom 1995-2000
df_all.loc[(df_all["year"] >= 2001) & (df_all["year"] < 2004), "year_group"] = 6 # Dot-com Bust + Early 2000s Recession 2001-2003
df_all.loc[(df_all["year"] >= 2004) & (df_all["year"] < 2008), "year_group"] = 7 # Housing Boom + Pre-GFC 2004-2007
df_all.loc[(df_all["year"] >= 2008) & (df_all["year"] < 2013), "year_group"] = 8 # Great Recession and Aftermath 2008-2012
df_all["yearg_1"] = (df_all["year_group"] == 1).astype(int)
df_all["yearg_2"] = (df_all["year_group"] == 2).astype(int)
df_all["yearg_3"] = (df_all["year_group"] == 3).astype(int)
df_all["yearg_4"] = (df_all["year_group"] == 4).astype(int)
df_all["yearg_5"] = (df_all["year_group"] == 5).astype(int)
df_all["yearg_6"] = (df_all["year_group"] == 6).astype(int)
df_all["yearg_7"] = (df_all["year_group"] == 7).astype(int)
df_all["yearg_8"] = (df_all["year_group"] == 8).astype(int)

# State fixed effects (drop one to avoid multicollinearity)
state_dummies = pd.get_dummies(df_all["stfips"], prefix="state", drop_first=True).astype(float)
df_all = pd.concat([df_all, state_dummies], axis=1)
state_cols = sorted(state_dummies.columns)

# Year fixed effects (drop one to avoid multicollinearity)
year_dummies = pd.get_dummies(df_all["year"], prefix="year", drop_first=True)
df_all = pd.concat([df_all, year_dummies], axis=1)
year_cols = year_dummies.columns.tolist()

# Constants for annualizing hourly wages
hours_per_year_pt = 1000
hours_per_year_ft = 2000

# Winsorize wage at 1st and 99th percentiles and save original wage column
wage_p01 = df_all["wage"].quantile(0.01)
wage_p99 = df_all["wage"].quantile(0.99)
df_all["wage_original"] = df_all["wage"]
df_all["wage"] = df_all["wage"].clip(lower=wage_p01, upper=wage_p99)

# Create log wages (after winsorization): valid if wage is positive and not missing
df_all["lnwage"] = np.where(df_all["wage"].notna() & (df_all["wage"] > 0), np.log(df_all["wage"]), np.nan)
df_wageonly = df_all[
    df_all["wage"].notna() &
    (df_all["wage"] > 0) &
    (df_all["employed_ptft_robert"].isin([1, 2]))
].copy()
df_wageonly["lnwage"] = np.log(df_wageonly["wage"])

# Only include part-time or full-time employed individuals for mean wage by year and employment type
df_valid_raw = df_all[
    df_all["employed_ptft_robert"].isin([1, 2]) & 
    df_all["wage"].notna() & 
    (df_all["wage"] > 0)
]
mean_wage_by_year = df_valid_raw.groupby(["year", "employed_ptft_robert"])["wage"].mean().unstack()
mean_wage_by_year.columns = ["Part-Time", "Full-Time"]

# Employment shares (non-employment, part-time, full-time) computed on full sample
employment_shares = df_all.groupby("year")[["actual_nonemp", "actual_part", "actual_full"]].mean()

# ------------------------- Generate actual moments -------------------------------------------------

# Create part-time and full-time log wages
df_all["lnwage_pt_real"] = np.where(df_all["employed_ptft_robert"] == 1, np.log(df_all["wage"]), np.nan)
df_all["lnwage_ft_real"] = np.where(df_all["employed_ptft_robert"] == 2, np.log(df_all["wage"]), np.nan)

# Create part-time and full-time samples
df_pt = df_all[df_all["employed_ptft_robert"] == 1].copy()
df_ft = df_all[df_all["employed_ptft_robert"] == 2].copy()

# Regressors for linear probability models with state and year fixed effects
X_aux = sm.add_constant(
    df_all[[
        "edu_2", "edu_3", "race_2", "race_3", "age", "agesq",
        "child_1", "child_2", "child_3"
    ] + state_cols + year_cols]
)

# Regressors for accepted log wage regressions with state and year fixed effects (PT)
X_pt = sm.add_constant(
    df_pt[[
        "edu_2", "edu_3", "race_2", "race_3", "age", "agesq"
    ] + state_cols + year_cols]
)

# Regressors for accepted log wage regressions with state and year fixed effects (FT)
X_ft = sm.add_constant(
    df_ft[[
        "edu_2", "edu_3", "race_2", "race_3", "age", "agesq"
    ] + state_cols + year_cols]
)

# Ensure X's are fully numeric
X_aux = X_aux.astype(float)
X_pt = X_pt.astype(float)
X_ft = X_ft.astype(float)

# Linear probability auxiliary regression: Employed (PT or FT) vs NE
reg_emp = sm.OLS(df_all["employed"], X_aux).fit(cov_type="HC1")
se_emp = reg_emp.bse

# Linear probability auxiliary regression: PT vs NE — mask to only keep NE and PT
mask_pt_ne = df_all["employed_ptft_robert"].isin([0, 1])
share_employed_pt_ne = df_all.loc[mask_pt_ne, "employed_pt"].mean()
reg_emp_pt = sm.OLS(df_all.loc[mask_pt_ne, "employed_pt"], X_aux.loc[mask_pt_ne]).fit(cov_type="HC1")
se_emp_pt = reg_emp_pt.bse

# Linear probability auxiliary regression: FT vs NE — mask to only keep NE and FT
mask_ft_ne = df_all["employed_ptft_robert"].isin([0, 2])
share_employed_ft_ne = df_all.loc[mask_ft_ne, "employed_ft"].mean()
reg_emp_ft = sm.OLS(df_all.loc[mask_ft_ne, "employed_ft"], X_aux.loc[mask_ft_ne]).fit(cov_type="HC1")
se_emp_ft = reg_emp_ft.bse

# Part-time accepted wage auxiliary regression
reg_pt = sm.OLS(df_pt["lnwage_pt_real"], X_pt).fit(cov_type="HC1")
se_pt = reg_pt.bse
resid_pt = df_pt["lnwage_pt_real"] - reg_pt.predict(X_pt)
r2_pt = reg_pt.rsquared
pred_pt = reg_pt.predict(X_pt)
n_pt = len(pred_pt)
var_pred_pt = np.var(pred_pt, ddof=1)

# Full-time accepted wage auxiliary regression
reg_ft = sm.OLS(df_ft["lnwage_ft_real"], X_ft).fit(cov_type="HC1")
se_ft = reg_ft.bse
resid_ft = df_ft["lnwage_ft_real"] - reg_ft.predict(X_ft)
r2_ft = reg_ft.rsquared
pred_ft = reg_ft.predict(X_ft)
n_ft = len(pred_ft)
var_pred_ft = np.var(pred_ft, ddof=1)

s1 = df_all["actual_nonemp"].mean()
s2 = df_all["actual_part"].mean()

s3 = reg_emp_pt.params["const"]
s4 = reg_emp_pt.params["edu_2"]
s5 = reg_emp_pt.params["edu_3"]
s6 = reg_emp_pt.params["race_2"]
s7 = reg_emp_pt.params["race_3"]
s8 = reg_emp_pt.params["age"]
s9 = reg_emp_pt.params["agesq"]
s10 = reg_emp_pt.params["child_1"]
s11 = reg_emp_pt.params["child_2"]
s12 = reg_emp_pt.params["child_3"]

s13 = reg_emp_ft.params["const"]
s14 = reg_emp_ft.params["edu_2"]
s15 = reg_emp_ft.params["edu_3"]
s16 = reg_emp_ft.params["race_2"]
s17 = reg_emp_ft.params["race_3"]
s18 = reg_emp_ft.params["age"]
s19 = reg_emp_ft.params["agesq"]
s20 = reg_emp_ft.params["child_1"]
s21 = reg_emp_ft.params["child_2"]
s22 = reg_emp_ft.params["child_3"]

s23 = reg_pt.params["const"]
s24 = reg_pt.params["edu_2"]
s25 = reg_pt.params["edu_3"]
s26 = reg_pt.params["race_2"]
s27 = reg_pt.params["race_3"]
s28 = reg_pt.params["age"]
s29 = reg_pt.params["agesq"]
s30 = np.var(resid_pt, ddof=1)
s31 = r2_pt

s32 = reg_ft.params["const"]
s33 = reg_ft.params["edu_2"]
s34 = reg_ft.params["edu_3"]
s35 = reg_ft.params["race_2"]
s36 = reg_ft.params["race_3"]
s37 = reg_ft.params["age"]
s38 = reg_ft.params["agesq"]
s39 = np.var(resid_ft, ddof=1)
s40 = r2_ft

s41 = pred_ft.mean() - pred_pt.mean()

s42 = df_pt.loc[df_pt["edu"] == 1, "lnwage_pt_real"].mean()
s43 = df_pt.loc[df_pt["edu"] == 2, "lnwage_pt_real"].mean()
s44 = df_pt.loc[df_pt["edu"] == 3, "lnwage_pt_real"].mean()
s45 = df_pt.loc[df_pt["race"] == 1, "lnwage_pt_real"].mean()
s46 = df_pt.loc[df_pt["race"] == 2, "lnwage_pt_real"].mean()
s47 = df_pt.loc[df_pt["race"] == 3, "lnwage_pt_real"].mean()
s48 = df_pt.loc[df_pt["age_group"] == 1, "lnwage_pt_real"].mean()
s49 = df_pt.loc[df_pt["age_group"] == 2, "lnwage_pt_real"].mean()
s50 = df_pt.loc[df_pt["age_group"] == 3, "lnwage_pt_real"].mean()
s51 = df_pt.loc[df_pt["age_group"] == 4, "lnwage_pt_real"].mean()
s52 = df_pt.loc[df_pt["age_group"] == 5, "lnwage_pt_real"].mean()
s53 = df_pt.loc[df_pt["age_group"] == 6, "lnwage_pt_real"].mean()
s54 = df_pt.loc[df_pt["year_group"] == 1, "lnwage_pt_real"].mean()
s55 = df_pt.loc[df_pt["year_group"] == 2, "lnwage_pt_real"].mean()
s56 = df_pt.loc[df_pt["year_group"] == 3, "lnwage_pt_real"].mean()
s57 = df_pt.loc[df_pt["year_group"] == 4, "lnwage_pt_real"].mean()
s58 = df_pt.loc[df_pt["year_group"] == 5, "lnwage_pt_real"].mean()
s59 = df_pt.loc[df_pt["year_group"] == 6, "lnwage_pt_real"].mean()
s60 = df_pt.loc[df_pt["year_group"] == 7, "lnwage_pt_real"].mean()
s61 = df_pt.loc[df_pt["year_group"] == 8, "lnwage_pt_real"].mean()

s62 = df_ft.loc[df_ft["edu"] == 1, "lnwage_ft_real"].mean()
s63 = df_ft.loc[df_ft["edu"] == 2, "lnwage_ft_real"].mean()
s64 = df_ft.loc[df_ft["edu"] == 3, "lnwage_ft_real"].mean()
s65 = df_ft.loc[df_ft["race"] == 1, "lnwage_ft_real"].mean()
s66 = df_ft.loc[df_ft["race"] == 2, "lnwage_ft_real"].mean()
s67 = df_ft.loc[df_ft["race"] == 3, "lnwage_ft_real"].mean()
s68 = df_ft.loc[df_ft["age_group"] == 1, "lnwage_ft_real"].mean()
s69 = df_ft.loc[df_ft["age_group"] == 2, "lnwage_ft_real"].mean()
s70 = df_ft.loc[df_ft["age_group"] == 3, "lnwage_ft_real"].mean()
s71 = df_ft.loc[df_ft["age_group"] == 4, "lnwage_ft_real"].mean()
s72 = df_ft.loc[df_ft["age_group"] == 5, "lnwage_ft_real"].mean()
s73 = df_ft.loc[df_ft["age_group"] == 6, "lnwage_ft_real"].mean()
s74 = df_ft.loc[df_ft["year_group"] == 1, "lnwage_ft_real"].mean()
s75 = df_ft.loc[df_ft["year_group"] == 2, "lnwage_ft_real"].mean()
s76 = df_ft.loc[df_ft["year_group"] == 3, "lnwage_ft_real"].mean()
s77 = df_ft.loc[df_ft["year_group"] == 4, "lnwage_ft_real"].mean()
s78 = df_ft.loc[df_ft["year_group"] == 5, "lnwage_ft_real"].mean()
s79 = df_ft.loc[df_ft["year_group"] == 6, "lnwage_ft_real"].mean()
s80 = df_ft.loc[df_ft["year_group"] == 7, "lnwage_ft_real"].mean()
s81 = df_ft.loc[df_ft["year_group"] == 8, "lnwage_ft_real"].mean()

s82 = df_all.loc[mask_pt_ne & (df_all["edu"] == 1), "employed_pt"].mean()
s83 = df_all.loc[mask_pt_ne & (df_all["edu"] == 2), "employed_pt"].mean()
s84 = df_all.loc[mask_pt_ne & (df_all["edu"] == 3), "employed_pt"].mean()
s85 = df_all.loc[mask_pt_ne & (df_all["race"] == 1), "employed_pt"].mean()
s86 = df_all.loc[mask_pt_ne & (df_all["race"] == 2), "employed_pt"].mean()
s87 = df_all.loc[mask_pt_ne & (df_all["race"] == 3), "employed_pt"].mean()
s88 = df_all.loc[mask_pt_ne & (df_all["age_group"] == 1), "employed_pt"].mean()
s89 = df_all.loc[mask_pt_ne & (df_all["age_group"] == 2), "employed_pt"].mean()
s90 = df_all.loc[mask_pt_ne & (df_all["age_group"] == 3), "employed_pt"].mean()
s91 = df_all.loc[mask_pt_ne & (df_all["age_group"] == 4), "employed_pt"].mean()
s92 = df_all.loc[mask_pt_ne & (df_all["age_group"] == 5), "employed_pt"].mean()
s93 = df_all.loc[mask_pt_ne & (df_all["age_group"] == 6), "employed_pt"].mean()
s94 = df_all.loc[mask_pt_ne & (df_all["year_group"] == 1), "employed_pt"].mean()
s95 = df_all.loc[mask_pt_ne & (df_all["year_group"] == 2), "employed_pt"].mean()
s96 = df_all.loc[mask_pt_ne & (df_all["year_group"] == 3), "employed_pt"].mean()
s97 = df_all.loc[mask_pt_ne & (df_all["year_group"] == 4), "employed_pt"].mean()
s98 = df_all.loc[mask_pt_ne & (df_all["year_group"] == 5), "employed_pt"].mean()
s99 = df_all.loc[mask_pt_ne & (df_all["year_group"] == 6), "employed_pt"].mean()
s100 = df_all.loc[mask_pt_ne & (df_all["year_group"] == 7), "employed_pt"].mean()
s101 = df_all.loc[mask_pt_ne & (df_all["year_group"] == 8), "employed_pt"].mean()
s102 = df_all.loc[mask_pt_ne & (df_all["child_1"] == 1), "employed_pt"].mean()
s103 = df_all.loc[mask_pt_ne & (df_all["child_2"] == 1), "employed_pt"].mean()
s104 = df_all.loc[mask_pt_ne & (df_all["child_3"] == 1), "employed_pt"].mean()

s105 = df_all.loc[mask_ft_ne & (df_all["edu"] == 1), "employed_ft"].mean()
s106 = df_all.loc[mask_ft_ne & (df_all["edu"] == 2), "employed_ft"].mean()
s107 = df_all.loc[mask_ft_ne & (df_all["edu"] == 3), "employed_ft"].mean()
s108 = df_all.loc[mask_ft_ne & (df_all["race"] == 1), "employed_ft"].mean()
s109 = df_all.loc[mask_ft_ne & (df_all["race"] == 2), "employed_ft"].mean()
s110 = df_all.loc[mask_ft_ne & (df_all["race"] == 3), "employed_ft"].mean()
s111 = df_all.loc[mask_ft_ne & (df_all["age_group"] == 1), "employed_ft"].mean()
s112 = df_all.loc[mask_ft_ne & (df_all["age_group"] == 2), "employed_ft"].mean()
s113 = df_all.loc[mask_ft_ne & (df_all["age_group"] == 3), "employed_ft"].mean()
s114 = df_all.loc[mask_ft_ne & (df_all["age_group"] == 4), "employed_ft"].mean()
s115 = df_all.loc[mask_ft_ne & (df_all["age_group"] == 5), "employed_ft"].mean()
s116 = df_all.loc[mask_ft_ne & (df_all["age_group"] == 6), "employed_ft"].mean()
s117 = df_all.loc[mask_ft_ne & (df_all["year_group"] == 1), "employed_ft"].mean()
s118 = df_all.loc[mask_ft_ne & (df_all["year_group"] == 2), "employed_ft"].mean()
s119 = df_all.loc[mask_ft_ne & (df_all["year_group"] == 3), "employed_ft"].mean()
s120 = df_all.loc[mask_ft_ne & (df_all["year_group"] == 4), "employed_ft"].mean()
s121 = df_all.loc[mask_ft_ne & (df_all["year_group"] == 5), "employed_ft"].mean()
s122 = df_all.loc[mask_ft_ne & (df_all["year_group"] == 6), "employed_ft"].mean()
s123 = df_all.loc[mask_ft_ne & (df_all["year_group"] == 7), "employed_ft"].mean()
s124 = df_all.loc[mask_ft_ne & (df_all["year_group"] == 8), "employed_ft"].mean()
s125 = df_all.loc[mask_ft_ne & (df_all["child_1"] == 1), "employed_ft"].mean()
s126 = df_all.loc[mask_ft_ne & (df_all["child_2"] == 1), "employed_ft"].mean()
s127 = df_all.loc[mask_ft_ne & (df_all["child_3"] == 1), "employed_ft"].mean()

# Define number of moments globally
num_moments = 127

# Weights using estimated robust variances for regression-based moments
weights = {
    3: 1 / se_emp_pt["const"]**2,
    4: 1 / se_emp_pt["edu_2"]**2,
    5: 1 / se_emp_pt["edu_3"]**2,
    6: 1 / se_emp_pt["race_2"]**2,
    7: 1 / se_emp_pt["race_3"]**2,
    8: 1 / se_emp_pt["age"]**2,
    9: 1 / se_emp_pt["agesq"]**2,
    10: 1 / se_emp_pt["child_1"]**2,
    11: 1 / se_emp_pt["child_2"]**2,
    12: 1 / se_emp_pt["child_3"]**2,

    13: 1 / se_emp_ft["const"]**2,
    14: 1 / se_emp_ft["edu_2"]**2,
    15: 1 / se_emp_ft["edu_3"]**2,
    16: 1 / se_emp_ft["race_2"]**2,
    17: 1 / se_emp_ft["race_3"]**2,
    18: 1 / se_emp_ft["age"]**2,
    19: 1 / se_emp_ft["agesq"]**2,
    20: 1 / se_emp_ft["child_1"]**2,
    21: 1 / se_emp_ft["child_2"]**2,
    22: 1 / se_emp_ft["child_3"]**2,

    23: 1 / se_pt["const"]**2,
    24: 1 / se_pt["edu_2"]**2,
    25: 1 / se_pt["edu_3"]**2,
    26: 1 / se_pt["race_2"]**2,
    27: 1 / se_pt["race_3"]**2,
    28: 1 / se_pt["age"]**2,
    29: 1 / se_pt["agesq"]**2,

    32: 1 / se_ft["const"]**2,
    33: 1 / se_ft["edu_2"]**2,
    34: 1 / se_ft["edu_3"]**2,
    35: 1 / se_ft["race_2"]**2,
    36: 1 / se_ft["race_3"]**2,
    37: 1 / se_ft["age"]**2,
    38: 1 / se_ft["agesq"]**2
}

# Weights using empirical variances for non-regression based moments
weights[1] = 1 / (np.var(df_all["actual_nonemp"], ddof=1) / len(df_all))
weights[2] = 1 / (np.var(df_all["actual_part"], ddof=1) / len(df_all))


weights[30] = 1 / (2 * np.var(resid_pt, ddof=1)**2 / (len(resid_pt) - X_pt.shape[1]))
weights[31] = 1 / (4 * r2_pt * (1 - r2_pt)**2 / len(df_pt))

weights[39] = 1 / (2 * np.var(resid_ft, ddof=1)**2 / (len(resid_ft) - X_ft.shape[1]))
weights[40] = 1 / (4 * r2_ft * (1 - r2_ft)**2 / len(df_ft))

weights[41] = 1 / ((var_pred_ft / n_ft) + (var_pred_pt / n_pt))

weights[42] = 1 / np.var(df_pt.loc[df_pt["edu"] == 1, "lnwage_pt_real"])
weights[43] = 1 / np.var(df_pt.loc[df_pt["edu"] == 2, "lnwage_pt_real"])
weights[44] = 1 / np.var(df_pt.loc[df_pt["edu"] == 3, "lnwage_pt_real"])
weights[45] = 1 / np.var(df_pt.loc[df_pt["race"] == 1, "lnwage_pt_real"])
weights[46] = 1 / np.var(df_pt.loc[df_pt["race"] == 2, "lnwage_pt_real"])
weights[47] = 1 / np.var(df_pt.loc[df_pt["race"] == 3, "lnwage_pt_real"])
weights[48] = 1 / np.var(df_pt.loc[df_pt["age_group"] == 1, "lnwage_pt_real"])
weights[49] = 1 / np.var(df_pt.loc[df_pt["age_group"] == 2, "lnwage_pt_real"])
weights[50] = 1 / np.var(df_pt.loc[df_pt["age_group"] == 3, "lnwage_pt_real"])
weights[51] = 1 / np.var(df_pt.loc[df_pt["age_group"] == 4, "lnwage_pt_real"])
weights[52] = 1 / np.var(df_pt.loc[df_pt["age_group"] == 5, "lnwage_pt_real"])
weights[53] = 1 / np.var(df_pt.loc[df_pt["age_group"] == 6, "lnwage_pt_real"])
weights[54] = 1 / np.var(df_pt.loc[df_pt["year_group"] == 1, "lnwage_pt_real"])
weights[55] = 1 / np.var(df_pt.loc[df_pt["year_group"] == 2, "lnwage_pt_real"])
weights[56] = 1 / np.var(df_pt.loc[df_pt["year_group"] == 3, "lnwage_pt_real"])
weights[57] = 1 / np.var(df_pt.loc[df_pt["year_group"] == 4, "lnwage_pt_real"])
weights[58] = 1 / np.var(df_pt.loc[df_pt["year_group"] == 5, "lnwage_pt_real"])
weights[59] = 1 / np.var(df_pt.loc[df_pt["year_group"] == 6, "lnwage_pt_real"])
weights[60] = 1 / np.var(df_pt.loc[df_pt["year_group"] == 7, "lnwage_pt_real"])
weights[61] = 1 / np.var(df_pt.loc[df_pt["year_group"] == 8, "lnwage_pt_real"])

weights[62] = 1 / np.var(df_ft.loc[df_ft["edu"] == 1, "lnwage_ft_real"])
weights[63] = 1 / np.var(df_ft.loc[df_ft["edu"] == 2, "lnwage_ft_real"])
weights[64] = 1 / np.var(df_ft.loc[df_ft["edu"] == 3, "lnwage_ft_real"])
weights[65] = 1 / np.var(df_ft.loc[df_ft["race"] == 1, "lnwage_ft_real"])
weights[66] = 1 / np.var(df_ft.loc[df_ft["race"] == 2, "lnwage_ft_real"])
weights[67] = 1 / np.var(df_ft.loc[df_ft["race"] == 3, "lnwage_ft_real"])
weights[68] = 1 / np.var(df_ft.loc[df_ft["age_group"] == 1, "lnwage_ft_real"])
weights[69] = 1 / np.var(df_ft.loc[df_ft["age_group"] == 2, "lnwage_ft_real"])
weights[70] = 1 / np.var(df_ft.loc[df_ft["age_group"] == 3, "lnwage_ft_real"])
weights[71] = 1 / np.var(df_ft.loc[df_ft["age_group"] == 4, "lnwage_ft_real"])
weights[72] = 1 / np.var(df_ft.loc[df_ft["age_group"] == 5, "lnwage_ft_real"])
weights[73] = 1 / np.var(df_ft.loc[df_ft["age_group"] == 6, "lnwage_ft_real"])
weights[74] = 1 / np.var(df_ft.loc[df_ft["year_group"] == 1, "lnwage_ft_real"])
weights[75] = 1 / np.var(df_ft.loc[df_ft["year_group"] == 2, "lnwage_ft_real"])
weights[76] = 1 / np.var(df_ft.loc[df_ft["year_group"] == 3, "lnwage_ft_real"])
weights[77] = 1 / np.var(df_ft.loc[df_ft["year_group"] == 4, "lnwage_ft_real"])
weights[78] = 1 / np.var(df_ft.loc[df_ft["year_group"] == 5, "lnwage_ft_real"])
weights[79] = 1 / np.var(df_ft.loc[df_ft["year_group"] == 6, "lnwage_ft_real"])
weights[80] = 1 / np.var(df_ft.loc[df_ft["year_group"] == 7, "lnwage_ft_real"])
weights[81] = 1 / np.var(df_ft.loc[df_ft["year_group"] == 8, "lnwage_ft_real"])

weights[82] = 1 / (share_employed_pt_ne * (1 - share_employed_pt_ne) / df_all[(mask_pt_ne) & (df_all["edu"] == 1)].shape[0])
weights[83] = 1 / (share_employed_pt_ne * (1 - share_employed_pt_ne) / df_all[(mask_pt_ne) & (df_all["edu"] == 2)].shape[0])
weights[84] = 1 / (share_employed_pt_ne * (1 - share_employed_pt_ne) / df_all[(mask_pt_ne) & (df_all["edu"] == 3)].shape[0])
weights[85] = 1 / (share_employed_pt_ne * (1 - share_employed_pt_ne) / df_all[(mask_pt_ne) & (df_all["race"] == 1)].shape[0])
weights[86] = 1 / (share_employed_pt_ne * (1 - share_employed_pt_ne) / df_all[(mask_pt_ne) & (df_all["race"] == 2)].shape[0])
weights[87] = 1 / (share_employed_pt_ne * (1 - share_employed_pt_ne) / df_all[(mask_pt_ne) & (df_all["race"] == 3)].shape[0])
weights[88] = 1 / (share_employed_pt_ne * (1 - share_employed_pt_ne) / df_all[(mask_pt_ne) & (df_all["age_group"] == 1)].shape[0])
weights[89] = 1 / (share_employed_pt_ne * (1 - share_employed_pt_ne) / df_all[(mask_pt_ne) & (df_all["age_group"] == 2)].shape[0])
weights[90] = 1 / (share_employed_pt_ne * (1 - share_employed_pt_ne) / df_all[(mask_pt_ne) & (df_all["age_group"] == 3)].shape[0])
weights[91] = 1 / (share_employed_pt_ne * (1 - share_employed_pt_ne) / df_all[(mask_pt_ne) & (df_all["age_group"] == 4)].shape[0])
weights[92] = 1 / (share_employed_pt_ne * (1 - share_employed_pt_ne) / df_all[(mask_pt_ne) & (df_all["age_group"] == 5)].shape[0])
weights[93] = 1 / (share_employed_pt_ne * (1 - share_employed_pt_ne) / df_all[(mask_pt_ne) & (df_all["age_group"] == 6)].shape[0])
weights[94] = 1 / (share_employed_pt_ne * (1 - share_employed_pt_ne) / df_all[(mask_pt_ne) & (df_all["year_group"] == 1)].shape[0])
weights[95] = 1 / (share_employed_pt_ne * (1 - share_employed_pt_ne) / df_all[(mask_pt_ne) & (df_all["year_group"] == 2)].shape[0])
weights[96] = 1 / (share_employed_pt_ne * (1 - share_employed_pt_ne) / df_all[(mask_pt_ne) & (df_all["year_group"] == 3)].shape[0])
weights[97] = 1 / (share_employed_pt_ne * (1 - share_employed_pt_ne) / df_all[(mask_pt_ne) & (df_all["year_group"] == 4)].shape[0])
weights[98] = 1 / (share_employed_pt_ne * (1 - share_employed_pt_ne) / df_all[(mask_pt_ne) & (df_all["year_group"] == 5)].shape[0])
weights[99] = 1 / (share_employed_pt_ne * (1 - share_employed_pt_ne) / df_all[(mask_pt_ne) & (df_all["year_group"] == 6)].shape[0])
weights[100] = 1 / (share_employed_pt_ne * (1 - share_employed_pt_ne) / df_all[(mask_pt_ne) & (df_all["year_group"] == 7)].shape[0])
weights[101] = 1 / (share_employed_pt_ne * (1 - share_employed_pt_ne) / df_all[(mask_pt_ne) & (df_all["year_group"] == 8)].shape[0])
weights[102] = 1 / (share_employed_pt_ne * (1 - share_employed_pt_ne) / df_all[(mask_pt_ne) & (df_all["child_1"] == 1)].shape[0])
weights[103] = 1 / (share_employed_pt_ne * (1 - share_employed_pt_ne) / df_all[(mask_pt_ne) & (df_all["child_2"] == 1)].shape[0])
weights[104] = 1 / (share_employed_pt_ne * (1 - share_employed_pt_ne) / df_all[(mask_pt_ne) & (df_all["child_3"] == 1)].shape[0])

weights[105] = 1 / (share_employed_ft_ne * (1 - share_employed_ft_ne) / df_all[(mask_ft_ne) & (df_all["edu"] == 1)].shape[0])
weights[106] = 1 / (share_employed_ft_ne * (1 - share_employed_ft_ne) / df_all[(mask_ft_ne) & (df_all["edu"] == 2)].shape[0])
weights[107] = 1 / (share_employed_ft_ne * (1 - share_employed_ft_ne) / df_all[(mask_ft_ne) & (df_all["edu"] == 3)].shape[0])
weights[108] = 1 / (share_employed_ft_ne * (1 - share_employed_ft_ne) / df_all[(mask_ft_ne) & (df_all["race"] == 1)].shape[0])
weights[109] = 1 / (share_employed_ft_ne * (1 - share_employed_ft_ne) / df_all[(mask_ft_ne) & (df_all["race"] == 2)].shape[0])
weights[110] = 1 / (share_employed_ft_ne * (1 - share_employed_ft_ne) / df_all[(mask_ft_ne) & (df_all["race"] == 3)].shape[0])
weights[111] = 1 / (share_employed_ft_ne * (1 - share_employed_ft_ne) / df_all[(mask_ft_ne) & (df_all["age_group"] == 1)].shape[0])
weights[112] = 1 / (share_employed_ft_ne * (1 - share_employed_ft_ne) / df_all[(mask_ft_ne) & (df_all["age_group"] == 2)].shape[0])
weights[113] = 1 / (share_employed_ft_ne * (1 - share_employed_ft_ne) / df_all[(mask_ft_ne) & (df_all["age_group"] == 3)].shape[0])
weights[114] = 1 / (share_employed_ft_ne * (1 - share_employed_ft_ne) / df_all[(mask_ft_ne) & (df_all["age_group"] == 4)].shape[0])
weights[115] = 1 / (share_employed_ft_ne * (1 - share_employed_ft_ne) / df_all[(mask_ft_ne) & (df_all["age_group"] == 5)].shape[0])
weights[116] = 1 / (share_employed_ft_ne * (1 - share_employed_ft_ne) / df_all[(mask_ft_ne) & (df_all["age_group"] == 6)].shape[0])
weights[117] = 1 / (share_employed_ft_ne * (1 - share_employed_ft_ne) / df_all[(mask_ft_ne) & (df_all["year_group"] == 1)].shape[0])
weights[118] = 1 / (share_employed_ft_ne * (1 - share_employed_ft_ne) / df_all[(mask_ft_ne) & (df_all["year_group"] == 2)].shape[0])
weights[119] = 1 / (share_employed_ft_ne * (1 - share_employed_ft_ne) / df_all[(mask_ft_ne) & (df_all["year_group"] == 3)].shape[0])
weights[120] = 1 / (share_employed_ft_ne * (1 - share_employed_ft_ne) / df_all[(mask_ft_ne) & (df_all["year_group"] == 4)].shape[0])
weights[121] = 1 / (share_employed_ft_ne * (1 - share_employed_ft_ne) / df_all[(mask_ft_ne) & (df_all["year_group"] == 5)].shape[0])
weights[122] = 1 / (share_employed_ft_ne * (1 - share_employed_ft_ne) / df_all[(mask_ft_ne) & (df_all["year_group"] == 6)].shape[0])
weights[123] = 1 / (share_employed_ft_ne * (1 - share_employed_ft_ne) / df_all[(mask_ft_ne) & (df_all["year_group"] == 7)].shape[0])
weights[124] = 1 / (share_employed_ft_ne * (1 - share_employed_ft_ne) / df_all[(mask_ft_ne) & (df_all["year_group"] == 8)].shape[0])
weights[125] = 1 / (share_employed_ft_ne * (1 - share_employed_ft_ne) / df_all[(mask_ft_ne) & (df_all["child_1"] == 1)].shape[0])
weights[126] = 1 / (share_employed_ft_ne * (1 - share_employed_ft_ne) / df_all[(mask_ft_ne) & (df_all["child_2"] == 1)].shape[0])
weights[127] = 1 / (share_employed_ft_ne * (1 - share_employed_ft_ne) / df_all[(mask_ft_ne) & (df_all["child_3"] == 1)].shape[0])

# Construct base, mean-normalized and log-normalized inverse variance weighting vectors
W_base = np.array([weights.get(i, 1.0) for i in range(1, num_moments + 1)])              # base weight vector
W_mean_normalized = W_base / np.mean(W_base)                                             # normalize so the mean weight is 1.0
W_log_normalized = np.exp(np.log(W_base) - np.mean(np.log(W_base)))                      # log-normalized weights

W_manual_scaled = np.ones(num_moments)                                                   # Manual scaling for weighting vector
W_manual_scaled[3:9] = 0.0                                                               # all ones except s4-s9 set to 0.0
W_manual_scaled[13:19] = 0.0                                                             # and s14-s19 set to 0.0

moment_definitions = [
    (f"s{i}: {desc}", eval(f"s{i}"), f"m{i}")
    for i, desc in enumerate([
        "Proportion in NE",
        "Proportion in PT",
        "Probability in PT vs NE constant",
        "Probability in PT vs NE edu2",
        "Probability in PT vs NE edu3",
        "Probability in PT vs NE race2",
        "Probability in PT vs NE race3",
        "Probability in PT vs NE age",
        "Probability in PT vs NE agesq",
        "Probability in PT vs NE child1",
        "Probability in PT vs NE child2",
        "Probability in PT vs NE child3",
        "Probability in FT vs NE constant",
        "Probability in FT vs NE edu2",
        "Probability in FT vs NE edu3",
        "Probability in FT vs NE race2",
        "Probability in FT vs NE race3",
        "Probability in FT vs NE age",
        "Probability in FT vs NE agesq",
        "Probability in FT vs NE child1",
        "Probability in FT vs NE child2",
        "Probability in FT vs NE child3",
        "OLS PT lnwage regression constant",
        "OLS PT lnwage regression edu2",
        "OLS PT lnwage regression edu3",
        "OLS PT lnwage regression race2",
        "OLS PT lnwage regression race3",
        "OLS PT lnwage regression age",
        "OLS PT lnwage regression agesq",
        "OLS PT lnwage regression variance",
        "R² of PT lnwage regression",
        "OLS FT lnwage regression constant",
        "OLS FT lnwage regression edu2",
        "OLS FT lnwage regression edu3",
        "OLS FT lnwage regression race2",
        "OLS FT lnwage regression race3",
        "OLS FT lnwage regression age",
        "OLS FT lnwage regression agesq",
        "OLS FT lnwage regression variance",
        "R² of FT lnwage regression",
        "Mean predicted wage gap (FT - PT)",
        "Mean PT lnwage edu=1",
        "Mean PT lnwage edu=2",
        "Mean PT lnwage edu=3",
        "Mean PT lnwage race=1",
        "Mean PT lnwage race=2",
        "Mean PT lnwage race=3",
        "Mean PT lnwage age 20–24",
        "Mean PT lnwage age 25–20",
        "Mean PT lnwage age 30–34",
        "Mean PT lnwage age 35–39",
        "Mean PT lnwage age 40–44",
        "Mean PT lnwage age 45–50",
        "Mean PT lnwage 1977–1979",
        "Mean PT lnwage 1980–1982",
        "Mean PT lnwage 1983–1989",
        "Mean PT lnwage 1990–1994",
        "Mean PT lnwage 1995–2000",
        "Mean PT lnwage 2001–2003",
        "Mean PT lnwage 2004–2007",
        "Mean PT lnwage 2008–2012",
        "Mean FT lnwage edu=1",
        "Mean FT lnwage edu=2",
        "Mean FT lnwage edu=3",
        "Mean FT lnwage race=1",
        "Mean FT lnwage race=2",
        "Mean FT lnwage race=3",
        "Mean FT lnwage age 20–24",
        "Mean FT lnwage age 25–29",
        "Mean FT lnwage age 30–34",
        "Mean FT lnwage age 35–39",
        "Mean FT lnwage age 40–44",
        "Mean FT lnwage age 45–50",
        "Mean FT lnwage 1977–1979",
        "Mean FT lnwage 1980–1982",
        "Mean FT lnwage 1983–1989",
        "Mean FT lnwage 1990–1994",
        "Mean FT lnwage 1995–2000",
        "Mean FT lnwage 2001–2003",
        "Mean FT lnwage 2004–2007",
        "Mean FT lnwage 2008–2012",
        "Share PT|NE+PT, edu1",
        "Share PT|NE+PT, edu2",
        "Share PT|NE+PT, edu3",
        "Share PT|NE+PT, race1",
        "Share PT|NE+PT, race2",
        "Share PT|NE+PT, race3",
        "Share PT|NE+PT, age 20–24",
        "Share PT|NE+PT, age 25–29",
        "Share PT|NE+PT, age 30–34",
        "Share PT|NE+PT, age 35–39",
        "Share PT|NE+PT, age 40–44",
        "Share PT|NE+PT, age 45–50",
        "Share PT|NE+PT, year 1977–1979",
        "Share PT|NE+PT, year 1980–1982",
        "Share PT|NE+PT, year 1983–1989",
        "Share PT|NE+PT, year 1990–1994",
        "Share PT|NE+PT, year 1995–2000",
        "Share PT|NE+PT, year 2001–2003",
        "Share PT|NE+PT, year 2004–2007",
        "Share PT|NE+PT, year 2008–2012",
        "Share PT|NE+PT, child_1",
        "Share PT|NE+PT, child_2",
        "Share PT|NE+PT, child_3",
        "Share FT|NE+FT, edu1",
        "Share FT|NE+FT, edu2",
        "Share FT|NE+FT, edu3",
        "Share FT|NE+FT, race1",
        "Share FT|NE+FT, race2",
        "Share FT|NE+FT, race3",
        "Share FT|NE+FT, age 20–24",
        "Share FT|NE+FT, age 25–29",
        "Share FT|NE+FT, age 30–34",
        "Share FT|NE+FT, age 35–39",
        "Share FT|NE+FT, age 40–44",
        "Share FT|NE+FT, age 45–50",
        "Share FT|NE+FT, year 1977–1979",
        "Share FT|NE+FT, year 1980–1982",
        "Share FT|NE+FT, year 1983–1989",
        "Share FT|NE+FT, year 1990–1994",
        "Share FT|NE+FT, year 1995–2000",
        "Share FT|NE+FT, year 2001–2003",
        "Share FT|NE+FT, year 2004–2007",
        "Share FT|NE+FT, year 2008–2012",
        "Share FT|NE+FT, child_1",
        "Share FT|NE+FT, child_2",
        "Share FT|NE+FT, child_3"
    ], start=1)
]

# ------------------------- Tax Calculations---------------------------------------------------------

# Load federal tax brackets once from full historical file and compute federal tax liability
with open("/Users/robertsauer/Downloads/Keane/Python/federal_tax_brackets_by_year.json", "r") as f:
    FEDERAL_BRACKETS = json.load(f)
# Load state tax rates once from full historical file and compute state tax liability
with open("/Users/robertsauer/Downloads/Keane/Python/state_tax_rates_by_year.json", "r") as f:
    STATE_TAX_RATES = json.load(f)
# Load social security rate, social security cap and Medicare rate
with open("/Users/robertsauer/Downloads/Keane/Python/fica_rates_by_year.json", "r") as f:
    FICA_RATES = json.load(f)
# Load EITC schedules
with open("/Users/robertsauer/Downloads/Keane/Python/eitc_schedules_by_year.json", "r") as f:
    EITC_SCHEDULES = json.load(f)

def calculate_federal_tax_vectorized(wages, year, filing_status):
    brackets = FEDERAL_BRACKETS.get(str(year), {}).get(filing_status, [])
    if not brackets:
        return np.zeros_like(wages)
    tax = np.zeros_like(wages, dtype=float)
    for lower, upper, rate in brackets:
        if upper == "Infinity":
            upper = float("inf")
        taxable = np.clip(wages, lower, upper) - lower
        taxable = np.where(wages > lower, taxable, 0)
        tax += taxable * rate
    return tax

def calculate_state_tax_vectorized(wages, year, statefips_array):
    year_str = str(year)
    tax_rates_for_year = STATE_TAX_RATES.get(year_str, {})
    rates = np.array([tax_rates_for_year.get(str(sf), 0.0) for sf in statefips_array])
    return wages * rates

def calculate_fica_tax_vectorized(wages, year):
    params = FICA_RATES.get(str(year), {})
    ss_rate = params.get("ss_rate", 0.0)
    ss_cap = params.get("ss_cap", float("inf"))
    medicare_rate = params.get("medicare_rate", 0.0)
    wages = np.asarray(wages)
    ss_tax = np.minimum(wages, ss_cap) * ss_rate
    medicare_tax = wages * medicare_rate
    return ss_tax + medicare_tax

def calculate_eitc_vectorized(wages, year, filing_status, num_children):
    # Cap number of children at 3 as per IRS definition
    year_str = str(year)
    filing_status_str = str(filing_status)
    children_keys = np.clip(num_children, 0, 3).astype(str)

    eitc_values = np.zeros_like(wages)

    for i in range(len(wages)):
        schedule = EITC_SCHEDULES.get(year_str, {}).get(filing_status_str, {}).get(children_keys[i])
        if not schedule:
            print(
                f"[DEBUG] Missing schedule for year={year_str}, "
                f"status={filing_status_str}, children={children_keys[i]}"
            )
            continue
        try:
            rate = schedule["rate"]
            max_credit = schedule["max_credit"]
            phase_out_start = schedule["phase_out_start"]
            phase_out_end = schedule["phase_out_end"]
        except KeyError as e:
            print(
                f"[DEBUG] Missing or invalid schedule for year={year_str}, "
                f"status={filing_status_str}, children={children_keys[i]}: {e}"
            )
            continue

        # ✅ Skip invalid cases with zero rate to avoid division by zero
        if rate == 0:
            print(
                f"[DEBUG] Zero EITC rate for year={year_str}, "
                f"status={filing_status_str}, children={children_keys[i]}"
            )
            continue

        income = wages[i]
        if income <= max_credit / rate:
            credit = rate * income
        elif income <= phase_out_end:
            credit = max_credit - rate * (income - phase_out_start)
        else:
            credit = 0.0

        eitc_values[i] = credit

    return eitc_values

# ------------------------- Fixed Random Draws for Wages and Utility Shocks -------------------------

np.random.seed(42)
n = len(df_all)
eta_1 = np.random.normal(0, 1, n)
eta_2 = np.random.normal(0, 1, n)
eta_3 = np.random.normal(0, 1, n)
eta_4 = np.random.normal(0, 1, n)

# ------------------------- Toggles for Simulation/Estimation/Weighting/Standard Errors -------------

SKIP_ESTIMATION = False               # Set to True to simulate only

USE_WEIGHTED_SMM = False              # Set to False for unweighted SMM
USE_LOG_NORMALIZED_WEIGHTS = False    # Set to True to use log normalized weighting in SMM
USE_MANUAL_SCALED_WEIGHTS = False     # Set to True to use manual scaling for selected moments

assert not (USE_LOG_NORMALIZED_WEIGHTS and USE_MANUAL_SCALED_WEIGHTS), \
    "Cannot use both log-normalized and manual-scaled weights at the same time. Choose one."

if not USE_WEIGHTED_SMM:
    W = np.ones(num_moments)         # Equal 1.0 weight for all moments in SMM
elif USE_MANUAL_SCALED_WEIGHTS:
    W = W_manual_scaled              # Manually scaled weights
elif USE_LOG_NORMALIZED_WEIGHTS:
    W = W_log_normalized             # Log normalized weights in SMM
else:
    W = W_mean_normalized            # Mean normalized weights in SMM

RUN_DELTA = True                     # Set to False to disable Delta Method for standard errors
RUN_BOOTSTRAP = False                # Set to True to activate Bootstrap Method for standard errors

# ------------------------- Simulate Model ----------------------------------------------------------

# Parameters to be estimated
param_names = [
    "alpha_pt",
    "gamma_age_pt",
    "gamma_agesq_pt"
]

# Initial values for parameters
theta0 = [
    0.995658,
    0.013966,
    -0.013518
]

# Custom bump settings for Nelder-Mead simplex
custom_bump_percent = 0.25  # % bump for non-zero params 0.10 = 10% (default is 0.05)
bump_if_zero = 0.50         # Absolute bump for zero-valued params (default is 0.00025)

assert len(theta0) == len(param_names), "Mismatch between theta0 and param_names"

# Fixed Parameters
fixed_params = {
    "mu_pt": -0.161175,
    "beta_edu_2_pt": 0.139161,
    "beta_edu_3_pt": 0.384016,
    "beta_race_2_pt": -0.103667,
    "beta_race_3_pt": -0.028591,
    "beta_age_pt": 0.049530,
    "beta_agesq_pt": -0.075483,
    "beta_yeart_pt": 0.036662,
    "beta_yeartsq_pt": -0.014824,
    "beta_yeartcb_pt": 0.000366,
    "beta_yearg_1_pt": -0.173470,
    "sigma_pt": -0.445268,
    "alpha_pt": 0.995658,
    "gamma_edu_2_pt": -0.050364,
    "gamma_edu_3_pt": -0.056926,
    "gamma_race_2_pt": 0.061106,
    "gamma_race_3_pt": 0.082067,
    "gamma_age_pt": 0.013966,
    "gamma_agesq_pt": -0.013518,
    "gamma_yeart_pt": -0.003093,
    "gamma_yeartsq_pt": 0.005415,
    "gamma_yeartcb_pt": -0.000350,
    "gamma_yearg_1_pt": -0.005816,
    "gamma_child_1_pt": 0.012321,
    "gamma_child_2_pt": 0.022688,
    "gamma_child_3_pt": 0.061515,
    "mu_ft": -2.363735,
    "beta_edu_2_ft": 0.522888,
    "beta_edu_3_ft": 0.883247,
    "beta_race_2_ft": -0.096450,
    "beta_race_3_ft": -0.021890,
    "beta_age_ft": 0.147076,
    "beta_agesq_ft": -0.168225,
    "beta_yeart_ft": 0.055035,
    "beta_yeartsq_ft": -0.092040,
    "beta_yeartcb_ft": 0.007020,
    "beta_yearg_1_ft": -0.024561,
    "sigma_ft": -0.495230,
    "alpha_ft": 1.151183,
    "gamma_edu_2_ft": -0.045414,
    "gamma_edu_3_ft": 0.053443,
    "gamma_race_2_ft": 0.020022,
    "gamma_race_3_ft": 0.064552,
    "gamma_age_ft": -0.004648,
    "gamma_agesq_ft": 0.008459,
    "gamma_yeart_ft": -0.006842,
    "gamma_yeartsq_ft": 0.006878,
    "gamma_yeartcb_ft": 0.001803,
    "gamma_yearg_1_ft": 0.031595,
    "gamma_child_1_ft": 0.024772,
    "gamma_child_2_ft": 0.093772,
    "gamma_child_3_ft": 0.187734,
    "a_21": 0.491443,
    "a_22": -2.703244,
    "a_11": 0.000000, # non-identified parameter
    "rho": 0.500000   # non-identified parameter
}

def simulate_moments(theta, df_template):

    # Merge estimated and fixed parameters
    theta_dict = dict(zip(param_names, theta))
    params = {**fixed_params, **theta_dict}

    def get_param(name):
        if name not in params:
            raise ValueError(f"Missing parameter: {name}")
        return params[name]

    # Retrieve wage offer function parameters
    mu_pt = get_param("mu_pt")
    beta_edu_2_pt = get_param("beta_edu_2_pt")
    beta_edu_3_pt = get_param("beta_edu_3_pt")
    beta_race_2_pt = get_param("beta_race_2_pt")
    beta_race_3_pt = get_param("beta_race_3_pt")
    beta_age_pt = get_param("beta_age_pt")
    beta_agesq_pt = get_param("beta_agesq_pt")
    beta_yeart_pt = get_param("beta_yeart_pt")
    beta_yeartsq_pt = get_param("beta_yeartsq_pt")
    beta_yeartcb_pt = get_param("beta_yeartcb_pt")
    beta_yearg_1_pt = get_param("beta_yearg_1_pt")
    sigma_pt = np.exp(get_param("sigma_pt"))
    mu_ft = get_param("mu_ft")
    beta_edu_2_ft = get_param("beta_edu_2_ft")
    beta_edu_3_ft = get_param("beta_edu_3_ft")
    beta_race_2_ft = get_param("beta_race_2_ft")
    beta_race_3_ft = get_param("beta_race_3_ft")
    beta_age_ft = get_param("beta_age_ft")
    beta_agesq_ft = get_param("beta_agesq_ft")
    beta_yeart_ft = get_param("beta_yeart_ft")
    beta_yeartsq_ft = get_param("beta_yeartsq_ft")
    beta_yeartcb_ft = get_param("beta_yeartcb_ft")
    beta_yearg_1_ft = get_param("beta_yearg_1_ft")
    sigma_ft = np.exp(get_param("sigma_ft"))

    # Retrieve disutility of work parameters
    alpha_pt = get_param("alpha_pt")
    gamma_edu_2_pt = get_param("gamma_edu_2_pt")
    gamma_edu_3_pt = get_param("gamma_edu_3_pt")
    gamma_race_2_pt = get_param("gamma_race_2_pt")
    gamma_race_3_pt = get_param("gamma_race_3_pt")
    gamma_age_pt = get_param("gamma_age_pt")
    gamma_agesq_pt = get_param("gamma_agesq_pt")
    gamma_yeart_pt = get_param("gamma_yeart_pt")
    gamma_yeartsq_pt = get_param("gamma_yeartsq_pt")
    gamma_yeartcb_pt = get_param("gamma_yeartcb_pt")
    gamma_yearg_1_pt = get_param("gamma_yearg_1_pt")
    gamma_child_1_pt = get_param("gamma_child_1_pt")
    gamma_child_2_pt = get_param("gamma_child_2_pt")
    gamma_child_3_pt = get_param("gamma_child_3_pt")
    alpha_ft = get_param("alpha_ft")
    gamma_edu_2_ft = get_param("gamma_edu_2_ft")
    gamma_edu_3_ft = get_param("gamma_edu_3_ft")
    gamma_race_2_ft = get_param("gamma_race_2_ft")
    gamma_race_3_ft = get_param("gamma_race_3_ft")
    gamma_age_ft = get_param("gamma_age_ft")
    gamma_agesq_ft = get_param("gamma_agesq_ft")
    gamma_yeart_ft = get_param("gamma_yeart_ft")
    gamma_yeartsq_ft = get_param("gamma_yeartsq_ft")
    gamma_yeartcb_ft = get_param("gamma_yeartcb_ft")
    gamma_yearg_1_ft = get_param("gamma_yearg_1_ft")
    gamma_child_1_ft = get_param("gamma_child_1_ft")
    gamma_child_2_ft = get_param("gamma_child_2_ft")
    gamma_child_3_ft = get_param("gamma_child_3_ft")

    # Retrieve utility shocks and utility parameters
    a_11 = np.exp(get_param("a_11"))
    a_21 = get_param("a_21")
    a_22 = np.exp(get_param("a_22"))
    rho = get_param("rho")

    # Define variables and stochastic terms
    n_obs = len(df_template)
    edu = df_template["edu"].values
    edu_2 = df_template["edu_2"].values
    edu_3 = df_template["edu_3"].values
    race = df_template["race"].values
    race_2 = df_template["race_2"].values
    race_3 = df_template["race_3"].values
    age = df_template["age"].values
    agesq = df_template["agesq"].values
    yeart = df_template["yeart"].values
    yeartsq = df_template["yeartsq"].values
    yeartcb = df_template["yeartcb"].values
    yearg_1 = df_template["yearg_1"].values
    children = df_template["children"].values
    child_1 = df_template["child_1"].values
    child_2 = df_template["child_2"].values
    child_3 = df_template["child_3"].values
    state_cols = sorted([col for col in df_template.columns if col.startswith("state_")])
    eta_pt_wage = eta_1[:n_obs]
    eta_ft_wage = eta_2[:n_obs]
    eta_pt_utility = eta_3[:n_obs]
    eta_ft_utility = eta_4[:n_obs]

    # Variances and Cholesky decompositions
    eps_pt_wage = sigma_pt * eta_pt_wage
    eps_ft_wage = sigma_ft * eta_ft_wage
    eps_pt_utility = a_11 * eta_pt_utility
    eps_ft_utility = a_21 * eta_pt_utility + a_22 * eta_ft_utility

    # Part-time wage offer function
    lnwage_pt = (
        mu_pt
        + (beta_edu_2_pt * edu_2)
        + (beta_edu_3_pt * edu_3)
        + (beta_race_2_pt * race_2)
        + (beta_race_3_pt * race_3)
        + (beta_age_pt * age)
        + (beta_agesq_pt * agesq)
        + (beta_yeart_pt * yeart)
        + (beta_yeartsq_pt * yeartsq)
        + (beta_yeartcb_pt * yeartcb)
        + (beta_yearg_1_pt * yearg_1)
        + eps_pt_wage
    )

    wage_pt = np.exp(lnwage_pt)

    # Full-time wage offer function
    lnwage_ft = (
        mu_ft
        + (beta_edu_2_ft * edu_2)
        + (beta_edu_3_ft * edu_3)
        + (beta_race_2_ft * race_2)
        + (beta_race_3_ft * race_3)
        + (beta_age_ft * age)
        + (beta_agesq_ft * agesq)
        + (beta_yeart_ft * yeart)
        + (beta_yeartsq_ft * yeartsq)
        + (beta_yeartcb_ft * yeartcb)
        + (beta_yearg_1_ft * yearg_1)
        + eps_ft_wage
    )

    wage_ft = np.exp(lnwage_ft)

    # Disutility of work functions
    disutil_pt = (
        alpha_pt
        + (gamma_edu_2_pt * edu_2)
        + (gamma_edu_3_pt * edu_3)
        + (gamma_race_2_pt * race_2)
        + (gamma_race_3_pt * race_3)
        + (gamma_age_pt * age)
        + (gamma_agesq_pt * agesq)
        + (gamma_yeart_pt * yeart)
        + (gamma_yeartsq_pt * yeartsq)
        + (gamma_yeartcb_pt * yeartcb)
        + (gamma_yearg_1_pt * yearg_1)
        + (gamma_child_1_pt * child_1)
        + (gamma_child_2_pt * child_2)
        + (gamma_child_3_pt * child_3)
    )

    disutil_ft = (
        alpha_ft
        + (gamma_edu_2_ft * edu_2)
        + (gamma_edu_3_ft * edu_3)
        + (gamma_race_2_ft * race_2)
        + (gamma_race_3_ft * race_3)
        + (gamma_age_ft * age)
        + (gamma_agesq_ft * agesq)
        + (gamma_yeart_ft * yeart)
        + (gamma_yeartsq_ft * yeartsq)
        + (gamma_yeartcb_ft * yeartcb)
        + (gamma_yearg_1_ft * yearg_1)
        + (gamma_child_1_ft * child_1)
        + (gamma_child_2_ft * child_2)
        + (gamma_child_3_ft * child_3)
    )

    # Initialize df_sim
    df_sim = df_template.copy()

    # Inputs into tax functions
    df_sim["wage_pt"] = wage_pt
    df_sim["wage_ft"] = wage_ft
    year = int(df_template["year"].iloc[0])
    filing_status = "single" 

    # Annualize wages for tax calculation
    annual_wage_pt = wage_pt * hours_per_year_pt
    annual_wage_ft = wage_ft * hours_per_year_ft

    # Calculate federal tax liability (in $/year)
    federal_tax_pt = calculate_federal_tax_vectorized(annual_wage_pt, year, filing_status)
    federal_tax_ft = calculate_federal_tax_vectorized(annual_wage_ft, year, filing_status)

    # Calculate state tax liability (in $/year)
    statefips = df_template["stfips"].values
    state_tax_pt = calculate_state_tax_vectorized(annual_wage_pt, year, statefips)
    state_tax_ft = calculate_state_tax_vectorized(annual_wage_ft, year, statefips)

    # Calculate social security + Medicare tax liability (in $/year)
    fica_tax_pt = calculate_fica_tax_vectorized(annual_wage_pt, year)
    fica_tax_ft = calculate_fica_tax_vectorized(annual_wage_ft, year)

    # Compute EITC (in $/year)
    num_children = df_template["children"].values
    eitc_pt = calculate_eitc_vectorized(annual_wage_pt, year, filing_status, num_children)
    eitc_ft = calculate_eitc_vectorized(annual_wage_ft, year, filing_status, num_children)

    # Calculate total liability (in $/year)
    total_tax_pt = federal_tax_pt + state_tax_pt + fica_tax_pt - eitc_pt
    total_tax_ft = federal_tax_ft + state_tax_ft + fica_tax_ft - eitc_ft

    # ----------------- DEBUG: Check Tax Year Application -----------------
    DEBUG_TAX = False  # 🔁 Set to True to enable tax year debug output

    if DEBUG_TAX:
        print("\n🔍 DEBUG TAX CHECK: Showing 5 sample individuals")
        sample_indices = np.random.choice(len(df_template), size=5, replace=False)
        for i in sample_indices:
            person_year = df_template["year"].iloc[i]
            person_wage = wage_pt[i] * hours_per_year_pt
            person_fed_tax = federal_tax_pt[i]
            person_statefips = df_template["stfips"].iloc[i]
            person_state_tax = state_tax_pt[i]
            person_filing = "single"  # or use from data if available
            print(f"Obs {i}: Year={person_year}, State={person_statefips}, Filing={person_filing}")
            print(
                f"         Wage = ${person_wage:,.2f}, "
                f"Federal Tax = ${person_fed_tax:,.2f}, "
                f"State Tax = ${person_state_tax:,.2f}"
            )
    # --------------------------------------------------------------------------

    # Convert back to hourly by dividing tax by hours/year
    wage_pt_aftertax = wage_pt - (total_tax_pt / hours_per_year_pt)
    wage_ft_aftertax = wage_ft - (total_tax_ft / hours_per_year_ft)
    df_sim["wage_pt_aftertax"] = wage_pt_aftertax
    df_sim["wage_ft_aftertax"] = wage_ft_aftertax

    # Deflate nominal wages (before and after tax) into real 1983 dollars
    df_sim["cpi"] = df_sim["year"].map(CPI_BY_YEAR)
    df_sim["deflator"] = 100.0 / df_sim["cpi"]
    df_sim["real_wage_pt"] = wage_pt * df_sim["deflator"]
    df_sim["real_wage_ft"] = wage_ft * df_sim["deflator"]
    df_sim["real_wage_pt_aftertax"] = wage_pt_aftertax * df_sim["deflator"]
    df_sim["real_wage_ft_aftertax"] = wage_ft_aftertax * df_sim["deflator"]

    # Extract deflated after-tax wages as arrays
    real_wage_pt_aftertax = df_sim["real_wage_pt_aftertax"].values
    real_wage_ft_aftertax = df_sim["real_wage_ft_aftertax"].values

    consump_pt = real_wage_pt_aftertax
    consump_ft = real_wage_ft_aftertax

    # Clip consumption at zero to avoid negative values
    consump_pt = np.maximum(consump_pt, 0)
    consump_ft = np.maximum(consump_ft, 0)

    # Clip disutilities
    disutil_pt_clipped = np.clip(disutil_pt, -10, 10)
    disutil_ft_clipped = np.clip(disutil_ft, -10, 10)

    # Utility functions
    U0 = np.zeros(len(df_template))
    U1 = (consump_pt ** (1 - rho)) / (1 - rho) - np.exp(disutil_pt_clipped) + eps_pt_utility
    U2 = (consump_ft ** (1 - rho)) / (1 - rho) - np.exp(disutil_ft_clipped) + eps_ft_utility

    df_sim["U0"] = U0
    df_sim["U1"] = U1
    df_sim["U2"] = U2

    choices = np.argmax(np.vstack([U0, U1, U2]), axis=0)

    df_sim["sim_nonemp"] = (choices == 0).astype(int)
    df_sim["sim_part"] = (choices == 1).astype(int)
    df_sim["sim_full"] = (choices == 2).astype(int)

# ------------------------- Generate model moments --------------------------------------------------

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

    # Create part-time and full-time log wages
    df_sim["lnwage_pt_real"] = np.where(df_sim["sim_part"] == 1, np.log(df_sim["real_wage_pt"]), np.nan)
    df_sim["lnwage_ft_real"] = np.where(df_sim["sim_full"] == 1, np.log(df_sim["real_wage_ft"]), np.nan)

    # Create simulated employment status variable for masking
    df_sim["sim_employed_ptft"] = np.nan
    df_sim.loc[df_sim["sim_nonemp"] == 1, "sim_employed_ptft"] = 0
    df_sim.loc[df_sim["sim_part"] == 1, "sim_employed_ptft"] = 1
    df_sim.loc[df_sim["sim_full"] == 1, "sim_employed_ptft"] = 2

    df_pt_sim = df_sim[df_sim["sim_part"] == 1].copy()
    df_ft_sim = df_sim[df_sim["sim_full"] == 1].copy()

    # Regressors for simulated linear probability models with state and year fixed effects
    X_sim = sm.add_constant(
        df_sim[[
            "edu_2", "edu_3", "race_2", "race_3", "age", "agesq",
            "child_1", "child_2", "child_3"
        ] + state_cols + year_cols]
    )

    # Regressors for simulated log wage regressions with state and year fixed effects (PT)
    X_pt_sim = sm.add_constant(
        df_pt_sim[[
            "edu_2", "edu_3", "race_2", "race_3",
            "age", "agesq"
        ] + state_cols + year_cols]
    )

    # Regressors for simulated log wage regressions with state and year fixed effects (FT)
    X_ft_sim = sm.add_constant(
        df_ft_sim[[
            "edu_2", "edu_3", "race_2", "race_3",
            "age", "agesq"
        ] + state_cols + year_cols]
    )

    # Ensure X's are fully numeric
    X_sim = X_sim.astype(float)
    X_pt_sim = X_pt_sim.astype(float) 
    X_ft_sim = X_ft_sim.astype(float)

    # Linear probability auxiliary regression: Simulated Employed (PT or FT) vs NE
    df_sim["sim_employed"] = ((df_sim["sim_part"] == 1) | (df_sim["sim_full"] == 1)).astype(int)
    reg_emp_sim = sm.OLS(df_sim["sim_employed"], X_sim).fit(cov_type="HC1")
    se_emp_sim = reg_emp_sim.bse

    # Linear probability auxiliary regression: Simulated PT vs NE — mask to only keep NE and PT
    mask_pt_ne_sim = df_sim["sim_employed_ptft"].isin([0, 1])
    reg_emp_pt_sim = sm.OLS(df_sim.loc[mask_pt_ne_sim, "sim_part"], X_sim.loc[mask_pt_ne_sim]).fit(cov_type="HC1")
    se_emp_pt_sim = reg_emp_pt_sim.bse

    # Linear probability auxiliary regression: Simulated FT vs NE — mask to only keep NE and FT
    mask_ft_ne_sim = df_sim["sim_employed_ptft"].isin([0, 2])
    reg_emp_ft_sim = sm.OLS(df_sim.loc[mask_ft_ne_sim, "sim_full"], X_sim.loc[mask_ft_ne_sim]).fit(cov_type="HC1")
    se_emp_ft_sim = reg_emp_ft_sim.bse

    # Part-time accepted auxiliary wage regression
    reg_pt_sim = sm.OLS(df_pt_sim["lnwage_pt_real"], X_pt_sim).fit(cov_type="HC1")
    se_pt_sim = reg_pt_sim.bse
    resid_pt_sim = df_pt_sim["lnwage_pt_real"] - reg_pt_sim.predict(X_pt_sim)
    r2_pt_sim = reg_pt_sim.rsquared
    pred_pt_sim = reg_pt_sim.predict(X_pt_sim)

    # Full-time accepted wage auxiliary regression
    reg_ft_sim = sm.OLS(df_ft_sim["lnwage_ft_real"], X_ft_sim).fit(cov_type="HC1")
    se_ft_sim = reg_pt_sim.bse
    resid_ft_sim = df_ft_sim["lnwage_ft_real"] - reg_ft_sim.predict(X_ft_sim)
    r2_ft_sim = reg_ft_sim.rsquared
    pred_ft_sim = reg_ft_sim.predict(X_ft_sim)

    m1 = df_sim["sim_nonemp"].mean()
    m2 = df_sim["sim_part"].mean()

    m3 = reg_emp_pt_sim.params["const"]
    m4 = reg_emp_pt_sim.params["edu_2"]
    m5 = reg_emp_pt_sim.params["edu_3"]
    m6 = reg_emp_pt_sim.params["race_2"]
    m7 = reg_emp_pt_sim.params["race_3"]
    m8 = reg_emp_pt_sim.params["age"]
    m9 = reg_emp_pt_sim.params["agesq"]
    m10 = reg_emp_pt_sim.params["child_1"]
    m11 = reg_emp_pt_sim.params["child_2"]
    m12 = reg_emp_pt_sim.params["child_3"]

    m13 = reg_emp_ft_sim.params["const"]
    m14 = reg_emp_ft_sim.params["edu_2"]
    m15 = reg_emp_ft_sim.params["edu_3"]
    m16 = reg_emp_ft_sim.params["race_2"]
    m17 = reg_emp_ft_sim.params["race_3"]
    m18 = reg_emp_ft_sim.params["age"]
    m19 = reg_emp_ft_sim.params["agesq"]
    m20 = reg_emp_ft_sim.params["child_1"]
    m21 = reg_emp_ft_sim.params["child_2"]
    m22 = reg_emp_ft_sim.params["child_3"]

    m23 = reg_pt_sim.params["const"]
    m24 = reg_pt_sim.params["edu_2"]
    m25 = reg_pt_sim.params["edu_3"]
    m26 = reg_pt_sim.params["race_2"]
    m27 = reg_pt_sim.params["race_3"]
    m28 = reg_pt_sim.params["age"]
    m29 = reg_pt_sim.params["agesq"]
    m30 = np.var(resid_pt_sim, ddof=1)
    m31 = r2_pt_sim

    m32 = reg_ft_sim.params["const"]
    m33 = reg_ft_sim.params["edu_2"]
    m34 = reg_ft_sim.params["edu_3"]
    m35 = reg_ft_sim.params["race_2"]
    m36 = reg_ft_sim.params["race_3"]
    m37 = reg_ft_sim.params["age"]
    m38 = reg_ft_sim.params["agesq"]
    m39 = np.var(resid_ft_sim, ddof=1)
    m40 = r2_ft_sim

    m41 = pred_ft_sim.mean() - pred_pt_sim.mean()

    m42 = df_pt_sim.loc[df_pt_sim["edu"] == 1, "lnwage_pt_real"].mean()
    m43 = df_pt_sim.loc[df_pt_sim["edu"] == 2, "lnwage_pt_real"].mean()
    m44 = df_pt_sim.loc[df_pt_sim["edu"] == 3, "lnwage_pt_real"].mean()
    m45 = df_pt_sim.loc[df_pt_sim["race"] == 1, "lnwage_pt_real"].mean()
    m46 = df_pt_sim.loc[df_pt_sim["race"] == 2, "lnwage_pt_real"].mean()
    m47 = df_pt_sim.loc[df_pt_sim["race"] == 3, "lnwage_pt_real"].mean()
    m48 = df_pt_sim.loc[df_pt_sim["age_group"] == 1, "lnwage_pt_real"].mean()
    m49 = df_pt_sim.loc[df_pt_sim["age_group"] == 2, "lnwage_pt_real"].mean()
    m50 = df_pt_sim.loc[df_pt_sim["age_group"] == 3, "lnwage_pt_real"].mean()
    m51 = df_pt_sim.loc[df_pt_sim["age_group"] == 4, "lnwage_pt_real"].mean()
    m52 = df_pt_sim.loc[df_pt_sim["age_group"] == 5, "lnwage_pt_real"].mean()
    m53 = df_pt_sim.loc[df_pt_sim["age_group"] == 6, "lnwage_pt_real"].mean()
    m54 = df_pt_sim.loc[df_pt_sim["year_group"] == 1, "lnwage_pt_real"].mean()
    m55 = df_pt_sim.loc[df_pt_sim["year_group"] == 2, "lnwage_pt_real"].mean()
    m56 = df_pt_sim.loc[df_pt_sim["year_group"] == 3, "lnwage_pt_real"].mean()
    m57 = df_pt_sim.loc[df_pt_sim["year_group"] == 4, "lnwage_pt_real"].mean()
    m58 = df_pt_sim.loc[df_pt_sim["year_group"] == 5, "lnwage_pt_real"].mean()
    m59 = df_pt_sim.loc[df_pt_sim["year_group"] == 6, "lnwage_pt_real"].mean()
    m60 = df_pt_sim.loc[df_pt_sim["year_group"] == 7, "lnwage_pt_real"].mean()
    m61 = df_pt_sim.loc[df_pt_sim["year_group"] == 8, "lnwage_pt_real"].mean()

    m62 = df_ft_sim.loc[df_ft_sim["edu"] == 1, "lnwage_ft_real"].mean()
    m63 = df_ft_sim.loc[df_ft_sim["edu"] == 2, "lnwage_ft_real"].mean()
    m64 = df_ft_sim.loc[df_ft_sim["edu"] == 3, "lnwage_ft_real"].mean()
    m65 = df_ft_sim.loc[df_ft_sim["race"] == 1, "lnwage_ft_real"].mean()
    m66 = df_ft_sim.loc[df_ft_sim["race"] == 2, "lnwage_ft_real"].mean()
    m67 = df_ft_sim.loc[df_ft_sim["race"] == 3, "lnwage_ft_real"].mean()
    m68 = df_ft_sim.loc[df_ft_sim["age_group"] == 1, "lnwage_ft_real"].mean()
    m69 = df_ft_sim.loc[df_ft_sim["age_group"] == 2, "lnwage_ft_real"].mean()
    m70 = df_ft_sim.loc[df_ft_sim["age_group"] == 3, "lnwage_ft_real"].mean()
    m71 = df_ft_sim.loc[df_ft_sim["age_group"] == 4, "lnwage_ft_real"].mean()
    m72 = df_ft_sim.loc[df_ft_sim["age_group"] == 5, "lnwage_ft_real"].mean()
    m73 = df_ft_sim.loc[df_ft_sim["age_group"] == 6, "lnwage_ft_real"].mean()
    m74 = df_ft_sim.loc[df_ft_sim["year_group"] == 1, "lnwage_ft_real"].mean()
    m75 = df_ft_sim.loc[df_ft_sim["year_group"] == 2, "lnwage_ft_real"].mean()
    m76 = df_ft_sim.loc[df_ft_sim["year_group"] == 3, "lnwage_ft_real"].mean()
    m77 = df_ft_sim.loc[df_ft_sim["year_group"] == 4, "lnwage_ft_real"].mean()
    m78 = df_ft_sim.loc[df_ft_sim["year_group"] == 5, "lnwage_ft_real"].mean()
    m79 = df_ft_sim.loc[df_ft_sim["year_group"] == 6, "lnwage_ft_real"].mean()
    m80 = df_ft_sim.loc[df_ft_sim["year_group"] == 7, "lnwage_ft_real"].mean()
    m81 = df_ft_sim.loc[df_ft_sim["year_group"] == 8, "lnwage_ft_real"].mean()

    m82 = df_sim.loc[mask_pt_ne_sim & (df_sim["edu"] == 1), "sim_part"].mean()
    m83 = df_sim.loc[mask_pt_ne_sim & (df_sim["edu"] == 2), "sim_part"].mean()
    m84 = df_sim.loc[mask_pt_ne_sim & (df_sim["edu"] == 3), "sim_part"].mean()
    m85 = df_sim.loc[mask_pt_ne_sim & (df_sim["race"] == 1), "sim_part"].mean()
    m86 = df_sim.loc[mask_pt_ne_sim & (df_sim["race"] == 2), "sim_part"].mean()
    m87 = df_sim.loc[mask_pt_ne_sim & (df_sim["race"] == 3), "sim_part"].mean()
    m88 = df_sim.loc[mask_pt_ne_sim & (df_sim["age_group"] == 1), "sim_part"].mean()
    m89 = df_sim.loc[mask_pt_ne_sim & (df_sim["age_group"] == 2), "sim_part"].mean()
    m90 = df_sim.loc[mask_pt_ne_sim & (df_sim["age_group"] == 3), "sim_part"].mean()
    m91 = df_sim.loc[mask_pt_ne_sim & (df_sim["age_group"] == 4), "sim_part"].mean()
    m92 = df_sim.loc[mask_pt_ne_sim & (df_sim["age_group"] == 5), "sim_part"].mean()
    m93 = df_sim.loc[mask_pt_ne_sim & (df_sim["age_group"] == 6), "sim_part"].mean()
    m94 = df_sim.loc[mask_pt_ne_sim & (df_sim["year_group"] == 1), "sim_part"].mean()
    m95 = df_sim.loc[mask_pt_ne_sim & (df_sim["year_group"] == 2), "sim_part"].mean()
    m96 = df_sim.loc[mask_pt_ne_sim & (df_sim["year_group"] == 3), "sim_part"].mean()
    m97 = df_sim.loc[mask_pt_ne_sim & (df_sim["year_group"] == 4), "sim_part"].mean()
    m98 = df_sim.loc[mask_pt_ne_sim & (df_sim["year_group"] == 5), "sim_part"].mean()
    m99 = df_sim.loc[mask_pt_ne_sim & (df_sim["year_group"] == 6), "sim_part"].mean()
    m100 = df_sim.loc[mask_pt_ne_sim & (df_sim["year_group"] == 7), "sim_part"].mean()
    m101 = df_sim.loc[mask_pt_ne_sim & (df_sim["year_group"] == 8), "sim_part"].mean()
    m102 = df_sim.loc[mask_pt_ne_sim & (df_sim["child_1"] == 1), "sim_part"].mean()
    m103 = df_sim.loc[mask_pt_ne_sim & (df_sim["child_2"] == 1), "sim_part"].mean()
    m104 = df_sim.loc[mask_pt_ne_sim & (df_sim["child_3"] == 1), "sim_part"].mean()

    m105 = df_sim.loc[mask_ft_ne_sim & (df_sim["edu"] == 1), "sim_full"].mean()
    m106 = df_sim.loc[mask_ft_ne_sim & (df_sim["edu"] == 2), "sim_full"].mean()
    m107 = df_sim.loc[mask_ft_ne_sim & (df_sim["edu"] == 3), "sim_full"].mean()
    m108 = df_sim.loc[mask_ft_ne_sim & (df_sim["race"] == 1), "sim_full"].mean()
    m109 = df_sim.loc[mask_ft_ne_sim & (df_sim["race"] == 2), "sim_full"].mean()
    m110 = df_sim.loc[mask_ft_ne_sim & (df_sim["race"] == 3), "sim_full"].mean()
    m111 = df_sim.loc[mask_ft_ne_sim & (df_sim["age_group"] == 1), "sim_full"].mean()
    m112 = df_sim.loc[mask_ft_ne_sim & (df_sim["age_group"] == 2), "sim_full"].mean()
    m113 = df_sim.loc[mask_ft_ne_sim & (df_sim["age_group"] == 3), "sim_full"].mean()
    m114 = df_sim.loc[mask_ft_ne_sim & (df_sim["age_group"] == 4), "sim_full"].mean()
    m115 = df_sim.loc[mask_ft_ne_sim & (df_sim["age_group"] == 5), "sim_full"].mean()
    m116 = df_sim.loc[mask_ft_ne_sim & (df_sim["age_group"] == 6), "sim_full"].mean()
    m117 = df_sim.loc[mask_ft_ne_sim & (df_sim["year_group"] == 1), "sim_full"].mean()
    m118 = df_sim.loc[mask_ft_ne_sim & (df_sim["year_group"] == 2), "sim_full"].mean()
    m119 = df_sim.loc[mask_ft_ne_sim & (df_sim["year_group"] == 3), "sim_full"].mean()
    m120 = df_sim.loc[mask_ft_ne_sim & (df_sim["year_group"] == 4), "sim_full"].mean()
    m121 = df_sim.loc[mask_ft_ne_sim & (df_sim["year_group"] == 5), "sim_full"].mean()
    m122 = df_sim.loc[mask_ft_ne_sim & (df_sim["year_group"] == 6), "sim_full"].mean()
    m123 = df_sim.loc[mask_ft_ne_sim & (df_sim["year_group"] == 7), "sim_full"].mean()
    m124 = df_sim.loc[mask_ft_ne_sim & (df_sim["year_group"] == 8), "sim_full"].mean()
    m125 = df_sim.loc[mask_ft_ne_sim & (df_sim["child_1"] == 1), "sim_full"].mean()
    m126 = df_sim.loc[mask_ft_ne_sim & (df_sim["child_2"] == 1), "sim_full"].mean()
    m127 = df_sim.loc[mask_ft_ne_sim & (df_sim["child_3"] == 1), "sim_full"].mean()

    simulated_values = {f"m{i}": eval(f"m{i}") for i in range(1, num_moments + 1)}

    return simulated_values, df_sim

# ------------------------- Estimate by SMM with Indirect Inference ---------------------------------

# SMM objective function
def smm_loss(theta, df_sample):
    sim_values, _ = simulate_moments(theta, df_sample)

    sim_moments = np.array([sim_values[mname] for _, _, mname in moment_definitions])
    actual_moments = np.array([aval for _, aval, _ in moment_definitions])

    diff = sim_moments - actual_moments

    if USE_WEIGHTED_SMM:
        weighted_squared_diffs = W * diff**2
        return np.sum(weighted_squared_diffs)
    else:
        return np.sum(diff**2)

# Verbose wrapper for loss function to show iteration progress
class VerboseLoss:
    def __init__(self, base_loss_func, df_sample, param_names=None):
        self.loss_func = base_loss_func
        self.df_sample = df_sample
        self.iteration = 0
        self.param_names = param_names

    def __call__(self, theta):
        start_time = time.time()
        loss_val = self.loss_func(theta, self.df_sample)
        end_time = time.time()
        elapsed = end_time - start_time
        self.iteration += 1

        if self.param_names:
            params_str = ", ".join(f"{name}={val:.6f}" for name, val in zip(self.param_names, theta))
        else:
            params_str = ", ".join(f"{val:.6f}" for val in theta)

        print(f"[{self.iteration:03d}] Time={elapsed:.2f}s | {params_str} | Loss={loss_val:.6f}")
        return loss_val

def build_initial_simplex(theta0, bump_percent=custom_bump_percent, bump_if_zero=bump_if_zero):
    n_params = len(theta0)
    simplex = np.tile(theta0, (n_params + 1, 1))
    for i in range(n_params):
        bump = bump_if_zero if theta0[i] == 0 else bump_percent * abs(theta0[i])
        simplex[i + 1, i] += bump
    return simplex

# Run estimation or skip
if SKIP_ESTIMATION:
    print("⏭️ Skipping estimation — using initial parameters.")
    base_params = theta0
    base_result = type("Result", (object,), {"x": theta0})()
else:
    verbose_loss = VerboseLoss(smm_loss, df_all, param_names)
    start_time = time.time()

    # Build and use custom initial simplex
    init_simplex = build_initial_simplex(np.array(theta0))
    base_result = minimize(
        verbose_loss,
        x0=theta0,
        method="Nelder-Mead",
        options={"disp": True, "initial_simplex": init_simplex}
    )

    end_time = time.time()
    elapsed_seconds = end_time - start_time
    elapsed_minutes = elapsed_seconds / 60
    print(f"\n⏱️ Estimation completed in {elapsed_seconds:.2f} seconds ({elapsed_minutes:.2f} minutes).")
    print()
    base_params = base_result.x

theta_hat = base_params
for name, val in zip(param_names, base_params):
    print(f"{name:16s} = {val:.6f}")

# Simulate moments and choices at theta_hat
simulated_moments, df_sim = simulate_moments(theta_hat, df_all)

# --------------------------- CALCULATE STANDARD ERRORS ---------------------------------------------

if not SKIP_ESTIMATION:

    # DELTA METHOD

    if RUN_DELTA:
        start_time_se = time.time()

        # Always ensure actual moments are correctly float
        actual_moments_array = np.array([actual for (_, actual, _) in moment_definitions], dtype=float)

        def numerical_jacobian(theta_hat, df_sample, h_vec=None):
            k = len(theta_hat)
            m = len(moment_definitions)
            J = np.zeros((m, k))

            # Default to uniform bump if not provided
            if h_vec is None:
                h_vec = np.full(k, 1e-5)

            for i in range(k):
                bump = h_vec[i]
                theta_up = np.array(theta_hat, dtype=float)
                theta_down = np.array(theta_hat, dtype=float)
                theta_up[i] += bump
                theta_down[i] -= bump

                # Simulate moments
                m_up_dict = simulate_moments(theta_up, df_sample)[0]
                m_down_dict = simulate_moments(theta_down, df_sample)[0]

                # Convert dicts to arrays
                m_up = np.array([m_up_dict[moment_name] for (_, _, moment_name) in moment_definitions])
                m_down = np.array([m_down_dict[moment_name] for (_, _, moment_name) in moment_definitions])

                # Numerical derivative
                J[:, i] = (m_up - m_down) / (2 * bump)

            return J

        def estimate_standard_errors(theta_hat, df_sample):
            # Customize bump sizes - 1e-3 was lowered from 1e-4
            h_vec = np.array([
                1e-3 if name.startswith(("alpha", "a_", "gamma"))
                else 1e-5
                for name in param_names
            ])
            J = numerical_jacobian(theta_hat, df_sample, h_vec=h_vec)

            sim_dict = simulate_moments(theta_hat, df_sample)[0]
            sim_array = np.array([sim_dict[moment_name] for (_, _, moment_name) in moment_definitions])
            moment_diff = sim_array - actual_moments_array

            if USE_WEIGHTED_SMM:
                W_matrix = np.diag(W)
                S = np.diag(moment_diff ** 2)
                G = J
                V = np.linalg.inv(G.T @ W_matrix @ G) @ (G.T @ W_matrix @ S @ W_matrix @ G) @ np.linalg.inv(G.T @ W_matrix @ G)
            else:
                S = np.diag(moment_diff ** 2)
                V = np.linalg.inv(J.T @ J) @ J.T @ S @ J @ np.linalg.inv(J.T @ J)

            std_errors = np.sqrt(np.diag(V))
            return std_errors

        try:
            standard_errors = estimate_standard_errors(base_result.x, df_all)
            end_time_se = time.time()
            duration_sec = end_time_se - start_time_se
            duration_min = duration_sec / 60
            print(f"\n⏱️ Delta Method standard error computation completed in {duration_sec:.2f} seconds ({duration_min:.2f} minutes).")
            print("\n🧮 Delta Method Standard Errors:")
            for name, se in zip(param_names, standard_errors):
                print(f"{name:16} = {se:.6f}")
        except np.linalg.LinAlgError as e:
            print("\n❌ Delta Method failed due to a linear algebra error:")
            print(e)
            standard_errors = None  # So later checks won't break
        except Exception as e:
            print("\n❌ Delta Method failed due to an unexpected error:")
            print(e)
            standard_errors = None

    #  BOOTSTRAP METHOD

    if RUN_BOOTSTRAP:
        start_time_bs = time.time()
        B = 10  # number of bootstrap replications

        def bootstrap_standard_errors(df_sample, theta_start, n_bootstrap):
            k = len(theta_start)
            boot_estimates = np.zeros((n_bootstrap, k))
            for b in range(n_bootstrap):
                sample_indices = np.random.choice(len(df_sample), len(df_sample), replace=True)
                df_boot = df_sample.iloc[sample_indices].reset_index(drop=True)
                result = minimize(smm_loss, x0=theta_start, args=(df_boot,), method="Nelder-Mead", options={"disp": False})
                boot_estimates[b, :] = result.x
                print(f"[Bootstrap {b+1}/{n_bootstrap}] Completed.")
                print()
            return np.std(boot_estimates, axis=0)

        bootstrap_errors = bootstrap_standard_errors(df_all, theta_hat, n_bootstrap=B)

        end_time_bs = time.time()
        duration_sec = end_time_bs - start_time_bs
        duration_min = duration_sec / 60
        print(f"\n⏱️ Bootstrapping standard error computation completed in {duration_sec:.2f} seconds ({duration_min:.2f} minutes).")

        threshold = 1e-8  # Tolerance for "effectively zero"
        zero_se_mask = (bootstrap_errors < threshold)

        if np.any(zero_se_mask):
            print("\n⚠️ WARNING: Some standard errors are zero or very close to zero!")
            for name, se in zip(param_names, standard_errors):
                if se < threshold:
                    print(f" - {name} has SE = {se:.2e}")
        else:
            print("\n✅ All standard errors are above the threshold.")

        print("\n🧮 Bootstrapping Standard Errors:")
        for name, se in zip(param_names, bootstrap_errors):
            print(f"{name:16} = {se:.6f}")

    # ------------------ p-VALUE DISPLAY BLOCK -----------------------------------------------------

    from scipy.stats import norm

    # Choose standard errors and compute p-values
    if 'standard_errors' in locals() and standard_errors is not None:
        se_used = standard_errors
    elif 'bootstrap_errors' in locals() and bootstrap_errors is not None:
        se_used = bootstrap_errors
    else:
        se_used = [np.nan] * len(param_names)

    if isinstance(se_used, (list, np.ndarray)):
        p_values = 2 * (1 - norm.cdf(np.abs(base_params / se_used)))
    else:
        p_values = [np.nan] * len(param_names)

    print("\n📊 Estimates, Standard Errors, and p-values:")
    print(f"{'Parameter':16s} {'Estimate':>10} {'Std. Error':>12} {'p-value':>10}")
    print("-" * 50)
    for name, est, se, p in zip(param_names, base_params, se_used, p_values):
        print(f"{name:16s} {est:10.6f} {se:12.6f} {p:10.4f}")

    # Save compact CSV
    os.makedirs("figures", exist_ok=True)
    df_results = pd.DataFrame({
        "Parameter": param_names,
        "Estimate": base_params,
        "StdError": se_used,
        "p_value": p_values
    })
    df_results.to_csv(os.path.join(BASE_DIR, "estimates_with_pvalues.csv"), index=False)

    # ------------------ Build and Save Labeled Table ----------------------------------------------

    # Automated readable labels
    plot_labels = [name.replace("_", " ").title() for name in param_names]

    # Automated LaTeX labels
    latex_labels = [f"${name.replace('_', '\\_')}$" for name in param_names]

    # Automated parameter types based on name
    def infer_param_type(name):
        if name.startswith("mu"):
            return "mu"
        elif name.startswith("beta"):
            return "beta"
        elif name.startswith("sigma"):
            return "sigma"
        elif name.startswith("alpha"):
            return "alpha"
        elif name.startswith("gamma"):
            return "gamma"
        else:
            return "other"

    param_types = [infer_param_type(name) for name in param_names]

    # Final labeled DataFrame
    df_results_labeled = pd.DataFrame({
        "Parameter": param_names,
        "Estimate": base_params,
        "StdError": se_used,
        "p_value": p_values,
        "Label": plot_labels,
        "LaTeX": latex_labels,
        "Group": param_types
    })

    # Save
    diagnostics_path = os.path.join(BASE_DIR, "estimates_diagnostics.csv")
    df_results_labeled.to_csv(diagnostics_path, index=False)
#    print(f"📁 Saved labeled diagnostics to {diagnostics_path}")

# ------------------------ Toggles for Diagnostic Plots ---------------------------------------------

ANNOTATE_EITC_EXPANSIONS = False       # Set to False to suppress EITC expansion markers

EITC_EXPANSION_EVENTS = {
    1978: "Permanent EITC",
    1986: "TRA86 Expansion",
    1990: "OBRA90: 2 Child Tier",
    1993: "OBRA93: Major Boost",
    2001: "EGTRRA: Marriage Relief",
    2009: "ARRA: 3 Child Tier"
}

PLOT_DIAGNOSTICS = False               # Master toggle to enable or disable three plots on actual data
SAVE_LOG_WAGE_PLOT = True
SAVE_MEAN_WAGE_PLOT = True
SAVE_EMPLOYMENT_SHARE_PLOT = True

LOG_WAGE_PLOT_PATH = os.path.join(BASE_DIR, "wage_log_distributions_overlay.png")
MEAN_WAGE_PLOT_PATH = os.path.join(BASE_DIR, "mean_raw_wages_by_sector_over_time.png")
EMPLOYMENT_SHARE_PLOT_PATH = os.path.join(BASE_DIR, "employment_shares_by_type_over_time.png")

PLOT_SIMULATED_WAGE_SUMMARY = False    # Set to True to enable plot for simulated wage data

# ------------------------ Print Simulated Choices and Utilities at Estimated Parameters ------------

# Print average simulated utilities and plot average simulated utilities by year
print()
print("✅ Final Mean Utilities:", df_sim["U0"].mean(), df_sim["U1"].mean(), df_sim["U2"].mean())
print()

df_sim.groupby("year")[["U0", "U1", "U2"]].mean().plot(
    figsize=(10, 6), title="Mean Utilities Over Time"
)
plt.ylabel("Mean Utility")
plt.grid(True)

if ANNOTATE_EITC_EXPANSIONS:
    for year, label in EITC_EXPANSION_EVENTS.items():
        plt.axvline(x=year, color="red", linestyle="--", alpha=0.5)
        plt.text(year + 0.1, plt.ylim()[1]*0.9, label, rotation=90,
                 color="red", fontsize=8, verticalalignment='center')

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "mean_utilities_by_year.png"), dpi=300)
#print("✅ Saved utility plot to mean_utilities_by_year.png")
plt.show()

# ------------------------- Print Summary of Simulated Wages: Means and Standard Deviations ---------

df_sim["tax_pt"] = df_sim["wage_pt"] - df_sim["wage_pt_aftertax"]
df_sim["tax_ft"] = df_sim["wage_ft"] - df_sim["wage_ft_aftertax"]
df_sim["net_to_gross_ratio_pt"] = df_sim["wage_pt_aftertax"] / df_sim["wage_pt"]
df_sim["net_to_gross_ratio_ft"] = df_sim["wage_ft_aftertax"] / df_sim["wage_ft"]
df_sim["real_wage_pt_net"] = df_sim["real_wage_pt_aftertax"]
df_sim["real_wage_ft_net"] = df_sim["real_wage_ft_aftertax"]

wage_diagnostics = {
    "Variable": [
        "wage_pt", "wage_ft",
        "wage_pt_aftertax", "wage_ft_aftertax",
        "real_wage_pt", "real_wage_ft",
        "real_wage_pt_aftertax", "real_wage_ft_aftertax"
    ],
    "Mean": [
        df_sim["wage_pt"].mean(),
        df_sim["wage_ft"].mean(),
        df_sim["wage_pt_aftertax"].mean(),
        df_sim["wage_ft_aftertax"].mean(),
        df_sim["real_wage_pt"].mean(),
        df_sim["real_wage_ft"].mean(),
        df_sim["real_wage_pt_aftertax"].mean(),
        df_sim["real_wage_ft_aftertax"].mean()
    ],
    "Std Dev": [
        df_sim["wage_pt"].std(),
        df_sim["wage_ft"].std(),
        df_sim["wage_pt_aftertax"].std(),
        df_sim["wage_ft_aftertax"].std(),
        df_sim["real_wage_pt"].std(),
        df_sim["real_wage_ft"].std(),
        df_sim["real_wage_pt_aftertax"].std(),
        df_sim["real_wage_ft_aftertax"].std()
    ]
}

df_wage_diag = pd.DataFrame(wage_diagnostics)
print("\n📊 Simulated Wage Diagnostics Summary:")
print()
print(df_wage_diag.to_string(index=False))

# ------------------------- Compare Simulated and Actual Nominal Gross Hourly Wages Over Time -------

# Simulated nominal wages (not deflated)
sim_nominal_wages_by_year = pd.DataFrame(index=mean_wage_by_year.index)
sim_nominal_wages_by_year["Part-Time (Sim, Nominal)"] = df_sim[df_sim["sim_part"] == 1].groupby("year")["wage_pt"].mean()
sim_nominal_wages_by_year["Full-Time (Sim, Nominal)"] = df_sim[df_sim["sim_full"] == 1].groupby("year")["wage_ft"].mean()

# Re-inflate actual real wages back to nominal
df_actual_nominal = df_all[df_all["employed_ptft_robert"].isin([1, 2]) & df_all["wage"].notna()].copy()
df_actual_nominal["cpi"] = df_actual_nominal["year"].map(CPI_BY_YEAR)
df_actual_nominal["nominal_wage"] = df_actual_nominal["wage"] * (df_actual_nominal["cpi"] / 100.0)

actual_nominal_wages_by_year = df_actual_nominal.groupby(["year", "employed_ptft_robert"])["nominal_wage"].mean().unstack()
actual_nominal_wages_by_year.columns = ["Part-Time (Actual, Nominal)", "Full-Time (Actual, Nominal)"]

# Merge
wage_compare_nominal = actual_nominal_wages_by_year.copy()
wage_compare_nominal["Part-Time (Sim, Nominal)"] = sim_nominal_wages_by_year["Part-Time (Sim, Nominal)"]
wage_compare_nominal["Full-Time (Sim, Nominal)"] = sim_nominal_wages_by_year["Full-Time (Sim, Nominal)"]

# Plot
plt.figure(figsize=(10, 6))

plt.plot(
    wage_compare_nominal.index,
    wage_compare_nominal["Part-Time (Actual, Nominal)"],
    label="Part-Time (Actual, Nominal)",
    linestyle="-",
    marker="o"
)

plt.plot(
    wage_compare_nominal.index,
    wage_compare_nominal["Full-Time (Actual, Nominal)"],
    label="Full-Time (Actual, Nominal)",
    linestyle="-",
    marker="s"
)

plt.plot(
    wage_compare_nominal.index,
    wage_compare_nominal["Part-Time (Sim, Nominal)"],
    label="Part-Time (Sim, Nominal)",
    linestyle="--",
    marker="o"
)

plt.plot(
    wage_compare_nominal.index,
    wage_compare_nominal["Full-Time (Sim, Nominal)"],
    label="Full-Time (Sim, Nominal)",
    linestyle="--",
    marker="s"
)
plt.xlabel("Year")
plt.ylabel("Mean Gross Hourly Wage ($, Nominal)")
plt.title("Actual vs Simulated Nominal Mean Gross Hourly Wage by Sector Over Time")
plt.legend()
plt.grid(True)

if ANNOTATE_EITC_EXPANSIONS:
    for year, label in EITC_EXPANSION_EVENTS.items():
        plt.axvline(x=year, color="red", linestyle="--", alpha=0.5)
        plt.text(year + 0.1, plt.ylim()[1]*0.9, label, rotation=90,
                 color="red", fontsize=8, verticalalignment='center')

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "nominal_mean_wage_by_sector_fit.png"), dpi=300)
#print("✅ Saved nominal wage plot to nominal_mean_wage_by_sector_fit.png")
plt.show()

# ------------------------- Compare Simulated and Actual Real Gross Hourly Wages Over Time ----------

# Simulated real wages are already deflated in df_sim: real_wage_pt and real_wage_ft

sim_real_wages_by_year = pd.DataFrame(index=mean_wage_by_year.index)
sim_real_wages_by_year["Part-Time (Sim, Real)"] = df_sim[df_sim["sim_part"] == 1].groupby("year")["real_wage_pt"].mean()
sim_real_wages_by_year["Full-Time (Sim, Real)"] = df_sim[df_sim["sim_full"] == 1].groupby("year")["real_wage_ft"].mean()

# Now build actual real wages (already deflated in the dataset)
df_actual_real = df_all[df_all["employed_ptft_robert"].isin([1, 2]) & df_all["wage"].notna()].copy()
df_actual_real["real_wage"] = df_actual_real["wage"]

actual_real_wages_by_year = df_actual_real.groupby(["year", "employed_ptft_robert"])["real_wage"].mean().unstack()
actual_real_wages_by_year.columns = ["Part-Time (Actual, Real)", "Full-Time (Actual, Real)"]

# Merge
wage_compare_real = actual_real_wages_by_year.copy()
wage_compare_real["Part-Time (Sim, Real)"] = sim_real_wages_by_year["Part-Time (Sim, Real)"]
wage_compare_real["Full-Time (Sim, Real)"] = sim_real_wages_by_year["Full-Time (Sim, Real)"]

# Plot
plt.figure(figsize=(10, 6))

plt.plot(
    wage_compare_real.index,
    wage_compare_real["Part-Time (Actual, Real)"],
    label="Part-Time (Actual, Real)",
    linestyle="-",
    marker="o"
)

plt.plot(
    wage_compare_real.index,
    wage_compare_real["Full-Time (Actual, Real)"],
    label="Full-Time (Actual, Real)",
    linestyle="-",
    marker="s"
)

plt.plot(
    wage_compare_real.index,
    wage_compare_real["Part-Time (Sim, Real)"],
    label="Part-Time (Sim, Real)",
    linestyle="--",
    marker="o"
)

plt.plot(
    wage_compare_real.index,
    wage_compare_real["Full-Time (Sim, Real)"],
    label="Full-Time (Sim, Real)",
    linestyle="--",
    marker="s"
)
plt.xlabel("Year")
plt.ylabel("Mean Gross Hourly Wage (Real 1983 $)")
plt.title("Actual vs Simulated Real Mean Gross Hourly Wage by Sector Over Time")
plt.legend()
plt.grid(True)

if ANNOTATE_EITC_EXPANSIONS:
    for year, label in EITC_EXPANSION_EVENTS.items():
        plt.axvline(x=year, color="red", linestyle="--", alpha=0.5)
        plt.text(year + 0.1, plt.ylim()[1]*0.9, label, rotation=90,
                 color="red", fontsize=8, verticalalignment='center')

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "real_mean_wage_by_sector_fit.png"), dpi=300)
print("✅ Saved real mean wage fit plot to real_mean_wage_by_sector_fit.png")
plt.show()

# ------------------------- Compare Simulated and Actual Employment Choices by Year -----------------

actual_by_year = df_all.groupby("year")[["actual_nonemp", "actual_part", "actual_full"]].mean()
sim_by_year = df_sim.groupby("year")[["sim_nonemp", "sim_part", "sim_full"]].mean()

plt.figure(figsize=(10, 6))
plt.plot(actual_by_year.index, actual_by_year["actual_nonemp"], label="Nonemp (actual)", linestyle="-")
plt.plot(actual_by_year.index, actual_by_year["actual_part"], label="Part-time (actual)", linestyle="-")
plt.plot(actual_by_year.index, actual_by_year["actual_full"], label="Full-time (actual)", linestyle="-")
plt.plot(sim_by_year.index, sim_by_year["sim_nonemp"], label="Nonemp (simulated)", linestyle="--")
plt.plot(sim_by_year.index, sim_by_year["sim_part"], label="Part-time (simulated)", linestyle="--")
plt.plot(sim_by_year.index, sim_by_year["sim_full"], label="Full-time (simulated)", linestyle="--")
plt.xlabel("Year")
plt.ylabel("Proportion")
plt.title("Actual vs Simulated Choice Shares by Year (Base Model)")
plt.legend()
plt.grid(True)
plt.tight_layout()

if ANNOTATE_EITC_EXPANSIONS:
    for year, label in EITC_EXPANSION_EVENTS.items():
        plt.axvline(x=year, color="red", linestyle="--", alpha=0.5)
        plt.text(year + 0.1, plt.ylim()[1]*0.9, label, rotation=90,
                 color="red", fontsize=8, verticalalignment='center')

plt.savefig(os.path.join(BASE_DIR, "choice_shares_by_year.png"), dpi=300)
print("✅ Saved choice shares by year to choice_shares_by_year.png")
plt.show()

# ------------------------- Diagnostic: Actual vs Simulated Choice Shares by Education Level --------

# Group by education level (1 = low, 2 = mid, 3 = high)
edu_levels = [1, 2, 3]

summary_by_edu = []

for level in edu_levels:
    row = {
        "Education": level,
        "Actual Nonemp": df_all[df_all["edu"] == level]["actual_nonemp"].mean(),
        "Simulated Nonemp": df_sim[df_sim["edu"] == level]["sim_nonemp"].mean(),
        "Actual Part-time": df_all[df_all["edu"] == level]["actual_part"].mean(),
        "Simulated Part-time": df_sim[df_sim["edu"] == level]["sim_part"].mean(),
        "Actual Full-time": df_all[df_all["edu"] == level]["actual_full"].mean(),
        "Simulated Full-time": df_sim[df_sim["edu"] == level]["sim_full"].mean(),
    }
    summary_by_edu.append(row)

summary_df = pd.DataFrame(summary_by_edu)

# Print the table
print("\n📊 Choice Shares by Education Level (Actual vs Simulated):")
print(summary_df.to_string(index=False))

# Save summary table
summary_df.to_csv(os.path.join(BASE_DIR, "choice_shares_by_education.csv"), index=False)
print("✅ Saved grouped summary table to choice_shares_by_education.csv")

# Optional: plot grouped bars
fig, ax = plt.subplots(figsize=(10, 6))

bar_width = 0.12
x = range(len(edu_levels))

for i, category in enumerate(["Nonemp", "Part-time", "Full-time"]):
    actual = summary_df[f"Actual {category}"]
    sim = summary_df[f"Simulated {category}"]
    ax.bar([pos + i * bar_width for pos in x], actual, bar_width, label=f"Actual {category}")
    ax.bar([pos + i * bar_width + 3 * bar_width for pos in x], sim, bar_width, label=f"Simulated {category}", alpha=0.7)

ax.set_xticks([pos + 1.5 * bar_width for pos in x])
ax.set_xticklabels(["Low", "Mid", "High"])
ax.set_ylabel("Proportion")
ax.set_title("Employment Type Shares by Education Level")
ax.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "choice_shares_by_education.png"), dpi=300)
print("✅ Saved grouped bar chart to choice_shares_by_education.png")
plt.show()

# ------------------------- Compare Simulated and Actual Employment Choices by Age ------------------

# Group by age and take means over all years
actual_by_age = df_all.groupby("age")[["actual_nonemp", "actual_part", "actual_full"]].mean()
sim_by_age = df_sim.groupby("age")[["sim_nonemp", "sim_part", "sim_full"]].mean()

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(actual_by_age.index, actual_by_age["actual_nonemp"], label="Nonemp (actual)", linestyle="-")
plt.plot(actual_by_age.index, actual_by_age["actual_part"], label="Part-time (actual)", linestyle="-")
plt.plot(actual_by_age.index, actual_by_age["actual_full"], label="Full-time (actual)", linestyle="-")
plt.plot(sim_by_age.index, sim_by_age["sim_nonemp"], label="Nonemp (simulated)", linestyle="--")
plt.plot(sim_by_age.index, sim_by_age["sim_part"], label="Part-time (simulated)", linestyle="--")
plt.plot(sim_by_age.index, sim_by_age["sim_full"], label="Full-time (simulated)", linestyle="--")
plt.xlabel("Age")
plt.ylabel("Proportion")
plt.title("Actual vs Simulated Choice Shares by Age (Averaged Over Years)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "choice_shares_by_age.png"), dpi=300)
print("✅ Saved choice shares by age to choice_shares_by_age.png")
plt.show()

# -------------------- Diagnostic: Actual vs Simulated Choice Shares by Number of Children ----------

# Define child bins
df_all["child_bin"] = df_all["children"].clip(upper=3)
df_sim["child_bin"] = df_sim["children"].clip(upper=3)

child_bins = [0, 1, 2, 3]
summary_by_child = []

for val in child_bins:
    row = {
        "Children": f"{val}" if val < 3 else "3+",
        "Actual Nonemp": df_all[df_all["child_bin"] == val]["actual_nonemp"].mean(),
        "Simulated Nonemp": df_sim[df_sim["child_bin"] == val]["sim_nonemp"].mean(),
        "Actual Part-time": df_all[df_all["child_bin"] == val]["actual_part"].mean(),
        "Simulated Part-time": df_sim[df_sim["child_bin"] == val]["sim_part"].mean(),
        "Actual Full-time": df_all[df_all["child_bin"] == val]["actual_full"].mean(),
        "Simulated Full-time": df_sim[df_sim["child_bin"] == val]["sim_full"].mean(),
    }
    summary_by_child.append(row)

summary_df = pd.DataFrame(summary_by_child)

# Print the table
print("\n📊 Choice Shares by Number of Children (Actual vs Simulated):")
print(summary_df.to_string(index=False))

# Save summary table
summary_df.to_csv(os.path.join(BASE_DIR, "choice_shares_by_children.csv"), index=False)
print("✅ Saved grouped summary table to choice_shares_by_children.csv")

# Optional: plot grouped bars
fig, ax = plt.subplots(figsize=(10, 6))

bar_width = 0.12
x = range(len(child_bins))

for i, category in enumerate(["Nonemp", "Part-time", "Full-time"]):
    actual = summary_df[f"Actual {category}"]
    sim = summary_df[f"Simulated {category}"]
    ax.bar([pos + i * bar_width for pos in x], actual, bar_width, label=f"Actual {category}")
    ax.bar([pos + i * bar_width + 3 * bar_width for pos in x], sim, bar_width, label=f"Simulated {category}", alpha=0.7)

ax.set_xticks([pos + 1.5 * bar_width for pos in x])
ax.set_xticklabels(["0", "1", "2", "3+"])
ax.set_ylabel("Proportion")
ax.set_title("Employment Type Shares by Number of Children")
ax.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "choice_shares_by_children.png"), dpi=300)
print("✅ Saved grouped bar chart to choice_shares_by_children.png")
plt.show()

# -------------------- Diagnostic: Actual vs Simulated Choice Shares by Race --------------------

# Define race categories
race_labels = {
    1: "White",
    2: "Black",
    3: "Other"
}
race_values = sorted(race_labels.keys())

summary_by_race = []

for val in race_values:
    row = {
        "Race": race_labels[val],
        "Actual Nonemp": df_all[df_all["race"] == val]["actual_nonemp"].mean(),
        "Simulated Nonemp": df_sim[df_sim["race"] == val]["sim_nonemp"].mean(),
        "Actual Part-time": df_all[df_all["race"] == val]["actual_part"].mean(),
        "Simulated Part-time": df_sim[df_sim["race"] == val]["sim_part"].mean(),
        "Actual Full-time": df_all[df_all["race"] == val]["actual_full"].mean(),
        "Simulated Full-time": df_sim[df_sim["race"] == val]["sim_full"].mean(),
    }
    summary_by_race.append(row)

summary_df = pd.DataFrame(summary_by_race)

# Print the table
print("\n📊 Choice Shares by Race (Actual vs Simulated):")
print(summary_df.to_string(index=False))

# Save summary table
summary_df.to_csv(os.path.join(BASE_DIR, "choice_shares_by_race.csv"), index=False)
print("✅ Saved grouped summary table to choice_shares_by_race.csv")

# Optional: plot grouped bars
fig, ax = plt.subplots(figsize=(10, 6))

bar_width = 0.12
x = range(len(race_values))

for i, category in enumerate(["Nonemp", "Part-time", "Full-time"]):
    actual = summary_df[f"Actual {category}"]
    sim = summary_df[f"Simulated {category}"]
    ax.bar([pos + i * bar_width for pos in x], actual, bar_width, label=f"Actual {category}")
    ax.bar([pos + i * bar_width + 3 * bar_width for pos in x], sim, bar_width, label=f"Simulated {category}", alpha=0.7)

ax.set_xticks([pos + 1.5 * bar_width for pos in x])
ax.set_xticklabels([race_labels[val] for val in race_values])
ax.set_ylabel("Proportion")
ax.set_title("Employment Type Shares by Race")
ax.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "choice_shares_by_race.png"), dpi=300)
print("✅ Saved grouped bar chart to choice_shares_by_race.png")
plt.show()

# ------------------------- Build Moment Diagnostics ------------------------------------------------

# Collect actual and simulated moments
actual_values = np.array([actual for (_, actual, _) in moment_definitions])
simulated_values_array = np.array([simulated_moments[moment_name] for (_, _, moment_name) in moment_definitions])
moment_labels = [label for (label, _, _) in moment_definitions]

# Compute differences
moment_diff = simulated_values_array - actual_values

# Create diagnostics table
diagnostics = pd.DataFrame({
    "Moment": moment_labels,
    "Actual": actual_values,
    "Simulated": simulated_values_array,
    "Difference": moment_diff
})

# Print diagnostics
print("\n📊 Base Model Moment Diagnostics:")
print(diagnostics.to_string(index=False))
print("\nSum of Squared Differences:", np.sum(moment_diff**2))
print()

# Save diagnostics
os.makedirs("figures", exist_ok=True)
diagnostics.to_csv(os.path.join(BASE_DIR, "moment_diagnostics.csv"), index=False)
#print("✅ Saved diagnostics to moment_diagnostics.csv")

# ------------------------- Print Moment Weights ----------------------------------------------------

print("Index | Raw Weight       | Mean-Normalized  | Log-Normalized  | Manual-Scaled   | Final W Used")
print("------------------------------------------------------------------------------------------------")
for i in range(num_moments):
    label = moment_definitions[i][2]
    w_base = W_base[i]
    w_mean = W_mean_normalized[i]
    w_log = W_log_normalized[i]
    w_manual = W_manual_scaled[i] if 'W_manual_scaled' in locals() else float('nan')
    w_final = W[i]
    print(f"{i+1:5d} | {w_base:15.3f} | {w_mean:15.6f} | {w_log:15.6f} | {w_manual:15.6f} | {w_final:15.6f}  {label}")

# ------------------------- Plots for Actual Data ---------------------------------------------------

if PLOT_DIAGNOSTICS:

    # Plot original and winsorized log wage distributions
    log_wage_original = np.log(df_all["wage_original"][df_all["wage_original"] > 0])
    log_wage_winsorized = np.log(df_all["wage"][df_all["wage"] > 0])

    bin_edges = np.histogram_bin_edges(
        np.concatenate([log_wage_original, log_wage_winsorized]), bins=100
    )

    plt.figure(figsize=(10, 6))
    sns.histplot(log_wage_original, bins=bin_edges, color="steelblue", label="Original", alpha=0.6)
    sns.histplot(log_wage_winsorized, bins=bin_edges, color="seagreen", label="Winsorized", alpha=0.6)

    plt.title("Overlay: Log Wage Distributions (Original vs Winsorized)")
    plt.xlabel("Log Wage")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if SAVE_LOG_WAGE_PLOT:
        plt.savefig(LOG_WAGE_PLOT_PATH)
        print(f"✅ Saved overlay log wage histogram to {LOG_WAGE_PLOT_PATH}")
        plt.show()

    # Plot Actual Mean Real Gross Hourly Wages by Sector and Year
    plt.figure(figsize=(10, 6))
    plt.plot(mean_wage_by_year.index, mean_wage_by_year["Part-Time"], label="Part-Time", linestyle="--", marker="o")
    plt.plot(mean_wage_by_year.index, mean_wage_by_year["Full-Time"], label="Full-Time", linestyle="-", marker="s")
    plt.xlabel("Year")
    plt.ylabel("Actual Mean Gross Hourly Wage (Real 1983 $)")
    plt.title("Actual Mean Gross Hourly Wage by Sector Over Time")
    plt.legend()
    plt.grid(True)

    if ANNOTATE_EITC_EXPANSIONS:
        for year, label in EITC_EXPANSION_EVENTS.items():
            plt.axvline(x=year, color="red", linestyle="--", alpha=0.5)
            plt.text(year + 0.1, plt.ylim()[1]*0.9, label, rotation=90,
                     color="red", fontsize=8, verticalalignment='center')


    plt.tight_layout()

    if SAVE_MEAN_WAGE_PLOT:
        plt.savefig(MEAN_WAGE_PLOT_PATH, dpi=300)
        print(f"✅ Saved to {MEAN_WAGE_PLOT_PATH}")
        plt.show()

    # Plot Actual Employment Shares by Type Over Time
    plt.figure(figsize=(10, 6))
    plt.plot(employment_shares.index, employment_shares["actual_nonemp"], label="Non-Employed", linestyle="-", marker="o")
    plt.plot(employment_shares.index, employment_shares["actual_part"], label="Part-Time", linestyle="--", marker="s")
    plt.plot(employment_shares.index, employment_shares["actual_full"], label="Full-Time", linestyle=":", marker="^")
    plt.xlabel("Year")
    plt.ylabel("Share of Individuals")
    plt.title("Actual Employment Shares by Type Over Time")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)

    if ANNOTATE_EITC_EXPANSIONS:
        for year, label in EITC_EXPANSION_EVENTS.items():
            plt.axvline(x=year, color="red", linestyle="--", alpha=0.5)
            plt.text(year + 0.1, plt.ylim()[1]*0.9, label, rotation=90,
                     color="red", fontsize=8, verticalalignment='center')

    plt.tight_layout()

    if SAVE_EMPLOYMENT_SHARE_PLOT:
        plt.savefig(EMPLOYMENT_SHARE_PLOT_PATH, dpi=300)
        print(f"✅ Saved to {EMPLOYMENT_SHARE_PLOT_PATH}")
        plt.show()

# Simulated Wage Plotting

if PLOT_SIMULATED_WAGE_SUMMARY:
    # Group by year (Simulated Data)
    df_yearly = df_sim.groupby("year").agg({
        "wage_pt": "mean",
        "wage_ft": "mean",
        "wage_pt_aftertax": "mean",
        "wage_ft_aftertax": "mean",
        "real_wage_pt": "mean",
        "real_wage_ft": "mean",
        "tax_pt": "mean",
        "tax_ft": "mean",
        "net_to_gross_ratio_pt": "mean",
        "net_to_gross_ratio_ft": "mean",
        "cpi": "mean"
    }).reset_index()

    # Group by year for real net wages
    df_net_wages = df_sim.groupby("year")[["real_wage_pt_net", "real_wage_ft_net"]].mean().reset_index()

    # Plot 1: Net-to-Gross Ratios (Simulated Data)
    plt.figure(figsize=(10, 6))
    plt.plot(df_yearly["year"], df_yearly["net_to_gross_ratio_pt"], label="PT Net-to-Gross")
    plt.plot(df_yearly["year"], df_yearly["net_to_gross_ratio_ft"], label="FT Net-to-Gross")
    plt.title("Net-to-Gross Wage Ratios Over Time (Simulated Data)")
    plt.xlabel("Year")
    plt.ylabel("Ratio")
    plt.legend()
    plt.grid(True)

    if ANNOTATE_EITC_EXPANSIONS:
        for year, label in EITC_EXPANSION_EVENTS.items():
            plt.axvline(x=year, color="red", linestyle="--", alpha=0.5)
            plt.text(year + 0.1, plt.ylim()[1]*0.9, label, rotation=90,
                     color="red", fontsize=8, verticalalignment='center')

    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "net_to_gross_ratios_over_time.png"))
    plt.show()

    # Plot 2: Real Gross Wages (Simulated Data)
    plt.figure(figsize=(10, 6))
    plt.plot(df_yearly["year"], df_yearly["real_wage_pt"], label="PT Real Wage (Gross)")
    plt.plot(df_yearly["year"], df_yearly["real_wage_ft"], label="FT Real Wage (Gross)")
    plt.title("Real Gross Wages Over Time (Simulated Data)")
    plt.xlabel("Year")
    plt.ylabel("Real Wage")
    plt.legend()
    plt.grid(True)

    if ANNOTATE_EITC_EXPANSIONS:
        for year, label in EITC_EXPANSION_EVENTS.items():
            plt.axvline(x=year, color="red", linestyle="--", alpha=0.5)
            plt.text(year + 0.1, plt.ylim()[1]*0.9, label, rotation=90,
                     color="red", fontsize=8, verticalalignment='center')

    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "real_gross_wages_over_time.png"))
    plt.show()

    # Plot 3: Tax Burden (Simulated Data)
    plt.figure(figsize=(10, 6))
    plt.plot(df_yearly["year"], df_yearly["tax_pt"], label="PT Tax")
    plt.plot(df_yearly["year"], df_yearly["tax_ft"], label="FT Tax")
    plt.title("Tax Burden Over Time (Simulated Data)")
    plt.xlabel("Year")
    plt.ylabel("Tax ($/hr)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "tax_burden_over_time.png"))
    plt.show()

    # Plot 4: Real Net Wages (Simulated Data)
    plt.figure(figsize=(10, 6))
    plt.plot(df_net_wages["year"], df_net_wages["real_wage_pt_net"], label="PT Real Wage (Net)")
    plt.plot(df_net_wages["year"], df_net_wages["real_wage_ft_net"], label="FT Real Wage (Net)")
    plt.title("Real Net Wages Over Time (Simulated Data)")
    plt.xlabel("Year")
    plt.ylabel("Real Net Wage ($)")
    plt.legend()
    plt.grid(True)

    if ANNOTATE_EITC_EXPANSIONS:
        for year, label in EITC_EXPANSION_EVENTS.items():
            plt.axvline(x=year, color="red", linestyle="--", alpha=0.5)
            plt.text(year + 0.1, plt.ylim()[1]*0.9, label, rotation=90,
                     color="red", fontsize=8, verticalalignment='center')

    plt.tight_layout()
    save_path = os.path.join(BASE_DIR, "real_net_wages_over_time.png")
    plt.savefig(save_path)
    print(f"✅ Saved to {save_path}")
    plt.show()

    # Plot 5: CPI Trend (Actual Data)
    plt.figure(figsize=(10, 6))

    # Select correct CPI series based on toggle
    if USE_R_CPI:
        cpi_label = "R-CPI-U-RS (1977=100)"
        cpi_data = CPI_1977_BASE
    else:
        cpi_label = "CPI-U (1983=100)"
        cpi_data = CPI_1983_BASE

    # Sort by year to ensure proper plotting
    years_sorted = sorted(cpi_data.keys())
    cpi_values = [cpi_data[y] for y in years_sorted]

    plt.plot(years_sorted, cpi_values, label=cpi_label, color="purple")
    plt.title(f"CPI Over Time ({cpi_label})")
    plt.xlabel("Year")
    plt.ylabel("CPI Index")
    plt.grid(True)

    if ANNOTATE_EITC_EXPANSIONS:
        for year, label in EITC_EXPANSION_EVENTS.items():
            plt.axvline(x=year, color="red", linestyle="--", alpha=0.5)
            plt.text(year + 0.1, plt.ylim()[1]*0.9, label, rotation=90,
                     color="red", fontsize=8, verticalalignment='center')

    plt.tight_layout()
    plt.legend()
    plt.savefig(os.path.join(BASE_DIR, "cpi_over_time.png"))
    plt.show()    
# --------------------------------------------------END PROGRAM--------------------------------------


# In[ ]:




