

import pandas as pd
from JoeFiles.mle_contamination import fit_beta_from_passfail

# Your dataframe with columns: Inspection_ID, action (0/1)
#df = pd.read_csv("C:\Users\agorjk1\Box\NHH15 - USDA APHIS EDISON\05 PPQ Engagement\PPQ RBS Data (Folder shared with APHIS)\APL Created Data Related Items\Synthetic_Data\synthetic_pis_data.csv")

# Set/assume detection rate e (0 < e ≤ 1), e.g. 0.8
#e = 0.8

#res = fit_beta_from_passfail(df,id_col='INSPECTION_ID', detection_rate=e)

#print("gamma_hat:", res["gamma_hat"])
#print("std_err:", res["std_err"])
#print("95% CI:", (res["ci95_lo"], res["ci95_hi"]))
#print("Beta params:", res["alpha"], res["beta"])

# Lot-level table used in the fit
#print(res["lots"].head())



#from JoeFiles.mle_contamination_glm import fit_cloglog_intercept_mle
#res = fit_cloglog_intercept_mle(df,id_col='INSPECTION_ID', detection_rate=0.8)
# res["mean_r"], res["theta_hat"], res["lambda_hat"], CI for mean_r, etc.





import pandas as pd
from JoeFiles.mle_contamination_glm import fit_beta_glm_match_failrate

# df needs columns: Inspection_ID, action (0/1)
df = pd.read_csv(r"C:\Users\agorjk1\Box\NHH15 - USDA APHIS EDISON\05 PPQ Engagement\PPQ RBS Data (Folder shared with APHIS)\APL Created Data Related Items\Synthetic_Data\synthetic_pis_data.csv")

e = 0.8                 # detection rate ε
#target = 0.049          # observed overall failure rate
target = df.loc[df["action"]==1].shape[0]/df.shape[0] # observed overall failure rate

res = fit_beta_glm_match_failrate(df,id_col='INSPECTION_ID', detection_rate=e, target_fail_rate=target, sims=1000)

print("Mean contamination rate (per unit):", res["mean_r"])
print("Beta params:", res["alpha"], res["beta"])
print("Simulated failure rate (check):", res["achieved_fail_rate"])
print('')
