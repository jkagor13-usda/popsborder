import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, ttest_rel, f_oneway
import os

base_path = '/Users/robinms1/Library/CloudStorage/Box-Box/NHH15 - USDA APHIS EDISON/05 PPQ Engagement/PPQ RBS Data (Folder shared with APHIS)/APL Created Data Related Items/Model_Testing/Official_Results/'
experiments = ['Baseline', 'Model 1', 'Model 2', 'Model 3']

cols_of_interest = ['inspection_number',
                    'risk_unit_id', 'num_sample_units', 'num_plants', 'infected_plants',
                    'is_infected', 'is_detected', 'missed', 'was_inspected',
                    'inspected_sample_units']


def gather_experiment_data(filepath, experiments, cols_of_interest=cols_of_interest):
    '''
    Collect experiment data in one pandas DataFrame, with tags for experiment and replication number. Used to
    gather slippage data for analysis

    '''
    results_stack = []
    # for each experiment:
    for exp in experiments:

        exp_path = base_path + exp + '/Replications/'
        # Get all replications - assuming structure remains the same, no other files
        reps = os.listdir(exp_path)

        # for each replication:
        for rep in reps:
            # Collect all relevant info
            df_rep = pd.read_csv(exp_path + rep + '/synthetic_commodity_line_results_data.csv',
                                 usecols=cols_of_interest)

            # assuming same naming conventions (e.g., rep_0, rep_1, ...)
            df_rep['replication'] = rep
            df_rep['experiment'] = exp

            results_stack.append(df_rep)

    df_results = pd.concat(results_stack)

    return df_results


df_results = gather_experiment_data(base_path, experiments)

# Confirming these are all the same counts as expected
df_results['replication'].value_counts().head()

# Confirming these are all the same counts as expected
df_results['experiment'].value_counts()


df = df_results.copy()

df['num_plants_slipped'] = df['missed'].astype(int) * df['infected_plants']
df['prop_inspected'] = df['inspected_sample_units'] / df['num_plants']

total_slippage = df.groupby(['experiment', 'replication'])['num_plants_slipped'].sum().reset_index()

n_inspected_sample_units = df.groupby(['experiment', 'replication', 'inspection_number'])['inspected_sample_units'].sum().reset_index()
mean_inspected_sample_units = n_inspected_sample_units.groupby(['experiment', 'replication'])['inspected_sample_units'].mean().reset_index()

# Some outliers - rep 37 & 49
total_slippage.sort_values('num_plants_slipped').tail(15)

sns.boxplot(total_slippage, x='experiment', y='num_plants_slipped')


plt.ylabel('Number of Plants Slipped per Replication')
plt.xlabel('Experiment')
plt.title('Replication-Level Plant Slippage')


# TODO: Get summary numbers - quartiles etc here

sns.boxplot(total_slippage, x='experiment', y='num_plants_slipped')
plt.yscale('log')

plt.ylabel('Number of Plants Slipped per Replication')
plt.xlabel('Experiment')
plt.title('Replication-Level Plant Slippage')


# Reducing to <10k slippage
sns.boxplot(total_slippage[total_slippage['num_plants_slipped'] < 1e4], x='experiment', y='num_plants_slipped')
#plt.yscale('log')

plt.ylabel('Number of Plants Slipped per Replication')
plt.xlabel('Experiment')
plt.title('Replication-Level Plant Slippage')



total_slippage['log num_plants_slipped'] = np.log10(total_slippage['num_plants_slipped'])

total_slippage.groupby('experiment')['log num_plants_slipped'].plot(kind='kde')
plt.legend()

plt.xlim([0, 5])
plt.xlabel('log Plants Slipped')



# Total inspected sample units across all replications
n_inspected_sample_units.groupby('experiment')['inspected_sample_units'].sum()

# Why are these all the same? -- as expected
replication_samples = n_inspected_sample_units.groupby(['experiment', 'replication'])['inspected_sample_units'].sum()
print(replication_samples)



combined_results = replication_samples.reset_index().merge(total_slippage, on=['experiment', 'replication'])
print(combined_results)





sns.scatterplot(combined_results, x='inspected_sample_units', y='num_plants_slipped', hue='experiment')
plt.yscale('log')

plt.title('Plant Slippage vs Inspected Sample Units, Replication-Level')

#plt.xlim(left=0)




#### MAIN PLOT ####
mean_combined_results = combined_results.groupby('experiment')[['inspected_sample_units', 'num_plants_slipped']].mean()

sns.scatterplot(mean_combined_results, x='inspected_sample_units', y='num_plants_slipped', hue='experiment', s=50)
#plt.yscale('log')
plt.title('Plant Slippage vs Inspected Sample Units, Average')
# add poisson +/- st dev
#plt.ylim(bottom=0)
plt.legend(title='Experiment')
plt.xlabel('Sample Units Inspected')
plt.ylabel('# Plants Slipped')




std_combined_results = combined_results.groupby('experiment')[['inspected_sample_units', 'num_plants_slipped']].std() / np.sqrt(50)



sns.scatterplot(mean_combined_results, x='inspected_sample_units', y='num_plants_slipped', hue='experiment', s=50)

plt.errorbar(x=mean_combined_results['inspected_sample_units'], y=mean_combined_results['num_plants_slipped'], yerr=std_combined_results['num_plants_slipped'], ecolor=['blue', 'orange', 'green', 'red'], alpha=0.5, linestyle='')

#plt.yscale('log')
plt.title('Plant Slippage vs Inspected Sample Units, Average')
# add poisson +/- st dev
#plt.ylim(bottom=0)
plt.legend(title='Experiment', bbox_to_anchor=[0.5, 0.5])
plt.xlabel('Sample Units Inspected')
plt.ylabel('# Plants Slipped')




# One-way ANOVA - are any of the provided distributions statistically different from the others?
# Like a multidistributional analog of two-sample t-tests

all_slippages = [total_slippage[total_slippage['experiment'] == exp]['num_plants_slipped'].values for exp in total_slippage['experiment'].unique()]
f_oneway(*all_slippages)

experiment_labels = ['Baseline', 'Model 1', 'Model 2', 'Model 3']


def run_paired_ttests(total_slippage, experiment_labels):
    result_outputs = pd.DataFrame()

    k = 0
    for i in range(len(experiment_labels)):
        for j in range(i + 1, len(experiment_labels)):
            k += 1
            exp1 = experiment_labels[i]
            exp2 = experiment_labels[j]

            exp1_slippage = total_slippage[total_slippage['experiment'] == exp1].sort_values('replication')[
                'num_plants_slipped']
            exp2_slippage = total_slippage[total_slippage['experiment'] == exp2].sort_values('replication')[
                'num_plants_slipped']

            t_stat, p_val = ttest_rel(exp1_slippage, exp2_slippage)

            res = pd.DataFrame({'Experiment 1': exp1,
                                'Experiment 2': exp2,
                                't-statistic': t_stat,
                                'p-value': p_val}, index=[k])

            result_outputs = pd.concat([result_outputs, res])

    return result_outputs






# TODO: grab this
run_paired_ttests(total_slippage, experiment_labels=['Baseline', 'Model 1', 'Model 2', 'Model 3'])


agg_results = df_results.groupby(['experiment', 'replication'])[['num_plants_slipped', 'num_plants', 'inspected_sample_units', 'num_sample_units']].sum()
agg_results['prop_slipped'] = agg_results['num_plants_slipped'] / agg_results['num_plants']
agg_results['prop_sampled'] = agg_results['inspected_sample_units'] / agg_results['num_sample_units']


sns.scatterplot(agg_results, x='prop_sampled', y='prop_slipped', hue='experiment')
plt.yscale('log')

plt.title('Plant Slippage vs Inspected Sample Units, Replication-Level')


def calculate_slippage_metrics(df, experiment_labels=['high', 'low']):
    '''

    -- UNDER CONSTRUCTION --
    Given experiment data, calculate metrics for slippage

    Metrics to pull:
    - Total slippage (plants) per replication
    - - Total intercepted plants per replication
    - Total slipped sample units per replication
    - Mean inspected sample units per inspection number (per replication)
    - Total plants inspected (etc.)
    '''

    df['num_plants_slipped'] = df['missed'].astype(int) * df['infected_plants']
    df['prop_inspected'] = df['inspected_sample_units'] / df['num_plants']

    # Collect aggregate stats - per replication, per experiment

    # Plant-unit slippage
    total_slippage = df.groupby(['experiment', 'replication'])['num_plants_slipped'].sum().reset_index()

    # Number of inspected sample units per

    n_inspected_sample_units = df.groupby(['experiment', 'replication', 'inspection_number'])[
        'inspected_sample_units'].sum().reset_index()
    mean_inspected_sample_units = n_inspected_sample_units.groupby(['experiment', 'replication'])[
        'inspected_sample_units'].mean().reset_index()

    print(total_slippage)

    ## Slippage metrics

    ## Testing - check that only 2 experiments selected
    # Total slippage
    total_slippage0 = total_slippage[total_slippage['experiment'] == experiment_labels[0]]
    total_slippage1 = total_slippage[total_slippage['experiment'] == experiment_labels[1]]

    # Run two-sample t-test
    total_slippage_2stt = ttest_ind(total_slippage0['num_plants_slipped'], total_slippage1['num_plants_slipped'])

    # Run paired t-test
    total_slippage_ptt = ttest_rel(total_slippage0['num_plants_slipped'], total_slippage1['num_plants_slipped'])

    print(total_slippage_2stt)

    print(total_slippage_ptt)

calculate_slippage_metrics(df_results)



df_paired = pd.pivot_table(total_slippage[total_slippage['experiment'].isin(['Baseline', 'Model 1'])],
               values='num_plants_slipped', index='replication', columns='experiment')
df_paired.head()



symlogbins = [-1e4, -3e3, -1e3, -3e2, -1e2, -3e1, -1e1, -1, 0, 1, 1e1, 3e1, 1e2, 3e2, 1e3, 3e3, 1e4]


df_paired['Difference'] = df_paired['Model 1'] - df_paired['Baseline']

df_paired['Difference'].hist(bins=symlogbins, grid=False)
plt.ylabel('Count')
plt.xlabel('Slippage Difference (Model1-Baseline)')
plt.axvline(0, c='r', linestyle='--')
plt.xscale('symlog')

plt.title('Paired Replication-Level Slippage: Model 1 vs Baseline')


plt.figure(figsize=[6, 6])
xs = np.linspace(0, 30000, 10)
plt.scatter(df_paired['Baseline'], df_paired['Model 1'])
plt.plot(xs, xs, c='red', linestyle='--')

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Plant-Unit Slippage, Baseline')
plt.ylabel('Plant-Unit Slippage, Model 1')

plt.title('Model 1 vs Baseline Slippage by Replication')



# Number of replications where model 1 outperforms baseline
len(df_paired[df_paired['Difference'] < 0])



df_paired2 = pd.pivot_table(total_slippage[total_slippage['experiment'].isin(['Baseline', 'Model 2'])],
               values='num_plants_slipped', index='replication', columns='experiment')

df_paired2['Difference'] = df_paired2['Model 2'] - df_paired2['Baseline']


df_paired2['Difference'].hist(bins=symlogbins, grid=False)
plt.ylabel('Count')
plt.xlabel('Slippage Difference (Model2-Baseline)')
plt.axvline(0, c='r', linestyle='--')
plt.xscale('symlog')

plt.title('Paired Replication-Level Slippage: Model 2 vs Baseline')



plt.figure(figsize=[6, 6])
xs = np.linspace(0, 30000, 10)
plt.scatter(df_paired2['Baseline'], df_paired2['Model 2'])
plt.plot(xs, xs, c='red', linestyle='--')

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Plant-Unit Slippage, Baseline')
plt.ylabel('Plant-Unit Slippage, Model 2')

plt.title('Model 2 vs Baseline Slippage by Replication')



df_paired3 = pd.pivot_table(total_slippage[total_slippage['experiment'].isin(['Baseline', 'Model 3'])],
               values='num_plants_slipped', index='replication', columns='experiment')

df_paired3['Difference'] = df_paired3['Model 3'] - df_paired3['Baseline']


df_paired3['Difference'].hist(bins=symlogbins, grid=False)
plt.ylabel('Count')
plt.xlabel('Slippage Difference (Model3-Baseline)')
plt.axvline(0, c='r', linestyle='--')
plt.xscale('symlog')

plt.title('Paired Replication-Level Slippage: Model 3 vs Baseline')


plt.figure(figsize=[6, 6])
xs = np.linspace(0, 30000, 10)
plt.scatter(df_paired3['Baseline'], df_paired3['Model 3'])
plt.plot(xs, xs, c='red', linestyle='--')

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Plant-Unit Slippage, Baseline')
plt.ylabel('Plant-Unit Slippage, Model 3')

plt.title('Model 3 vs Baseline Slippage by Replication')




exp_units = df_results.groupby(['experiment', 'replication', 'inspection_number'])['inspected_sample_units'].sum().reset_index()
exp_inspected = exp_units.groupby(['experiment', 'replication'])['inspected_sample_units'].mean()

combined_results_full = combined_results.merge(exp_inspected.reset_index().rename(columns={'inspected_sample_units': 'inspected_sample_units_per_consignment'}))



mean_combined_results_consign = combined_results_full.groupby('experiment')[['inspected_sample_units_per_consignment', 'num_plants_slipped']].mean()

sns.scatterplot(mean_combined_results_consign, x='inspected_sample_units_per_consignment', y='num_plants_slipped', hue='experiment', s=50)
#plt.yscale('log')
plt.title('Plant Slippage vs Inspected Sample Units per Consignment, Average', fontsize=11)
# add poisson +/- st dev
#plt.ylim(bottom=0)
plt.legend(title='Experiment')
plt.xlabel('Average # Sample Units per Consignment Inspected')
plt.ylabel('# Plants Slipped')


combined_results_full.to_csv('replication-level_slippage-workload_04222026.csv')
