from slippage_model_utils.clarke_r_script_wrapper import *
from slippage_model_utils.clarke_model_support_functions import *
from JoeFiles.pis_rbs_calculator_data import *

if __name__ == "__main__":

    ### Read in Data
    dir = r'C:\Users\agorjk1\Box\NHH15 - USDA APHIS EDISON\05 PPQ Engagement\PPQ RBS Data (Folder shared with APHIS)\APL Created Data Related Items\Synthetic_Data'
    filename = os.path.join(dir, f'synthetic_pis_data.csv')
    # Load in task log from a single replication of a run
    df_pis_data = pd.read_csv(filename)

    filename = os.path.join(dir, f'synthetic_rbs_calc_data.csv')
    # Load in task log from a single replication of a run
    df_rbs_calculator = pd.read_csv(filename)

    ### Generate clarke inputs via input data
    inputs = gen_clarke_model_inputs(df_pis_data,df_rbs_calculator)

    # Run clarke model
    res = run_clarke_bb_group_model(inputs.ty,
                                    inputs.b,
                                    inputs.B,
                                    inputs.Nbar,
                                    inputs.freq,
                                    inputs.theta,
                                    inputs.R,
                                    inputs.start_val,
                                    inputs.se)
    print(f'   Alpha = {res["alpha"]}')
    print(f'   Beta = {res["beta"]}')
    print('')




