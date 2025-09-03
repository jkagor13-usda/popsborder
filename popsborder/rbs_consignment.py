import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import random


def mask_categoricals(df, cat_cols):
    """
    Replace all values in the specified categorical columns with unique gibberish strings (one per unique value), preserving one-to-one mapping.
    Returns a new DataFrame with masked values.
    """
    import random
    import string
    df_masked = df.copy()
    for col in cat_cols:
        if col in df_masked.columns:
            unique_vals = df_masked[col].unique()
            gibberish_map = {}
            used_gibberish = set()
            for i, val in enumerate(unique_vals):
                gibberish = ''.join(random.choices(string.ascii_letters, k=8))
                while gibberish in used_gibberish:
                    gibberish = ''.join(random.choices(string.ascii_letters, k=8))
                gibberish_map[val] = gibberish
                used_gibberish.add(gibberish)
            df_masked[col] = df_masked[col].map(gibberish_map)
    return df_masked


def generate_synthetic_consignment_data(num_rows=None, output_file='synthetic_data.csv', input_file='PIS_RBS_Calc.csv', mask_cats=False):

    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import LabelEncoder
    import numpy as np

    # Columns to use
    cols = [
        'INSPECTION_NUMBER',
        'INSPECTION_LOCATION_NAME',
        'PATHWAY',
        'COUNTRY_OF_ORIGIN_NAME',
        'PROPAGATIVE_MATERIAL_TYPE',
        'TOTAL_SAMPLING_UNITS',
        'TOTAL_PLANT_QUANTITY',
        'PRODUCER',
    ]

    # Read CSV file
    df = pd.read_csv(input_file)
    df = df[cols].dropna()

    if num_rows is None:
        num_rows = len(df)

    # Encode categoricals
    cat_cols = [
        'INSPECTION_NUMBER',
        'INSPECTION_LOCATION_NAME',
        'PATHWAY',
        'COUNTRY_OF_ORIGIN_NAME',
        'PROPAGATIVE_MATERIAL_TYPE',
        'PRODUCER'
    ]
    encoders = {}
    df_encoded = df.copy()
    for col in cat_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        encoders[col] = le

    # Fit a Gaussian Mixture Model to the encoded data
    n_components = min(10, len(df))  # limit number of components
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
    gmm.fit(df_encoded.values)

    # Sample synthetic data from the GMM
    synth_array, _ = gmm.sample(num_rows)
    synth_array = np.round(synth_array).astype(int)
    synth_array = np.clip(synth_array, 0, None)  # ensure no negative values

    # Create DataFrame from synthetic array
    synth_data = pd.DataFrame(synth_array, columns=cols)

    # Decode categoricals
    for col in cat_cols:
        le = encoders[col]
        # Clip values to valid range for decoding
        synth_data[col] = synth_data[col].clip(0, len(le.classes_)-1)
        synth_data[col] = le.inverse_transform(synth_data[col].astype(int))

    # Optionally mask categorical columns (optionally including INSPECTION_NUMBER)
    mask_cols = cat_cols if mask_cats is True else (mask_cats if isinstance(mask_cats, list) else [])
    if mask_cols:
        synth_data = mask_categoricals(synth_data, mask_cols)
        print("Unique value counts in masked synthetic data:")
        print(synth_data.nunique())

    synth_data.to_csv(output_file, index=False)
    print(f"Synthetic data generated and saved to {output_file}")
    return df, synth_data


def encode_categoricals(df_in, cat_cols):
    df_out = df_in.copy()
    code_maps = {}
    for col in cat_cols:
        df_out[col] = df_out[col].astype('category')
        code_maps[col] = dict(enumerate(df_out[col].cat.categories))
        df_out[col] = df_out[col].cat.codes
    return df_out, code_maps


def plot_pairplots(df, synth_data, cat_cols, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Exclude INSPECTION_NUMBER from encoding and plotting
    plot_cols = [col for col in cat_cols if col in df.columns and col != 'INSPECTION_NUMBER']
    df_encoded, df_code_maps = encode_categoricals(df, plot_cols)
    synth_encoded, synth_code_maps = encode_categoricals(synth_data, plot_cols)
    print("Unique value counts in input data:")
    print(df_encoded.nunique())
    print("Unique value counts in synthetic data:")
    print(synth_encoded.nunique())
    for cat in plot_cols:
        n_colors = df_encoded[cat].nunique()
        palette = sns.color_palette('husl', n_colors)
        print(f"Generating pairplot for input data (coded, hue={cat})...")
        g = sns.pairplot(df_encoded, hue=cat, palette=palette, corner=True, plot_kws={'alpha':0.5, 's':15})
        plt.suptitle(f'Input Data Pairplot (Categoricals Coded, hue={cat})', y=1.02)
        if g.fig.legends and len(g.fig.legends) > 0:
            legend = g.fig.legends[0]
            texts = legend.get_texts()
            new_labels = [df_code_maps[cat].get(int(t.get_text()), t.get_text()) for t in texts]
            for t, new_label in zip(texts, new_labels):
                t.set_text(new_label)
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f'input_pairplot_hue_{cat}.png')
        plt.savefig(plot_path)
        plt.close()

        n_colors_synth = synth_encoded[cat].nunique()
        if n_colors_synth < 2:
            print(f"Skipping synthetic pairplot for {cat}: only {n_colors_synth} unique value(s) in synthetic data.")
            continue
        palette_synth = sns.color_palette('husl', n_colors_synth)
        print(f"Generating pairplot for synthetic data (coded, hue={cat})...")
        g = sns.pairplot(synth_encoded, hue=cat, palette=palette_synth, corner=True, plot_kws={'alpha':0.5, 's':15})
        plt.suptitle(f'Synthetic Data Pairplot (Categoricals Coded, hue={cat})', y=1.02)
        if g.fig.legends and len(g.fig.legends) > 0:
            legend = g.fig.legends[0]
            texts = legend.get_texts()
            new_labels = [synth_code_maps[cat].get(int(t.get_text()), t.get_text()) for t in texts]
            for t, new_label in zip(texts, new_labels):
                t.set_text(new_label)
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f'synth_pairplot_hue_{cat}.png')
        plt.savefig(plot_path)
        plt.close()

