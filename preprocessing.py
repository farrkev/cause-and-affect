'''Splits data by modality and applies PCA.'''

import pandas as pd
from sklearn.decomposition import PCA

processed_recola_df = pd.read_csv("./data/recola_all_participants_clean_preprocessed.csv")

AUDIO_COLS = [col for col in processed_recola_df.columns if col.startswith("ComParE")]
VISUAL_COLS = [col for col in processed_recola_df.columns if col.startswith("VIDEO")]

# physiological activity
EDA_COLS = [col for col in processed_recola_df.columns if col.startswith("EDA")]
ECG_COLS = [col for col in processed_recola_df.columns if col.startswith("ECG")]

# emotional state indicators
TARGET_COLS = ["median_arousal", "median_valence"]

def reduce_dimensions(df: pd.DataFrame, features_to_reduce:list) -> pd.DataFrame:
    '''PCA on each mode (across all participants) retaining at least 95% variance.'''
    pca = PCA(n_components=0.95, svd_solver="full")

    pca_transformed = pca.fit_transform(
        df[features_to_reduce]
    )

    print(f"Dimensions reduced from {len(features_to_reduce)} to {pca.n_components_}.")

    pca_transformed_df = pd.DataFrame(
        pca_transformed,
        columns=[f"PC{n+1}" for n in range(pca.n_components_)]
    )

    # append the TARGET_COLS and "Participant" back into the DataFrame
    return pd.concat([
        df[TARGET_COLS + ["Participant"]].reset_index(drop=True),
        pca_transformed_df.reset_index(drop=True)
    ], axis=1)

subsets = {
    "visual": VISUAL_COLS,
    "audio": AUDIO_COLS,
    "physio": EDA_COLS + ECG_COLS
}

for subset_name, cols in subsets.items():
    print(f"Processing {subset_name}...")
    subset = processed_recola_df[["Participant"] + cols + TARGET_COLS]
    subset_pca_df = reduce_dimensions(subset, cols)
    subset_pca_df.to_csv(f"./data/subset_{subset_name}_pca.csv", index=False)
