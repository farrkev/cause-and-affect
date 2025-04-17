import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from itertools import product
import networkx as nx
import pandas as pd
from sklearn.decomposition import PCA
from causallearn.search.ScoreBased.GES import ges
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ConstraintBased.FCI import fci


SOURCE_DATA_PATH = "./data/recola_all_participants_clean_preprocessed.csv"
DAG_PATH = "./results_dag"


def generate_dag(
    search_method,
    modality: str,
    participant_id: int,
    participant_data: pd.DataFrame
):
    """
    Generate a DAG for a specific participant using the provided search method.

    Args:
        search_method (callable): The method used to generate the DAG.
        modality (str): The data modality ('audio', 'visual', 'physio').
        participant_id (int): The ID of the participant.
        participant_data (pd.DataFrame): Data for the participant.

    Returns:
        tuple: Participant ID and the generated DAG 
        (or None if an error occurs).
    """
    try:
        sm_name = search_method.__name__
        print(f"Generating DAG: {modality}, {sm_name}, {participant_id}")

        # defining the output path for the DAG
        output_path = os.path.join(
            DAG_PATH,
            modality,
            f"pruned_{sm_name}_dag_participant_{participant_id}.pkl"
        )

        # removing participant id
        participant_data = participant_data.drop(
            columns=["Participant"],
            errors="ignore"
        ).copy()

        graph = search_method(participant_data.to_numpy())

        graph_matrix = (
            graph["G"].graph if sm_name == "ges"
            else graph[0].graph if sm_name == "fci"
            else graph.G.graph if sm_name == "pc"
            else None
        )

        # extract edges from the learned DAG
        num_nodes = graph_matrix.shape[0]
        edges = []

        # iterate through all node pairs to determine edge types
        for i, j in product(range(num_nodes), range(num_nodes)):
            if graph_matrix[i, j] == -1 and graph_matrix[j, i] == 1:  # fully directed edge i --> j
                edges.append((str(i), str(j)))

        # ensure graph is dag
        while not nx.is_directed_acyclic_graph(nx.DiGraph(edges)):
            cycle = nx.find_cycle(nx.DiGraph(edges))
            edges.remove(cycle[-1])
            graph_matrix[int(cycle[-1][0]), int(cycle[-1][1])] = 0

        # update graph with pruned edges
        if sm_name == "ges":
            graph["G"].graph = graph_matrix
        elif sm_name == "fci":
            graph[0].graph = graph_matrix
        elif sm_name == "pc":
            graph.G.graph = graph_matrix

        # saving the DAG
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump(graph, f)

        print(f"Finished DAG: {modality}, {sm_name}, {participant_id}")

        return participant_id, graph

    except Exception as e:
        print(f"Error for {modality}, {sm_name}, {participant_id}: {e}")
        return participant_id, None

def load_and_reduce(
    path: str,
    pca_components: float,
    scope: list = None
):
    """
    Loads data, reduces dimensions using PCA, and returns dataframes for
    selected modalities.

    Args:
        path (str): Path to the input CSV file.
        pca_components (float): Explained variance for PCA, must be between
                                0 and 1.
        scope (list, optional): List containing any combination of 'visual',
                               'audio', 'physio'. Defaults to all.

    Returns:
        list: List of tuples containing a dataframe for each selected
        modality with PCA applied.
    """
    assert 0 < pca_components < 1, "`pca_components` must be between 0 and 1"

    preprocessed_recola_df = pd.read_csv(path)

    targets = ["median_arousal", "median_valence"]
    visual_features = []
    audio_features = []
    physio_features = []

    # mapping feature prefixes to their respective lists
    feature_groups = {
        "VIDEO": visual_features,
        "ComParE": audio_features,
        "EDA": physio_features,
        "ECG": physio_features
    }

    for col in preprocessed_recola_df.columns:
        for prefix, feature_list in feature_groups.items():
            if col.startswith(prefix):
                feature_list.append(col)
                break

    # mapping scope values to features
    scope_to_features = {
        "visual": visual_features,
        "audio": audio_features,
        "physio": physio_features
    }

    if scope is None:
        scope = list(scope_to_features.keys())

    results = []

    # performing PCA on selected modalities
    for modality in scope:
        if modality not in scope_to_features:
            raise ValueError(f"Invalid modality: {modality}")

        features = scope_to_features[modality]
        if not features:
            continue

        subset = preprocessed_recola_df[["Participant"] + features + targets]

        # drop targets and Participant for PCA
        feature_data = subset.drop(columns=targets + ["Participant"])

        pca = PCA(n_components=pca_components)
        pca_transformed = pca.fit_transform(feature_data)
        print(f"Reduction from {subset.shape[1] - 3} to {pca.n_components_}.")

        pca_df = pd.DataFrame(
            pca_transformed,
            columns=[f"PC{i+1}" for i in range(pca.n_components_)]
        )

        # combining PCA results with targets and Participant
        final_df = pd.concat(
            [subset[targets + ["Participant"]].reset_index(drop=True),
             pca_df.reset_index(drop=True)],
            axis=1
        )

        results.append((modality, final_df))

    return results


def main():
    """
    Main function to generate causal graphs based on input parameters.
    Usage: causal_graphs.py <optional: search_method/s> <optional: modality/ies>
    """
    # argument validation and parsing
    if len(sys.argv) > 3:
        print("Usage: causal_graphs.py <optional: search_method/s> <optional: modality/ies>")
        return

    search_methods = sys.argv[1].split(",") if len(sys.argv) > 1 else None
    scope = sys.argv[2].split(",") if len(sys.argv) > 2 else None

    # mapping of search methods to their respective functions
    search_methods_funcs = {
        "ges": ges,
        "pc": pc,
        "fci": fci,
    }

    # default to all search methods if none are specified
    if search_methods is None:
        search_methods = list(search_methods_funcs.keys())

    # load data and apply PCA
    reduced_data = load_and_reduce(SOURCE_DATA_PATH, 0.95, scope)

    # execute tasks concurrently
    tasks = [
        (search_methods_funcs[method], modality, p_id, group.copy())
        for method in search_methods
        for modality, result_df in reduced_data
        for p_id, group in result_df.groupby("Participant")
        # if method in ["pc", "fci"]
    ]

    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(generate_dag, method_func, modality, name, group)
            for method_func, modality, name, group in tasks
        ]

        # ensure all tasks complete
        for future in futures:
            future.result()


if __name__ == "__main__":
    main()
