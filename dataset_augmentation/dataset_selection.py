""" select images for augmentation based on rare combination of labels and rare label volumes"""
# augmentation/dataset_selection.py
import numpy as np
import pandas as pd

def determine_smart_threshold(combination_counts):
    """
    Dynamically determines a threshold for rare label combinations based on dataset distribution
    The threshold is calculated as: threshold = median - std_dev
    """
    median = np.median(combination_counts["count"].to_numpy())
    std_dev = np.std(combination_counts["count"].to_numpy())
    #return int(median - std_dev)
    return 1 # calculation works fine but using return value of threshold 1 so if the combo is only once or less then considered rare this is only for testing with a small dataset locally

def find_rare_label_values(df, label_columns):
    """
    Dynamically finds rare labels based on dataset size and label frequency
    The rarity threshold is defined as: threshold = 1 / sqrt(dataset size)
    """
    threshold = 1 / np.sqrt(len(df))
    #NOT WORKING AS SHOULD; STARTING OVER WITH THIS ONE
    pass

def select_images_for_augmentation(df):
    """
    Selects images with rare label combinations or rare individual labels, drops duplicates
    """
    label_columns = df.columns[1:]
    combination_counts = df[label_columns].value_counts().reset_index(name="count")
    threshold = determine_smart_threshold(combination_counts)
    rare_combinations = combination_counts[combination_counts["count"] <= threshold].drop(columns=["count"])

    #rare_label_columns = find_rare_labels(df, label_columns)
    #images_with_rare_labels = df[df[rare_label_columns].any(axis=1)]

    rare_df = pd.merge(df, rare_combinations, on=list(label_columns))
    return rare_df.drop_duplicates()