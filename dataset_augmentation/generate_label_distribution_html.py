import pandas as pd

# === CONFIG ===
original_excel = "./labels/phenotype_labels.xlsx"
augmented_excel = "./labels/augmented_images_phenotype_labels.xlsx"
sheet_original = "birch_labels_final"
sheet_augmented = "augmented_images_labels"
output_html = "./dataset_augmentation/label_stats_report.html"

# === Load Excel files ===
df_orig = pd.read_excel(original_excel, sheet_name=sheet_original)
df_aug = pd.read_excel(augmented_excel, sheet_name=sheet_augmented)
label_cols = [col for col in df_orig.columns if col != "filename"]

# === Comparison Summary ===
def get_summary(df):
    combo_counts = df[label_cols].value_counts()
    return {
        "total_rows": len(df),
        "unique_files": df["filename"].nunique(),
        "unique_combos": df[label_cols].drop_duplicates().shape[0],
        "most_common_count": combo_counts.max(),
        "least_common_count": combo_counts.min(),
        "single_occurrence_combos": (combo_counts == 1).sum(),
        "imbalance_ratio": round(combo_counts.max() / combo_counts.min(), 2) if combo_counts.min() > 0 else "∞"
    }

stats_orig = get_summary(df_orig)
stats_aug = get_summary(df_aug)

# === HTML Layout with escaped curly braces ===
html = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>Label Statistics Report</title>
  <style>
    body {{ font-family: sans-serif; padding: 2em; line-height: 1.6; }}
    h2 {{ margin-top: 2rem; }}
    ul {{ list-style: none; padding-left: 0; }}
    code {{ color: #444; }}
    pre {{ background: #f4f4f4; padding: 1rem; overflow-x: auto; border-radius: 5px; }}
    table {{ border-collapse: collapse; margin-top: 2rem; }}
    th, td {{ border: 1px solid #ccc; padding: 0.5em 1em; text-align: center; }}
    th {{ background-color: #eee; }}
  </style>
</head>
<body>
  <h1>Label Statistics Report</h1>
  <h2>Dataset Comparison Summary</h2>
  <table>
    <tr><th>Metric</th><th>Original</th><th>Augmented</th></tr>
    <tr><td>Total Rows</td><td>{total_rows_orig}</td><td>{total_rows_aug}</td></tr>
    <tr><td>Unique Filenames</td><td>{unique_files_orig}</td><td>{unique_files_aug}</td></tr>
    <tr><td>Unique Label Combos</td><td>{unique_combos_orig}</td><td>{unique_combos_aug}</td></tr>
    <tr><td>Most Common Combo Count</td><td>{most_common_count_orig}</td><td>{most_common_count_aug}</td></tr>
    <tr><td>Least Common Combo Count</td><td>{least_common_count_orig}</td><td>{least_common_count_aug}</td></tr>
    <tr><td>Single-Occurrence Combos</td><td>{single_occurrence_orig}</td><td>{single_occurrence_aug}</td></tr>
    <tr><td>Imbalance Ratio</td><td>{imbalance_ratio_orig}</td><td>{imbalance_ratio_aug}</td></tr>
  </table>
""".format(
    total_rows_orig=stats_orig["total_rows"],
    total_rows_aug=stats_aug["total_rows"],
    unique_files_orig=stats_orig["unique_files"],
    unique_files_aug=stats_aug["unique_files"],
    unique_combos_orig=stats_orig["unique_combos"],
    unique_combos_aug=stats_aug["unique_combos"],
    most_common_count_orig=stats_orig["most_common_count"],
    most_common_count_aug=stats_aug["most_common_count"],
    least_common_count_orig=stats_orig["least_common_count"],
    least_common_count_aug=stats_aug["least_common_count"],
    single_occurrence_orig=stats_orig["single_occurrence_combos"],
    single_occurrence_aug=stats_aug["single_occurrence_combos"],
    imbalance_ratio_orig=stats_orig["imbalance_ratio"],
    imbalance_ratio_aug=stats_aug["imbalance_ratio"]
)

# === Utility to Add Dataset Sections ===
def add_dataset_section(df, label):
    html = f"<h2>{label}</h2>"
    combo_counts = df[label_cols].value_counts()

    html += "<h4>Top 10 Most Common Label Combos</h4><pre>"
    html += combo_counts.head(10).to_string()
    html += "</pre>"

    html += "<h4>Bottom 10 Rarest Label Combos</h4><pre>"
    html += combo_counts.tail(10).to_string()
    html += "</pre>"

    html += "<h4>Rarest Label Values (per column)</h4>"
    for col in label_cols:
        rare_vals = df[col].value_counts().sort_values().head(3)
        html += f"<strong>{col}</strong><ul>"
        for val, count in rare_vals.items():
            html += f"<li> Value {val} is used: {count} times </li>"
        html += "</ul>"

    html += "<h4>Top Label Value Distributions</h4>"
    for col in label_cols:
        freqs = df[col].value_counts(normalize=True).head(30)
        html += f"<strong>{col}</strong><ul>"
        for val, pct in freqs.items():
            count = df[col].value_counts()[val]
            bar = '█' * int(pct * 40)
            html += f"<li> Value {val} is used: {count} times ({pct:.2%}) <code>{bar}</code></li>"
        html += "</ul>"
    return html

# === Add Both Datasets ===
html += add_dataset_section(df_orig, "Original Dataset")
html += add_dataset_section(df_aug, "Augmented Dataset")

html += "</body></html>"

# === Write file ===
with open(output_html, "w", encoding="utf-8") as f:
    f.write(html)

print(f"Report saved to {output_html}")
