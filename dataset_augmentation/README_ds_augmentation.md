# Dataset Augmentation Pipeline

The system dynamically augments images based on label combinations. It increases dataset size through baseline augmentation and balances label distribution by identifying rare label combinations using configurable rarity strategies (`fixed`, `percentile`, or `max`). Rare label types receive additional augmentations to create a balanced dataset.

There are two ways of augmenting an image:

* **Individual Augmentations:** Each image is processed with each individual transformation (e.g., rotate or flip or greyscaling to one photo). This is more for testing and exploration purposes.
* **Combined Augmentations:** Each image is processed with a random chain of transformations (e.g., rotate and flip and greyscaling all done to one photo)

Rare images (based on dataset selection) are  repeated in this process, leading to bigger amount of those types of images created.

## File Structure

<pre class="!overflow-visible" data-start="568" data-end="1126"><div class="contain-inline-size rounded-md border-[0.5px] border-token-border-medium relative bg-token-sidebar-surface-primary"><div class="flex items-center text-token-text-secondary px-4 py-2 text-xs font-sans justify-between h-9 bg-token-sidebar-surface-primary dark:bg-token-main-surface-secondary select-none rounded-t-[5px]"></div><div class="sticky top-9"><div class="absolute bottom-0 right-0 flex h-9 items-center pr-2"><div class="flex items-center rounded bg-token-sidebar-surface-primary px-2 font-sans text-xs text-token-text-secondary dark:bg-token-main-surface-secondary"><span class="" data-state="closed"></span></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="!whitespace-pre"><span><span>project/
├── config.py                      </span><span># Contains configuration variables (paths, sheet names, etc.)</span><span>
├── file_handling.py               </span><span># Functions for getting images and both original and augmented excel handling</span><span>
├── dataset_selection.py           </span><span># Functions to select images based on rarity, allowing extra augmentation rare label images</span><span>
├── transformations.py             </span><span># Augmentation functions (flip, rotate, etc.)</span><span>
├── augmentation_pipeline.py      </span><span># Pipeline functions for applying augmentations to one or multiple images</span><span>
└── main.py                        </span><span># Entry point to run the augmentation pipelines</span><span>
</span></span></code></div></div></pre>

## Data Flow

Use the main.py file to choose which augmentation pipeline you want to use and with what paremeters. There are multiple entry points to different augmentation pipelines for testing and exploration purposes. But the main augmentation pipeline `run_combined_augmentation_with_balancing` used to balance and increase dataset size has this structure:

PHASE 1 — Baseline Dataset Expansion:

- Reads original excel label file and filters to existing images
- Applies a configurable number of augmentations to every image
- Saves newly created images with a new name and saves their label information to a new augmentation excel
- Logs stats like top label values, unique label combos, etc.

PHASE 2 — Label Balancing:

- Reloads augmentation excel to get post-baseline label distributions
- Determines underrepresented label combinations using one of:
  - 'fixed': user-defined target value (e.g., 5), meaning each label combination must be generated to be present at least 5 times. Good for testing with a small dataset size
  - 'percentile': bottom X% (e.g., 0.9 = bottom 10%) USE THIS MOSTLY, others for testing
  - 'max': matches the most frequent combo THIS WILL TAKE A LONG TIME, and possible overfitting
- Distributes needed augmentations for rare combinations across original images
- Saves augmented images and updates augmented excel

In the main.py file there are helper functions to timing the execution time and for deleting augmented images locally for cleanup.

Add all folder locations etc to `config.py` file

### Example Configuration (`config.py`)

```python
image_directory = "./images/original_images"  		# Path to original input images
output_directory = "./images/augmented_images"  	# Output path for augmented images

excel_file_path = "./labels/phenotype_labels.xlsx"  	# Path to original labels Excel
augmented_excel_file_path = "./labels/augmented_images_phenotype_labels.xlsx"  # Path to augmented labels Excel
sheet_name = "birch_labels_final"  			# Sheet containing label data
augmented_sheet_name = "augmented_images_labels"	# Sheet containing augmented image label data

num_augmented_images = 2              # Every image gets 2 new versions (baseline)
target_count_strategy = "fixed"       # Balancing method
target_value = 5                      # Make every label combo have at least 5 total images
max_extra_aug_per_image = 4           # Cap per original image
```

- you should be able to run all the pipelines with the test images and excels that are here in this repository but for processing larger amounts of images, your laptop might not be able to handle it and doing that in a cloud based processing system might be easier.

## Recommendations

* For small datasets: use `fixed` strategy with `target_value = 5`
* For medium datasets (~600): try `percentile` with `target_value = 0.8`
* For production-ready balance: use `max` to fully equalize all label combinations

### **Output Explanation:**

The system creates:

* Augmented images → saved to the output folder
* Augmented label records → appended to `augmented_images_phenotype_labels.xlsx`
* Log summaries → printed and saved to `label_analysis_log.txt`

Each step logs label distributions (before and after), counts, and dataset growth metrics.

## Requirements

install requirements file

```
pip install -r requirements.txt
```

that should install these dependencies:

```
pillow
pandas
openpyxl
opencv-python
pylint
black
```

You also need python (should work at least with 3.8.10->)

### Unfinished things

Do we want one unified excel with both augmented image labels and original labels? This is easy to do if decided so.

Some combinations are too harsh, so adjusting parameters in augmentation functions needs fine tuning. Dont add channel shuffle and rgb in one? etc
