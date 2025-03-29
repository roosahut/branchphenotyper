# Dataset Augmentation Pipeline

The dataset_augmentation folder contains functions to create and save new images from original images by python based image augmentation to increase the dataset size. It also contains functions to read, analyse and write excel sheets with image labelling information. Based on image labeling distribution, data augmentation pipelines in this folder can create more photos from rare image label types and label combinations to balance the dataset.

There are two primary modes that can be applied to either individual images or several images at a time:

* **Individual Augmentations:** Each image is processed with each individual transformation (e.g., rotate or flip or greyscaling to one photo). This is more for testing and exploration purposes.
* **Combined Augmentations:** Each image is processed with a random chain of transformations (e.g., rotate and flip and greyscaling all done to one photo)

Rare images (based on dataset selection) can be repeated in this process, leading to bigger amount of those types of images created.

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

Use the main.py file to choose which augmentation pipeline you want to use and with what paremeters, but overall they all have this structure:

1. Read excel: read the image names (file names) and labels from excel and put to pandas dataframe
2. Dataset selection: choose the file nameswith the  rarest types of labels and rare combinations of labels based on label information in excel
3. Image input: get the chosen (rare) or all of the original images from an assigned folder
4. Image augmentation: based on pipeline chosen, apply either:
   1. One type of augmentation to one or several images and save each image that has one type of augmentation
   2. Randomly assign 2-n amount of augmentations to one or several images and save each images that has multiple augmentations
5. New excel: create or update a new excel that contains the new names of augmented images and the original images' labels

## Usage

See the main.py file in this folder for running the augmentations. Also there are helper functions to timing the execution time and for deleting augmented images locally for cleanup. 

Add all folder locations etc to config.py file
- you should be able to run all the pipelines with the test images and excels that are here in this repository but for processing larger amounts of images, your laptop might not be able to handle it and doing that in a cloud based processing system might be easier.

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
```
You also need python (should work at least with 3.8.10->)

### Unfinished things

for local teting with a small number of images, threshold to selecting rare label combinations is hardcoded as 1 so it is considered rare if that combo of labels is present once or less, but the actual calulation works fine with a larger dataset, but running that is hard locally. So need to remeber to change this back at some point.

counting the rare label values is unfinished, it was not working well enough so starting over with that one. 

Do we want one unified excel with both augmented image labels and original labels? This is easy to do if decided so.

Some combinations are too harsh, so adjusting parameters in augmentation functions needs fine tuning. Dont add channel shuffle and rgb in one? etc

If trying to name two files the same name -> handle this error withouhg crashing