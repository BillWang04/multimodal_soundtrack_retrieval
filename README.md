
# Soundtrack Recommendation for Film Productions: A Multimodal Approach

This supplementary material provides the code for training and evaluating the seven models discussed in our paper: M-Im, M-Ta, M-Plo, M-ImTa, M-ImPlo, M-TaPlo, and M-All. Due to copyright issues, we are unable to provide the original audio and images. Instead, we have the metadata from Wikipedia, without the image and video which can be scraped through YouTube.

An image, tags, and a plot summary for a film will be substituted with a blank image, `['tag1', 'tag2', 'tag3']`, and the phrase "this is a plot", respectively. Additionally, soundtracks will be replaced with white noise. The Python code to construct data based on a Wikipedia URL, including the list of Wikipedia URLs used in our experiments for reproducibility, will be provided upon acceptance. We also commit to releasing this supplementary material publicly upon acceptance to ensure reproducibility.

## 1. Create a Virtual Environment and Install Required Packages

```bash
conda create -n test python=3.8
pip install -r requirements.txt
```

## 2. Data Structure and Setup

Since we cannot distribute the original audio and image files from our dataset, we provide `./data/reference/meta.json` which contains the necessary metadata to scrape all audio and images from YouTube and Wikipedia.

### Directory Structure

The dataset uses a three-letter naming convention where each movie is assigned a unique three-letter identifier. Audio files for each movie are stored in nested directories that match this identifier. For example, audio files for movie "AAA" would be located in `./data/audio/A/A/A/`.

### File Descriptions

- **`eval.json`** - Evaluation dataset split
- **`train.json`** - Training dataset split  
- **`valid.json`** - Validation dataset split
- **`fid_to_text.json`** - Maps file IDs to their corresponding text descriptions
- **`meta.json`** - Complete overview of all movies and their associated metadata tags
- **`movie_all_audio_paths.pkl`** - Pickled list containing all audio file paths in the dataset

### Text Annotations

Text tags for each audio file are extracted from Wikipedia. The format and structure of these annotations can be found in `./data/meta/fid_to_text.txt`. The reference `meta.json` file provides a comprehensive mapping between movies and their attributed tags (note: this differs from the dataset's `./data/meta/meta.json`).

### Data Acquisition

Use the provided `./data/reference/meta.json` as your guide for scraping the complete dataset from the original YouTube and Wikipedia sources.

## 3. Train the Seven Models

```bash
cd train
python -m main --experiment_models Im
python -m main --experiment_models Ta
python -m main --experiment_models Plo
python -m main --experiment_models ImTa
python -m main --experiment_models ImPlo
python -m main --experiment_models TaPlo
python -m main --experiment_models All
```


## 4. Evaluate the Seven Models

### Model Checkpoints

Pre-trained model checkpoints are available on Hugging Face: 
**[https://huggingface.co/wangtonSoup/multimodal-soundtrack-retrieval](https://huggingface.co/wangtonSoup/multimodal-soundtrack-retrieval)**

# Install dependencies
pip install -r requirements.txt

Navigate to the evaluation directory:

```bash
python -m evaluation.extract
python -m evaluation.evaluate
```