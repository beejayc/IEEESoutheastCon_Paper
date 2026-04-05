# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research project evaluating few-shot learning with Vision Language Models for bearing defect detection using spectrograms. The paper is published at IEEE SoutheastCon 2025 (Paper ID: 1571115979).

**Key Features:**
- Supports both Anthropic Claude and Google Gemini models for vision-based classification
- Implements few-shot learning (1-3 reference examples) for bearing defect classification
- Classifies 4 defect types: ACCEPT, OUTER DENT, NEEDLE REJECT, RACEWAY DENT
- Evaluates model performance using confusion matrix with macro precision/recall/F1 metrics

## Project Structure

```
src/
├── mrun_simulation.py          # Main entry point - orchestrates the simulation
├── mrun_antropic_mdls.py       # Anthropic Claude integration
├── mrun_gemini_mdls.py         # Google Gemini integration
├── mutil_prompt_engr.py        # Prompt templates for spectrogram analysis
└── mutil_evalmetric.py         # Evaluation metrics calculation
requirements.txt                # Python dependencies
```

## Architecture & Key Components

### Simulation Flow (mrun_simulation.py)
1. Loads reference spectrogram images (one per defect class for 1-shot learning)
2. For each defect class, loads test samples from the dataset
3. Randomly samples 100 test images per class
4. For each test image:
   - Compiles few-shot prompt with reference + test image
   - Calls selected model (Claude or Gemini)
   - Parses response to extract predicted defect class
   - Matches predicted class against true class using regex
   - Updates confusion matrix
5. Calculates macro precision, recall, and F1 score

### Model Integration
- **Anthropic (mrun_antropic_mdls.py):**
  - Uses `Anthropic` client to call Claude models
  - Supports Claude 3 Opus, Claude 3.5 Sonnet
  - Encodes images as base64 for the API
  - Handles message formatting for vision API

- **Gemini (mrun_gemini_mdls.py):**
  - Uses Google's `generativeai` library
  - Supports Gemini 2.0 Flash, Gemini 1.5 Pro/Flash
  - Includes caching infrastructure (partially implemented, commented out)
  - Similar image handling and message compilation

### Prompt Engineering (mutil_prompt_engr.py)
- `fewshot_setobj`: System prompt for learning reference examples (instructs model to extract spectral features: peak frequencies, amplitudes, harmonics, etc.; limits analysis to <6000 Hz)
- `run_inference_sample`: Inference prompt for test image classification

### Evaluation (mutil_evalmetric.py)
- Calculates per-class precision, recall, and F1 score
- Computes macro averages across all classes
- Handles edge cases (division by zero)

## Setup & Dependencies

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **API Keys:**
   - Edit `mrun_antropic_mdls.py` line 22: Add Anthropic API key to `api_key=""`
   - Edit `mrun_gemini_mdls.py` line 27: Add Google API key to `api_key=""`

3. **Dataset:**
   - Update `BASEDIR` in `mrun_simulation.py` line 59 to point to your test dataset root folder
   - Update `ref_filepath` array (lines 44-49) to point to reference images for each defect class
   - Expected structure: `{BASEDIR}/{DEFECT_FOLDER}/image.jpg`

## Running the Simulation

```bash
cd src/
python mrun_simulation.py
```

**Key configuration variables in mrun_simulation.py:**
- `SIM_BASE`: Switch between "ANTHROPIC" or "GEMINI"
- `MODEL_NAME`: Specific model (e.g., "claude-3-opus-20240229" or "gemini-2.0-flash")
- `NUM_SHOTS`: Number of reference examples (1-3)
- `NUM_SAMPLES`: Test images per class (default: 100)
- `random_seed`: Reproducibility for sample shuffling

## Important Notes

### Hardcoded Dataset Paths
The codebase uses hardcoded absolute paths (`/home/balajic/Projects/...`) for the reference images and test datasets. These need to be updated in:
- `ref_filepath` array (lines 44-49 in mrun_simulation.py)
- `BASEDIR` (line 59 in mrun_simulation.py)

### Model Selection
- Switch between Anthropic and Gemini by changing `SIM_BASE` variable
- Prompt caching is partially implemented in Gemini integration (commented out in mrun_gemini_mdls.py)

### Defect Classification Logic
Classification is done via regex pattern matching on the model response. The model is expected to include one of the defect class names exactly in its output:
- Pattern: `\b{CLASS_NAME}\b` (word boundary match)
- If no direct match, checks all classes for partial matches

### Image Processing
Both integrations resize images to 400x400 pixels using PIL's LANCZOS resampling and encode as base64 for API transmission.

## Testing & Debugging

To debug prompt compilation without running inference:
```python
python mrun_antropic_mdls.py  # Writes "fewshot_msgs.txt"
python mrun_gemini_mdls.py    # Writes "inpmsg.txt"
```

These output files show the exact message structure sent to the APIs.