# Evaluation of Few-Shot Learning with Vision Language Models for Needle Bearing Defect Detection

[![IEEE SoutheastCon 2025](https://img.shields.io/badge/IEEE%20SoutheastCon-2025-blue)](https://ieeexplore.ieee.org/)
[![Paper ID: 1571115979](https://img.shields.io/badge/Paper%20ID-1571115979-green)](./IEEE_PaperID1571115979.pdf)

## 🎯 Overview

This research project demonstrates that **Vision Language Models (VLMs) can effectively detect needle bearing defects from vibration signal spectrograms using few-shot learning**, achieving performance that surpasses human experts and approaches state-of-the-art deep learning methods.

**Key Innovation:** Using large multimodal AI models (Claude, GPT-4o, Gemini) with minimal training examples (1-2 reference images) to classify bearing defects—eliminating the need for extensive labeled datasets and complex model training.

---

## 📊 Key Results

| Metric | Result | Significance |
|--------|--------|--------------|
| **Best VLM (GPT-4o)** | 95% F1 Score (K=2) | Matches SOTA approaches |
| **vs Human Experts** | +13% improvement | Surpasses manual classification |
| **vs ViT Fine-tuning** | 95% vs 98% F1 | 97% of SOTA performance |
| **Training Examples** | 1-2 per class | Drastically reduced data requirements |
| **Inference Speed** | <1 second | Real-time deployment capability |

### Model Performance Comparison (K=2 Few-Shot)

```
GPT-4o              ████████████████████ 95% F1
Claude-3-5-Sonnet   ███████████████████ 94% F1
Gemini-1.5-Pro      ███████████ 45% F1
ViT Fine-tuning     ██████████████████████ 98% F1
Human Experts       ███████████████ 82% F1
```

---

## 🔧 Project Structure

```
.
├── src/
│   ├── mrun_simulation.py              # Main orchestration script
│   ├── mrun_antropic_mdls.py           # Anthropic Claude integration
│   ├── mrun_gemini_mdls.py             # Google Gemini integration
│   ├── mutil_prompt_engr.py            # Prompt templates for analysis
│   └── mutil_evalmetric.py             # Evaluation metrics
├── IEEE_PaperID1571115979.pdf          # Full research paper
├── IEEE_SouthEastCon2025_...Slides.pdf # Conference presentation
├── IEEE_SoutheastCon_2025_Brochure.pdf # Conference brochure
├── requirements.txt                     # Python dependencies
└── README.md                           # This file
```

---

## 💡 Research Highlights

### Problem Statement
Needle bearing defect detection is critical for industrial maintenance, but:
- ❌ Traditional deep learning requires extensive labeled data
- ❌ Signal processing approaches need expert tuning
- ❌ Deployment complexity limits real-world adoption
- ✅ Few-shot learning with VLMs offers a practical alternative

### Innovation: Visual Spectrogram Classification (VSC)
The research bridges **vibration signal analysis** with **vision-based AI**:

1. **Convert** time-series vibration data → spectrograms (visual representation)
2. **Leverage** VLM vision capabilities for image understanding
3. **Use** few-shot prompting with reference examples (1-2 samples)
4. **Classify** bearing defects in seconds with minimal training

### Dataset: Real-World Needle Bearing Data

| Defect Class | Samples | Description |
|--------------|---------|-------------|
| **ACCEPT** (No Defect) | 2,800 | Healthy bearings |
| **DENTS** | 240 | Surface indentations on raceway |
| **NEEDLE REJECTS (NR)** | 237 | Damaged/missing rollers |
| **RACEWAY DENTS (RD)** | 252 | Indentations in inner/outer race |

**Data Collection:**
- ICP accelerometer (PCB Piezotronics M353B15)
- Sampling rate: 24 kHz
- Load: 50N at 900 RPM
- Duration: 2-month production environment study
- Ground truth: Human expert validation

---

## 🧪 Experimental Methodology

### Ablation Study: Spectrogram Optimization
Tested multiple configurations to identify optimal visual encoding:

**Best Configuration Found:**
- **Method:** Multitaper (MT) spectral estimation
- **Amplitude:** Linear scaling
- **Colormap:** Viridis
- **Result:** 0.88 F1 (with GPT-4o)

| Method | Colormap | Amplitude | F1 Score |
|--------|----------|-----------|----------|
| STFT | Viridis | Log | 0.56 |
| STFT | Jet | Linear | **0.82** |
| **MT** | **Viridis** | **Linear** | **0.88** |
| MT | Jet | Linear | 0.86 |

### Few-Shot Learning Strategy

**One-Shot (K=1):** Single reference example per class
- GPT-4o: 90% F1
- Claude-3-5-Sonnet: 86% F1
- Gemini-1.5-Pro: 40% F1

**Two-Shot (K=2):** Reference + extreme sample per class
- GPT-4o: **95% F1** ⭐
- Claude-3-5-Sonnet: **94% F1**
- Gemini-1.5-Pro: 45% F1

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- API keys for: Anthropic, OpenAI, or Google Cloud

### Installation

1. **Clone and setup:**
   ```bash
   cd src/
   pip install -r requirements.txt
   ```

2. **Configure API Keys:**
   - **Anthropic (Claude):** Edit `mrun_antropic_mdls.py` line 22
   - **Google (Gemini):** Edit `mrun_gemini_mdls.py` line 27
   - **OpenAI (GPT-4o):** Update as needed

3. **Prepare Dataset:**
   - Update `BASEDIR` in `mrun_simulation.py` (line 59) with your test dataset path
   - Update `ref_filepath` array (lines 44-49) with reference image paths
   - Expected structure:
     ```
     {BASEDIR}/
     ├── ACCEPT/
     ├── DENTS/
     ├── NR/
     └── RD/
     ```

### Quick Start

```bash
cd src/
python mrun_simulation.py
```

**Configuration Options in `mrun_simulation.py`:**

```python
SIM_BASE     = "ANTHROPIC"          # Switch: "ANTHROPIC" or "GEMINI"
MODEL_NAME   = "claude-3-opus-20240229"  # Model to use
NUM_SHOTS    = 1                    # Few-shot samples (1-3)
NUM_SAMPLES  = 100                  # Test samples per class
random_seed  = 42                   # Reproducibility
```

### Debug Without Inference

Generate prompt structures without API calls:

```bash
python mrun_antropic_mdls.py   # Outputs: fewshot_msgs.txt
python mrun_gemini_mdls.py     # Outputs: inpmsg.txt
```

These files show the exact message structures sent to each API.

---

## 📈 Results & Analysis

### Confusion Matrix - GPT-4o (Best Performer)

```
                 Predicted
                ACPT  DENT  NR   RD
    Actual ACPT  90     10    0    1
           DENT   2     97    1    0
           NR     5      0   94    1
           RD     0      0    0  100
```

**Key Observations:**
- Strong diagonal: 90-100% correct per class
- Minimal misclassification: Cross-class errors <5%
- Raceway Dents: Perfect classification (100% recall/precision)

### Performance Insights

1. **GPT-4o Superiority:** 
   - Superior visual reasoning for spectral patterns
   - Best handling of edge cases
   - Fastest inference time

2. **Claude-3-5-Sonnet:**
   - Nearly matches GPT-4o (94% vs 95%)
   - **Faster response** with prompt caching
   - Cost-effective alternative

3. **Gemini Models:**
   - Underperformed on spectrogram analysis
   - Better with structured data prompts
   - Partial caching implementation available

---

## 🔍 Technical Details

### Spectrogram Generation

#### Short-Time Fourier Transform (STFT)
```python
# Converts time-series to time-frequency representation
# Window: Hamming function
# Overlap: 50% between windows
fs = 24000 Hz
window_length = 256 samples
hop_length = 128 samples
```

#### Multitaper Spectral Estimation
```python
# Superior noise reduction via orthogonal tapers
TW (Time-Bandwidth) = 2.5
Slepian Tapers (L) = 4
# Better spectral resolution with smoother estimates
```

### Prompt Engineering Strategy

**System Prompt:** Instructs model to:
- Extract spectral features (frequencies, amplitudes, harmonics)
- Focus on 0-6 kHz frequency range
- Avoid hallucination; extract strictly from visual data

**Few-Shot Template:**
```
1. System context: "You are an expert spectogram analyzer"
2. Reference catalog: 1-2 labeled spectrograms per class
3. Test sample: New spectrogram to classify
4. Classification prompt: Explicit instructions for output format
```

---

## 📚 Publication & Resources

### Full Research Paper
📄 **[IEEE_PaperID1571115979.pdf](./IEEE_PaperID1571115979.pdf)**
- 8 pages, peer-reviewed IEEE publication
- Detailed methodology and results
- Mathematical formulations
- Complete references

### Conference Presentation
🎤 **[IEEE_SouthEastCon2025_PaperID1571115979_Slides.pdf](./IEEE_SouthEastCon2025_PaperID1571115979_Slides.pdf)**
- 22 slides covering research highlights
- Visual demonstrations
- Experiment setup details
- Key findings summary

### Conference Information
🏢 **[IEEE_SoutheastCon_2025_Brochure.pdf](./IEEE_SoutheastCon_2025_Brochure.pdf)**
- Conference details and schedule
- Additional participating papers

---

## 👥 Authors

| Name | Affiliation | Contact |
|------|------------|---------|
| **Balaji Chandrasekaran** | Schaeffler Group (Data Science Solutions), USA | ORCID: 0009-0006-3379-0717 |
| **Dr. Vamanie Perumal** | Indian Institute of Technology, Madras (IIT-M), India | perumal.vamanie@gmail.com |

---

## 🔬 Methodology Highlights

### Study Design
1. **Baseline Comparison:** ViT fine-tuning model (98% F1)
2. **Human Baseline:** Expert classification (82% F1)
3. **VLM Evaluation:** 6 models across 3 providers
4. **Cross-validation:** 100 samples per class, iterated twice

### Evaluation Metrics

```
Macro-Precision = Average precision across all defect classes
Macro-Recall    = Average recall across all defect classes
Macro F1-Score  = Harmonic mean of precision and recall
```

---

## 🎯 Key Findings

✅ **VLMs Surpass Human Experts**
- Human ensemble: 82% F1
- Best VLM (GPT-4o): 95% F1
- Improvement: +13 percentage points

✅ **Approaches SOTA Performance**
- ViT Fine-tuning: 98% F1 (requires extensive training)
- VLM Few-Shot: 95% F1 (requires 2 samples)
- 97% of SOTA with 99% less data

✅ **Practical Advantages**
- No model retraining needed
- Sub-second inference
- Minimal training data
- Direct deployment capability

✅ **Generalization Potential**
- Works across different spectrogram methods (STFT, Multitaper)
- Adaptable to other industrial applications
- Scalable to additional defect types

---

## ⚠️ Limitations & Future Work

### Current Limitations
- **Closed-source VLMs only:** Open-source alternatives not evaluated
- **Token constraints:** Limits spectrogram resolution (impacts fine-grained defect detection)
- **Sample catalog size:** Restricted to K=2 due to API cache timeouts
- **Single bearing type:** Focus on needle bearings exclusively

### Future Research Directions
1. **Larger datasets:** More diverse bearing types and defect categories
2. **Open-source VLMs:** Evaluate local/self-hosted alternatives
3. **Fine-tuning exploration:** Investigate improved few-shot with model tuning
4. **Multi-class expansion:** Additional industrial equipment diagnostics
5. **Token optimization:** Advanced compression/cropping techniques for higher resolution

---

## 📖 References

The research builds upon:
- **Foundation Models:** GPT-4o, Claude-3, Gemini-1.5 family
- **Vision Transformers:** ViT-based approaches for bearing classification
- **Signal Processing:** STFT and Multitaper spectral analysis
- **Multimodal AI:** LLaVA, CLIP-based vision-language architectures
- **Industrial Diagnostics:** CWRU bearing dataset and traditional approaches

See full references in the [research paper](./IEEE_PaperID1571115979.pdf).

---

## 📞 Contact & Support

For questions about this research:
- 📧 perumal.vamanie@gmail.com
- 🏢 IIT Madras Department of Engineering Design
- 🏭 Schaeffler Group Data Science Solutions

---

## 📄 Citation

```bibtex
@inproceedings{chandrasekaran2025few-shot,
  title={Evaluation of Few-Shot Learning with Vision Language Models 
         for Needle Bearing Defect Detection},
  author={Chandrasekaran, Balaji and Perumal, Vamanie},
  booktitle={IEEE SoutheastCon 2025},
  year={2025},
  pages={1300--1308},
  doi={10.1109/SoutheastCon57799.2025}
}
```

---

## 📄 License

This research is published in IEEE SoutheastCon 2025 proceedings.
Code and datasets follow the accompanying license agreements.

---

**Last Updated:** April 2025  
**Status:** Published at IEEE SoutheastCon 2025
