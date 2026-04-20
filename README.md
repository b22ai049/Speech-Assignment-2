# Speech Assignment 2: Robust Code-Switched Transcription & Zero-Shot Voice Cloning

This repository contains the complete pipeline for processing code-switched (Hinglish) academic lectures, translating them into Maithili (a low-resource language), and synthesizing the output using zero-shot voice cloning.

## Project Structure

- `src/`: Core Python modules.
  - `denoising.py`: Multi-band spectral subtraction and dereverberation.
  - `lid.py`: Dual-head CNN for frame-level language identification (English vs. Hindi).
  - `ngram_lm.py` & `transcription.py`: Whisper-v3 with N-gram logit biasing for technical terms.
  - `ipa_converter.py`: Custom Hinglish G2P conversion rules.
  - `translation.py` & `parallel_corpus.py`: 500+ word technical parallel corpus and semantic translation to Maithili.
  - `voice_embedding.py`: d-vector/x-vector speaker embedding extraction.
  - `prosody_warping.py`: F0 and energy contour matching via Dynamic Time Warping (FastDTW).
  - `synthesis.py`: VITS/YourTTS synthesis with 22.05kHz formatting.
  - `anti_spoofing.py`: LFCC + GMM based anti-spoofing classifier.
  - `adversarial.py`: FGSM adversarial noise injection.
  - `evaluation.py`: Computes WER, MCD, EER, and LID boundaries.
- `configs/config.yaml`: Centralized configuration settings.
- `data/`: Curated syllabi and Hinglish phonology JSON rules.
- `report/`: IEEE format report (`report.tex`) and implementation note (`implementation_note.tex`).
- `pipeline.py`: Main orchestrator script to run the entire end-to-end pipeline.

## Setup Instructions

1. **Install Dependencies:**
   Ensure you have Python 3.10+ installed. Install the dependencies using the provided environment file or requirements:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: PyTorch and TorchAudio should be installed with CUDA support if a GPU is available.*

2. **Run the Pipeline:**
   Execute the orchestrator script:
   ```bash
   python pipeline.py
   ```
   This will execute all four parts of the assignment sequentially, saving artifacts (transcripts, translated texts, synthesized audios) to the `outputs/` folder.

## Evaluation Metrics Output
Upon completion, the script generates an `evaluation_report.json` in the `outputs/` folder containing the WER, MCD, EER, and LID F1-scores.
