# AI-Powered Early Dyslexia Screening System — Development Stage 1

## 1. Project Overview

This is a research/development prototype for an AI-assisted early dyslexia
screening tool. **At this development stage, the application does not
produce a final dyslexia diagnosis or a final dyslexia-risk score.** It
contains two independent, evaluated ML pipelines:

1. An **English oral-reading / speech module**, built on the MPS dataset.
2. A **handwriting classification module**, built on a real handwriting
   dataset.

There is no behavioral module, no late fusion of the two modalities, and no
AI-generated interpretation or recommendation layer in this version.

## 2. Current Development Scope

Implemented:
- MPS dataset discovery/loading
- Audio preprocessing (mono, 16kHz, validated)
- Wav2Vec 2.0 embedding extraction
- MPS-derived reading-performance feature computation (WCPM, accuracy,
  substitution/deletion/insertion/miscue rates, pause statistics)
- A classical classifier (XGBoost or SVM) trained on a reading-performance
  target that comes directly from the MPS dataset's own annotations
- Handwriting dataset loading (ImageFolder-style layout)
- Handwriting preprocessing (crop/resize/normalize)
- ResNet18 training, evaluation, and inference
- A Streamlit UI to walk through Student Info → Speech Test → Handwriting
  Test → Results, displaying both model outputs and their evaluation
  metrics side by side

**Explicitly NOT implemented at this stage** (by design):
- Behavioral questionnaire, behavioral XGBoost/SVM, behavioral scoring
- A dyslexia label derived from MPS via a threshold on WCPM/accuracy/etc.
- Late fusion of speech + handwriting outputs
- A final dyslexia risk score or "Dyslexia detected" message
- Grad-CAM / SHAP / any AI-generated explanation or recommendation

## 3. Why MPS Is Not Used to Predict Dyslexia

The MPS dataset provides oral-reading performance annotations, not
clinically established dyslexia diagnoses. Building a "dyslexia" label by
thresholding WCPM, accuracy, or miscue rate would not be scientifically
valid and could mislead users of this tool. For that reason, the speech
module here is described and evaluated strictly as a **reading-performance
analysis pipeline**. The prediction target it trains on
(`speech.target_column` in `config.yaml`) must be an outcome that is
explicitly present in the MPS annotations themselves — the code will fail
loudly rather than manufacture one.

## 4. Architecture

```
MPS Dataset -> Audio preprocessing -> Wav2Vec 2.0 -> Reading-performance
features -> Classical classifier -> Evaluation metrics

Handwriting Dataset -> Image preprocessing -> ResNet18 -> Classification
-> Evaluation metrics

Streamlit UI: Home -> Student Info -> Speech Test -> Handwriting Test ->
Results (both outputs shown separately, no combination)
```

## 5. MPS Dataset Setup

1. Download the MPS dataset and place it under `data/mps/` (or point
   `speech.dataset_path` in `config.yaml` at wherever you've stored it).
   Audio files nested arbitrarily deep are found automatically (e.g.
   `data/mps/mps_dataset/audios/*.wav`), and any `.git` directory from a
   cloned repo is skipped.
2. This project's loader (`training/mps_dataset.py`) looks for annotations
   in two ways, preferring the first:
   - **A central `data.json` index** anywhere under the dataset root
     (there can be more than one — e.g. `data/mps/data.json` and
     `data/mps/mps_dataset/data.json` — both are loaded). It accepts
     either a JSON list of record dicts (each naming its audio file via a
     field like `audio_path`/`filename`/`file`/etc.) or a dict keyed
     directly by audio filename.
   - **Per-file annotation matching** (a file sharing the audio's base
     name in the same directory or a sibling `annotations/`/`labels/`/
     `transcripts/` folder) — used as a fallback.
   - As a last resort, student ID / prompt / attempt number are
     heuristically parsed from filenames like
     `3a10a_EN-OL-RC-234_2.wav` — this never fabricates the numeric
     alignment counts needed for feature computation, only fills in
     otherwise-missing metadata.
3. **The real `data.json` schema hasn't been confirmed against this
   code.** The first time you run `extract_reading_features.py`, check the
   logged `Example record keys` line — it prints the actual field names
   found in your `data.json`. Then update `speech.annotation_fields` in
   `config.yaml` to map this pipeline's internal names
   (`total_reference_words`, `correct_words`, `substitutions`,
   `deletions`, `insertions`, `reading_duration_seconds`,
   `hesitation_count`) to whatever your data.json actually calls them —
   no code changes needed for that.
4. Each matched record also needs the field named in
   `speech.target_column` for the prediction target, and the loader
   never invents one — do not threshold WCPM/accuracy into a dyslexia
   label (see Section 3).

## 6. Handwriting Dataset Setup

Place your handwriting dataset under `data/handwriting/`
(`handwriting.dataset_path` in `config.yaml`). Two layouts are supported:

**Flat** (one class per top-level folder):

```
data/handwriting/
    <class_name_1>/
        img001.png
    <class_name_2>/
        img001.png
```

**Nested** — this is the layout you'll use, since your data is organized as
a top-level diagnosis label with numbers/letters/paragraphs sub-folders
underneath:

```
data/handwriting/
    dyslexic/
        numbers/
            img001.png
            ...
        letters/
            img001.png
            ...
        paragraphs/
            img001.png
            ...
    non_dyslexic/
        numbers/
            img001.png
        letters/
            img001.png
        paragraphs/
            img001.png
```

You don't need to change any code — just put your existing `dyslexic/` and
`non_dyslexic/` folders (each already containing `numbers/`, `letters/`,
`paragraphs/`) directly inside `data/handwriting/`. The loader
(`training/handwriting_dataset.py`) walks each top-level folder
recursively, uses the top-level name (`dyslexic` / `non_dyslexic`) as the
training label, and records the sub-folder name (`numbers` / `letters` /
`paragraphs`) separately as `content_type` — it is not used as the
classification target, only kept as metadata.

By default, all three content types are combined into one training set. To
train on just one content type (e.g. only paragraphs), set in
`config.yaml`:

```yaml
handwriting:
  content_type_filter: ["paragraphs"]
```

or `["numbers", "letters"]`, etc. Leave it as `null` to use everything.

Because the labels here (`dyslexic` / `non_dyslexic`) are an actual
clinical designation you're providing — not a threshold invented from
performance metrics — this is a legitimate classification target, unlike
the MPS speech branch (see Section 3).

### 6a. Two things to verify before training

1. **A third `yes/` folder, if present, becomes a third training class
   automatically.** Class names come straight from the top-level folder
   names under `data/handwriting/`, with no allow-list. If you only want
   `dyslexic` / `non_dyslexic`, make sure no other folder sits alongside
   them (rename it out of `data/handwriting/`, merge its contents into one
   of the two classes, or move it elsewhere) before running
   `train_handwriting.py`.
2. **If `dyslexic/` contains the IAM Words dataset** (recognizable by a
   `words.txt` file and a `words/<writer>/<form-id>/*.png` structure),
   note that IAM is a public handwriting-recognition benchmark of general
   adult handwriting with no dyslexia diagnosis attached. If it's being
   used as filler/placeholder data under the `dyslexic` folder rather than
   genuinely dyslexia-diagnosed samples, the model would effectively learn
   to tell "IAM images" apart from "your non_dyslexic photos" (different
   capture conditions, image style, resolution) rather than anything
   related to dyslexia. Worth confirming before training.

`school_symptoms.txt` at the top of `data/handwriting/` is not read by
this pipeline (it's not an image and sits outside any class folder) — it's
ignored automatically.

The loader also supports the IAM Words layout's extra nesting
transparently: `words/<writer>/<form-id>/*.png` is walked recursively,
`content_type` is recorded as `"words"` for those images, and — since IAM
filenames don't carry a student ID — the form-id folder name (e.g.
`"a01-000u"`) is used to group images for the student-level split, so
multiple word images from the same form/page don't leak across train/test.

## 7. Installation

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 8. Training Commands

```bash
python training/extract_reading_features.py
python training/extract_speech_embeddings.py
python training/train_speech.py
python training/evaluate_speech.py

python training/train_handwriting.py
python training/evaluate_handwriting.py
```

## 9. Running the Application

```bash
streamlit run app.py
```

## 10. Model Artifact Locations

```
artifacts/
├── speech/
│   ├── reading_features.csv
│   ├── speech_features.csv
│   ├── speech_classifier.pkl
│   ├── feature_columns.json
│   ├── class_mapping.json
│   ├── metrics.json
│   ├── confusion_matrix.png
│   └── embeddings/
└── handwriting/
    ├── resnet18.pth
    ├── class_mapping.json
    ├── preprocessing.json
    ├── metrics.json
    └── confusion_matrix.png
```

## 11. Evaluation Metrics

Both pipelines report: Accuracy, Precision, Recall, F1, ROC-AUC (where
defined), Sensitivity, and Specificity, plus a confusion matrix. Any metric
that cannot be computed given the dataset/model configuration is reported
as `N/A` rather than fabricated.

## 12. Current Limitations

- The MPS dataset loader is written defensively but has not been validated
  against every possible MPS download layout — verify field names once you
  have the actual files.
- Student-level (grouped) train/test splitting requires a `student_id`
  field in MPS annotations, and a student/writer ID pattern in handwriting
  filenames; if these are absent, the code falls back to a random split and
  logs a warning. This is a real limitation, not a hidden assumption.
- Live speech inference on an arbitrary uploaded WAV file cannot compute
  reference-dependent reading features (accuracy, WCPM, etc.) without a
  known reference transcript and alignment; those are reported as 0 in the
  live inference feature dict, while pause-based features are still
  computed from the audio itself.
- No behavioral data, no fusion, no final dyslexia score. See below.

## 13. Future Development Stage (Not Implemented Here)

A later stage will integrate a **genuine, clinically appropriate,
English dyslexia-labelled speech dataset**, train a dedicated dyslexia
speech classifier on it, and combine that with the handwriting model via
late fusion into a final screening score — only after that dataset's label
definitions have been verified. That stage is intentionally out of scope
for this codebase.
