# 🍃 LeafGuard — Plant Disease Detection
 
Upload a photo of a plant leaf and LeafGuard tells you the likely disease, how confident it is, the top‑3 alternatives, and plain‑language next steps. It also shows **where on the leaf** the model looked, using an occlusion‑sensitivity heat‑map.
 
**🔗 Live demo:** https://huggingface.co/spaces/shhreyyyanshh/leaguard
 
> Deep Learning PBL project · B.Tech CSE (AI/ML), UPES Dehradun
 
## Features
- **Disease classification** across 14 plants / 38 classes (PlantVillage) with transfer learning (MobileNetV2 or ResNet18, ImageNet‑pretrained)
- **Automatic leaf detection**: finds the leaf, crops it onto a neutral background, then classifies
- **Whole‑leaf heat‑map**: hides each patch of the leaf in turn and shows how much it mattered to the prediction
- **Top‑3 predictions** with confidence and a low‑confidence warning
- **Advice** for each condition, shown in plain language
- **Scan history and stats** stored in SQLite
- **Evaluation page** with accuracy, macro‑F1, latency and confusion matrix
- Clean single‑page web UI, REST API (FastAPI)
## Tech stack
Python · PyTorch / torchvision · FastAPI · SQLite · HTML/CSS/JS · Docker (Hugging Face Spaces)
 
## Architecture
```
frontend/index.html --REST--> FastAPI  backend/app/main.py
                                 ├─ model.py   CNN + prediction + whole-leaf scan
                                 ├─ leaf.py    leaf detection and crop
                                 ├─ advice.py  class name -> crop, condition, advice
                                 └─ store.py   SQLite scan history + stats
 
scripts/train.py -> models/leafguard.pt -> evaluation/evaluate.py -> results.json + confusion_matrix.png
```
 
## Project structure
```
leafguard/
├── backend/app/     API, model, leaf detection, advice, storage
├── frontend/        Single-page web UI
├── scripts/         dataset download, training, setup check
├── evaluation/      evaluate.py (+ generated results)
├── tests/           pytest suite
├── docs/            REPORT.md, VIVA.md, PRESENTATION.md
├── models/          trained weights (generated)
├── dataset/         PlantVillage images (downloaded, not committed)
├── Dockerfile       for Hugging Face Spaces
└── run.bat / run.sh one-command launchers
```
 
## Quick start
 
**Requirements:** Python 3.10+ and internet on the first run.
 
**Windows:** double‑click `run.bat`.
**Linux / macOS:** `./run.sh`
 
The launcher installs dependencies, downloads a PlantVillage sample, trains a quick model, evaluates it, and starts the app at **http://localhost:8000**.
 
### Manual setup
```bash
git clone https://github.com/YOUR_USERNAME/leafguard.git
cd leafguard
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
 
python scripts/download_dataset.py --per-class 500   # or put a Kaggle PlantVillage copy in dataset/
python scripts/train.py --data dataset --epochs 8    # use a GPU if you have one
python evaluation/evaluate.py
uvicorn backend.app.main:app --port 8000
```
 
### Dataset
- Default: `scripts/download_dataset.py` fetches a per-class sample of PlantVillage from GitHub (no login).
- Full data: [PlantVillage](https://www.kaggle.com/datasets/emmarex/plantdisease) (15 classes, ~20k images) or [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) (38 classes, ~87k images). Unzip into `dataset/`. Nested folders are fine.
- Original data: Hughes & Salathé, 2015. Check the licence on the page you download from.
## Training
Per‑class 80/10/10 train/val/test split with a fixed seed. Epoch 1 trains only the new head; later epochs fine‑tune the whole network (AdamW, label smoothing 0.05). Augmentation: random crop, flips, rotation, colour jitter. The best‑validation checkpoint is saved, and the test split is never used for training or model selection.
 
## Results
Run `python evaluation/evaluate.py` to generate `evaluation/results.json` and `confusion_matrix.png`.
 
| Metric | Value |
|---|---|
| Test accuracy | _fill in from results.json_ |
| Macro F1 | _fill in_ |
| Mean latency | _fill in_ |
 
> Note: evaluation uses plain centre‑cropped test images and does not include the leaf‑detection step.
 
## API
| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/predict` | Upload an image (`file`); returns prediction, top‑3, advice, heat‑map |
| GET | `/api/health` | Server and model status |
| GET | `/api/classes` | Supported plants and conditions |
| GET | `/api/history` | Recent scans |
| DELETE | `/api/history` | Clear history |
| GET | `/api/analytics` | Model info and usage stats |
| GET | `/api/evaluation` | Saved evaluation results |
 
## Tests
```bash
pytest -q
```
Covers advice parsing, the "no model" 503 path, a prediction through an untrained network (response shape, heat‑map, history, analytics) and invalid‑upload rejection. It does not measure accuracy.
 
## Deploy to Hugging Face Spaces
Create a **Docker** Space, add the `Dockerfile` and a README with this header, then upload the project including `models/leafguard.pt`:
```yaml
---
title: LeafGuard
emoji: 🍃
sdk: docker
app_port: 7860
---
```
 
## Limitations
- Only knows the PlantVillage plants and classes. Other plants (e.g. mango) or problems like nutrient deficiency get a low‑confidence guess among known classes.
- PlantVillage photos are lab‑style (single leaf, plain background); field photos will score lower.
- Near‑duplicate images across splits can inflate test accuracy.
- The model is closed‑set: it always picks a class, even for a non‑leaf photo.
- Leaf detection works best on plain backgrounds; on cluttered photos it falls back to analysing the whole image.
- Advice is generic and **not a substitute for an agronomist's diagnosis**.
## Docs
[`docs/REPORT.md`](docs/REPORT.md) · [`docs/VIVA.md`](docs/VIVA.md) · [`docs/PRESENTATION.md`](docs/PRESENTATION.md)
 
## Author
**Shreyansh Awasthi** — B.Tech CSE (AI/ML), UPES Dehradun
 
## Acknowledgements
PlantVillage dataset (Hughes & Salathé, 2015) · PyTorch / torchvision · FastAPI
 
