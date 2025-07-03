# RadioDINO: A Foundation Model for Advanced Radiomics and Medical Imaging

**RadioDINO** is a family of self-supervised Vision Transformer (ViT) models specifically developed for radiomics and medical image analysis. It leverages the DINO and DINOv2 frameworks, trained on the large-scale **RadImageNet** dataset (1.35M images across CT, MRI, and ultrasound modalities), to learn rich, transferable features that perform strongly on downstream tasks like classification and segmentation — without relying on manual labels.

> **Authors:** Luca Zedda, Andrea Loddo, Cecilia Di Ruberto  
> **Affiliation:** Department of Mathematics and Computer Science, University of Cagliari  
> **Publication:** Computers in Biology and Medicine, Volume 195, 2025  
> **DOI:** [10.1016/j.compbiomed.2025.110583](https://doi.org/10.1016/j.compbiomed.2025.110583)

---

## 🔧 Example Usage

```python
from PIL import Image
from torchvision import transforms
import timm, torch

# Load pretrained model from Hugging Face Hub
model = timm.create_model("hf_hub:Snarcy/RadioDino-s16", pretrained=True)
model.eval().to("cuda" if torch.cuda.is_available() else "cpu")

# Prepare image
image = Image.open("your_image.png").convert("RGB")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])
input_tensor = transform(image).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")

# Extract features
with torch.no_grad():
    embedding = model(input_tensor)
```

---

## 📊 Benchmark Results

RadioDINO was evaluated across a wide set of medical imaging datasets and tasks, demonstrating state-of-the-art performance in most settings.

| Dataset         | Task           | Accuracy (%) | F1 Score (%) | AUC (%)   | Model Variant     |
|----------------|----------------|--------------|--------------|-----------|-------------------|
| BreastMNIST     | Classification | 91.67        | 88.98        | 95.55     | Radio DINO small  |
| PneumoniaMNIST  | Classification | 91.83        | 90.86        | 98.86     | Radio DINO small  |
| OrganAMNIST     | Classification | 97.35        | 97.20        | 99.93     | Radio DINO base   |
| OrganCMNIST     | Classification | 95.11        | 94.57        | 99.86     | Radio DINO base   |
| OrganSMNIST     | Classification | 82.30        | 77.73        | 98.27     | Radio DINO small  |
| BUSI            | Classification | 92.41        | 91.73        | 98.35     | Radio DINO base   |
| BUSI            | Segmentation   | —            | 69.5 (Dice)  | —         | Radio DINO base   |
| RadImageNet     | kNN (k=1)      | 66.2         | 35.3         | —         | Radio DINO tiny   |

---

## 🧠 Highlights

- Self-supervised learning with **DINO** / **DINOv2**
- Domain-specific pretraining on **RadImageNet**
- Strong embeddings for classification and segmentation
- Excellent performance with both zero-shot and fine-tuning
- Interpretable with Grad-CAM and attention visualizations

---

## 📄 License

This project is licensed under the **Creative Commons Attribution 4.0 International License (CC BY 4.0)**.

---

## 📚 Citation

If you use RadioDINO in your research, please cite the following:

```bibtex
@article{ZEDDA2025110583,
  title = {Radio DINO: A foundation model for advanced radiomics and AI-driven medical imaging analysis},
  journal = {Computers in Biology and Medicine},
  volume = {195},
  pages = {110583},
  year = {2025},
  doi = {10.1016/j.compbiomed.2025.110583},
  author = {Luca Zedda and Andrea Loddo and Cecilia Di Ruberto}
}
```

## 📁 Repository Content

```markdown
📁 examples/
├── embedding_classification.ipynb         # kNN classification using embeddings
├── embedding_visualization.ipynb          # PCA and UMAP visualizations of feature space
├── feature_extraction_dataset.ipynb       # Batch feature extraction from datasets
├── feature_extraction_single_image.ipynb  # Single-image embedding extraction and attention map
```

