# scripts/evaluation/mobilenetcnn_evaluator.py
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score
from utils.evals import find_best_threshold_accuracy, find_best_threshold_youden
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import transforms

class MobileNetEvaluator:
    def __init__(self, model, font_path="arial.ttf", image_size=(224, 224), batch_size = 32):
        self.image_size = image_size
        self.font = ImageFont.load_default() if font_path is None else ImageFont.truetype(font_path, 20)
        self.model = model
        self.batch_size = batch_size

    def text_to_img(self, text):
        if not isinstance(text, str):
            text = str(text)

        img = Image.new("RGB", self.image_size, color=(0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.text((5, 5), text, fill=(255, 255, 255), font=self.font)

        arr = np.array(img).astype(np.float32) / 255.0
        return np.expand_dims(arr, axis=0)  # (1, H, W, 3)

    def predict_scores(self, filepath):
        # Load CSV/Parquet
        if filepath.endswith(".csv"):
            df = pd.read_csv(filepath)
        else:
            df = pd.read_parquet(filepath)

        df1 = df.head(500)
        df2 = df.tail(500)
        
        df = pd.concat([df1, df2], ignore_index=True)
        
        fraud_names = df['fraudulent_name'].astype(str).tolist()
        real_names = df['real_name'].astype(str).tolist()
        labels = df['label'].astype(float).tolist()

        # Encode to embeddings
        fraud_embs = self.encode_texts_to_embeddings(fraud_names)
        real_embs = self.encode_texts_to_embeddings(real_names)

        # Cosine similarity (batch)
        import torch.nn.functional as F
        scores = F.cosine_similarity(fraud_embs, real_embs, dim=1).cpu().numpy()

        return labels, scores

    
    def compute_metrics(self, y_true, y_scores):
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        youden_thresh = find_best_threshold_youden(fpr, tpr, thresholds)
        best_acc, best_acc_thresh = find_best_threshold_accuracy(
            y_true, y_scores, thresholds
        )

        y_pred = (y_scores > youden_thresh).astype(int)

        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "roc_auc": roc_auc,
            "threshold": youden_thresh,
            "best_accuracy": best_acc,
            "best_accuracy_threshold": best_acc_thresh,
            "roc_curve": (fpr, tpr, thresholds),
        }
        
    def encode_texts_to_embeddings(self, texts):
        """
        Convert texts to glyphs and encode them with PyTorch MobileNet.
        """
        from model_utils.utils.text_to_glyph import text_to_glyphs_batch
        from torchvision import transforms
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        # Ensure model is on GPU
        self.model = self.model.to(device)
        self.model.eval()
    
        glyphs = text_to_glyphs_batch(texts, image_size=self.image_size)
    
        preprocess = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
        all_embeddings = []
    
        for i in range(0, len(glyphs), self.batch_size):
            batch_glyphs = glyphs[i:i + self.batch_size]
    
            # cpu -> gpu
            batch_tensors = torch.stack(
                [preprocess(img) for img in batch_glyphs]
            ).to(device, non_blocking=True)
    
            if device.type == "cuda":
                torch.cuda.synchronize()
    
            with torch.no_grad():
                embeddings = self.model(batch_tensors)
    
            all_embeddings.append(embeddings)
    
            print(
                f"Processed batch {i // self.batch_size + 1} / "
                f"{(len(glyphs) - 1) // self.batch_size + 1}"
            )
    
        # Concatenate on GPU
        embeddings = torch.cat(all_embeddings, dim=0)
        return embeddings
