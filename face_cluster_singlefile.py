#!/usr/bin/env python3
"""
face_cluster_singlefile.py

Single-file pipeline:
- MTCNN face detection (facenet-pytorch)
- InceptionResnetV1 embeddings (VGGFace2 pretrained) -> 512-d, L2-normalized
- DBSCAN clustering
- Cache embeddings to PKL
- Organize photos into cluster folders and create preview images
"""

import os
import argparse
import pickle
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
from tqdm import tqdm
from PIL import Image
import shutil

# ML libs
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.cluster import DBSCAN
#from face_cluster_singlefile import cosine_similarity

# ---------------------------
# Helpers
# ---------------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def save_pickle(obj, path: Path):
    ensure_dir(path.parent)
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(path: Path):
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

def l2_normalize(v: np.ndarray):
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two embeddings."""
    a_norm = a / (np.linalg.norm(a) + 1e-10)
    b_norm = b / (np.linalg.norm(b) + 1e-10)
    return float(np.dot(a_norm, b_norm))


# ---------------------------
# Detection & Encoding
# ---------------------------
class DetectorEncoder:
    def __init__(self, device: str = "cpu", image_size: int = 160):
        # MTCNN (detect & crop)
        mtcnn_device = "cpu" if device == "mps" else device  # MTCNN has MPS quirks
        self.mtcnn = MTCNN(image_size=image_size, margin=0, keep_all=True, device=mtcnn_device)
        # InceptionResnetV1 (encoder)
        self.device = device
        self.resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    def detect_faces(self, img_path: str, threshold: float = 0.75) -> List[np.ndarray]:
        """
        Returns list of face crops as numpy arrays in uint8 RGB (H,W,C)
        """
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Could not open {img_path}: {e}")
            return []

        # mtcnn returns face tensors normalized to [-1,1] by facenet-pytorch
        faces_t, probs = self.mtcnn(img, return_prob=True)
        if faces_t is None:
            return []

        faces = []
        # handle single face (3-d tensor) vs multiple (4-d)
        if faces_t.dim() == 3:
            faces_t = faces_t.unsqueeze(0)
            probs = [probs]

        for i, (f_t, p) in enumerate(zip(faces_t, probs)):
            if p is None or p < threshold:
                continue
            # convert tensor [-1,1] to uint8 0-255
            arr = ((f_t.permute(1, 2, 0).cpu().numpy() + 1) * 127.5).astype(np.uint8)
            faces.append(arr)
        return faces

    def encode_faces_batch(self, face_arrays: List[np.ndarray], batch_size: int = 16) -> List[np.ndarray]:
        """
        face_arrays: list of numpy arrays HWC uint8
        returns list of 512-d L2-normalized numpy arrays (float32)
        """
        embeddings = []
        if not face_arrays:
            return embeddings

        # Preprocess: convert to tensors and normalize like model expects
        import torchvision.transforms as transforms
        transform = transforms.Compose([
            transforms.ToTensor(),  # scales to [0,1]
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])  # -> [-1,1]
        ])

        tensors = [transform(Image.fromarray(arr)).to(self.device) for arr in face_arrays]
        for i in range(0, len(tensors), batch_size):
            batch = torch.stack(tensors[i:i+batch_size])
            with torch.no_grad():
                embs = self.resnet(batch).cpu().numpy()  # shape (B,512)
            # L2 normalize each
            for e in embs:
                e = e.astype(np.float32)
                e = e / (np.linalg.norm(e) + 1e-10)
                embeddings.append(e)
        return embeddings

# ---------------------------
# Clustering & Organization
# ---------------------------
"""def cluster_embeddings(embeddings: List[np.ndarray], eps: float = 0.9, min_samples: int = 2) -> np.ndarray:
    if len(embeddings) == 0:
        return np.array([])
    arr = np.vstack(embeddings).astype(np.float32)
    db = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
    labels = db.fit_predict(arr)
    return labels
"""


def organize_photos_by_cluster(image_paths: List[str], face_data: List[Dict], labels: np.ndarray, output_dir: Path, min_cluster_size: int = 1) -> Dict:
    """
    face_data: list of dicts per image: { 'image_path':..., 'faces': [face_arrays], 'face_indices': [global_face_idx,...] }
    labels: array of length total_faces mapping each face to cluster label
    This function copies original photo files into Person_{label} folders (unique photos per cluster)
    """
    clusters = {}
    total_faces = len(labels)
    # build a mapping from global face index -> photo path
    face_index_to_photo = []
    for img_idx, d in enumerate(face_data):
        n = len(d.get("faces", []))
        for k in range(n):
            face_index_to_photo.append(d["image_path"])

    # group photo paths per cluster (use set)
    for face_idx, label in enumerate(labels):
        if label == -1:
            continue
        clusters.setdefault(int(label), set()).add(face_index_to_photo[face_idx])

    stats = {"valid_clusters": 0, "photos_organized": 0, "cluster_info": {}}
    clusters_dir = output_dir / "clusters"
    ensure_dir(clusters_dir)

    for label, photos in clusters.items():
        if len(photos) < min_cluster_size:
            continue
        person_folder = clusters_dir / f"Person_{label}"
        ensure_dir(person_folder)
        copied = 0
        for p in photos:
            try:
                dest = person_folder / Path(p).name
                # handle duplicate names
                if dest.exists():
                    base = Path(p).stem
                    suf = Path(p).suffix
                    i = 1
                    while (person_folder / f"{base}_{i}{suf}").exists():
                        i += 1
                    dest = person_folder / f"{base}_{i}{suf}"
                shutil.copy2(p, dest)
                copied += 1
            except Exception as e:
                print(f"Failed copy {p} -> {person_folder}: {e}")
        stats["cluster_info"][label] = {"photos_count": len(photos), "copied_count": copied, "folder": str(person_folder)}
        stats["photos_organized"] += copied
        stats["valid_clusters"] += 1
    return stats

def cluster_embeddings(embeddings: List[np.ndarray], eps: float = 0.5, min_samples: int = 2) -> np.ndarray:
    if len(embeddings) == 0:
        return np.array([])

    arr = np.vstack(embeddings).astype(np.float32)

    # DBSCAN using cosine distance = 1 - cosine_similarity
    db = DBSCAN(
        metric="cosine",
        eps=1 - 0.65,     # 0.65 similarity threshold → eps = 0.35
        min_samples=min_samples
    )

    labels = db.fit_predict(arr)
    return labels


def create_cluster_previews(face_data: List[Dict], labels: np.ndarray, output_dir: Path, max_faces_per_cluster: int = 25, face_size: int = 128):
    """
    Create grid preview images per cluster (cropped faces).
    face_data: same structure as above
    """
    from math import ceil, sqrt
    # flatten face list with mapping to image and face array
    face_entries = []
    for d in face_data:
        for face_array in d.get("faces", []):
            face_entries.append(face_array)

    cluster_faces = {}
    for i, lab in enumerate(labels):
        if lab == -1:
            continue
        cluster_faces.setdefault(int(lab), []).append(face_entries[i])

    previews_dir = output_dir / "face_previews"
    ensure_dir(previews_dir)

    for label, faces in cluster_faces.items():
        k = min(len(faces), max_faces_per_cluster)
        grid_n = int(ceil(sqrt(k)))
        canvas = Image.new("RGB", (grid_n * face_size, grid_n * face_size), "white")
        for idx in range(k):
            face = faces[idx]
            if face.max() <= 1.0:
                face_img = Image.fromarray((face * 255).astype("uint8"))
            else:
                face_img = Image.fromarray(face.astype("uint8"))
            face_img = face_img.resize((face_size, face_size))
            r = idx // grid_n
            c = idx % grid_n
            canvas.paste(face_img, (c * face_size, r * face_size))
        out_path = previews_dir / f"cluster_{label}_preview.jpg"
        canvas.save(out_path, quality=90)

# Main pipeline
# ---------------------------
def run_pipeline(input_dir: Path, output_dir: Path, cache_file: Path, device: str, eps: float, min_samples: int, face_threshold: float, use_cache: bool):
    ensure_dir(output_dir)
    ensure_dir(cache_file.parent)

    # collect images
    exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff")
    image_paths = sorted([str(p) for p in input_dir.rglob("*") if p.suffix.lower() in exts])
    if not image_paths:
        print("No images found in", input_dir)
        return

    # initialize models
    de = DetectorEncoder(device=device)

    # try load cache
    cached = load_pickle(cache_file) if use_cache else None
    if cached is None:
        cached = {}

    face_data = []
    all_embeddings = []
    print(f"Processing {len(image_paths)} images...")

    global_face_idx = 0
    for img_path in tqdm(image_paths, desc="Images"):
        # cache key relative
        key = str(Path(img_path).resolve())
        if use_cache and key in cached:
            data = cached[key]
            faces = data.get("faces", [])
            embeddings = data.get("embeddings", [])
        else:
            faces = de.detect_faces(img_path, threshold=face_threshold)
            embeddings = de.encode_faces_batch(faces, batch_size=16) if faces else []
            if use_cache:
                cached[key] = {"faces": faces, "embeddings": embeddings}

        face_data.append({"image_path": img_path, "faces": faces})
        for emb in embeddings:
            all_embeddings.append(emb)
            global_face_idx += 1

    # save cache
    if use_cache:
        save_pickle(cached, cache_file)
        print("Saved embeddings cache to", cache_file)

    print(f"Total faces detected: {len(all_embeddings)}")
    
    if len(all_embeddings) == 0:
        print("No faces detected. Still creating empty search_data.pkl.")

    # clustering
    labels = cluster_embeddings(all_embeddings, eps=eps, min_samples=min_samples)
    print("Clustering done. Unique labels:", set(labels.tolist()))

    # organize photos
    stats = organize_photos_by_cluster(image_paths, face_data, labels, output_dir, min_cluster_size=1)
    print("Organization stats:", stats)

    # create previews
    create_cluster_previews(face_data, labels, output_dir)
    print("Preview images created in:", output_dir / "face_previews")
    
    # Save embeddings + labels + image_paths for search API
    search_data = {
    "embeddings": all_embeddings,
    "labels": labels.tolist(),
    "image_paths": image_paths
    }

    with open(output_dir / "search_data.pkl", "wb") as f:
        pickle.dump(search_data, f)
        
        print("✔ search_data.pkl created at:", output_dir / "search_data.pkl")


# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Face clustering (MTCNN + InceptionResnetV1 + DBSCAN)")
    p.add_argument("--input_dir", "-i", type=str, required=True, help="Input folder with images")
    p.add_argument("--output_dir", "-o", type=str, required=True, help="Output folder")
    p.add_argument("--cache_file", type=str, default="cache/embeddings.pkl", help="Cache pickle file")
    p.add_argument("--device", type=str, default="cpu", choices=["cpu","cuda","mps"], help="Device")
    p.add_argument("--eps", type=float, default=0.9, help="DBSCAN eps")
    p.add_argument("--min_samples", type=int, default=2, help="DBSCAN min_samples")
    p.add_argument("--face_threshold", type=float, default=0.75, help="MTCNN face probability threshold")
    p.add_argument("--no_cache", action="store_true", help="Disable cache usage")
    args = p.parse_args()

    run_pipeline(Path(args.input_dir), Path(args.output_dir), Path(args.cache_file), args.device, args.eps, args.min_samples, args.face_threshold, not args.no_cache)
