from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import shutil
from pathlib import Path
import uvicorn
import os
#from face_cluster_singlefile import cosine_similarity


# Import your pipeline
from face_cluster_singlefile import run_pipeline, DetectorEncoder, cosine_similarity
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow React Frontend
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Directories
UPLOAD_DIR = Path("images")
OUTPUT_DIR = Path("output")
CACHE_FILE = Path("cache/embeddings.pkl")

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
CACHE_FILE.parent.mkdir(exist_ok=True)

# Serve output images as static files
app.mount("/static", StaticFiles(directory="output"), name="static")


# --------------------------
# 1) Upload images endpoint
# --------------------------
@app.post("/images")
async def upload_images(files: list[UploadFile] = File(...)):
    saved = []

    for file in files:
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        saved.append(str(file_path))

    return {"status": "success", "saved_files": saved}


# --------------------------
# 2) Run clustering endpoint
# --------------------------
@app.post("/run_clustering")
async def run_clustering():
    input_dir = UPLOAD_DIR
    output_dir = OUTPUT_DIR

    # 1️⃣ Delete cache file (embeddings.pkl)
    if CACHE_FILE.exists():
        CACHE_FILE.unlink()
        print("Cache deleted")

    # 2️⃣ Delete old clusters folder
    clusters_dir = OUTPUT_DIR / "clusters"
    if clusters_dir.exists():
        shutil.rmtree(clusters_dir)
        print("Old clusters deleted")

    # 3️⃣ Delete old previews folder
    previews_dir = OUTPUT_DIR / "face_previews"
    if previews_dir.exists():
        shutil.rmtree(previews_dir)
        print("Old previews deleted")

    # 4️⃣ Recreate these folders cleanly
    clusters_dir.mkdir(parents=True, exist_ok=True)
    previews_dir.mkdir(parents=True, exist_ok=True)

    print("Running fresh clustering...")

    # 5️⃣ Run fresh pipeline (NO CACHE)
    run_pipeline(
        input_dir=input_dir,
        output_dir=output_dir,
        cache_file=CACHE_FILE,
        device="cpu",
        eps=0.9,
        min_samples=2,
        face_threshold=0.75,
        use_cache=False
    )

    return {"status": "fresh_clustering_done"}


# --------------------------
# 3) Get clusters endpoint
# --------------------------
@app.get("/clusters")
async def get_clusters():
    clusters_dir = OUTPUT_DIR / "clusters"
    cluster_list = []

    if not clusters_dir.exists():
        return []

    for folder in clusters_dir.iterdir():
        if not folder.is_dir() or not folder.name.startswith("Person_"):
            continue

        person_id = folder.name.replace("Person_", "")
        images = [f for f in folder.iterdir() if f.is_file()]

        if not images:
            continue

        preview_url = f"http://127.0.0.1:5000/static/clusters/Person_{person_id}/{images[0].name}"

        cluster_list.append({
            "id": int(person_id),
            "name": folder.name,
            "count": len(images),
            "preview": preview_url
        })

    return cluster_list

# --------------------------
# 4) Get cluster detail
# --------------------------
@app.get("/cluster/{cluster_id}")
async def cluster_detail(cluster_id: int):

    folder = OUTPUT_DIR / "clusters" / f"Person_{cluster_id}"

    if not folder.exists():
        return JSONResponse({"error": "Cluster not found"}, status_code=404)

    images = [
        f"http://127.0.0.1:5000/static/clusters/Person_{cluster_id}/{f.name}"
        for f in folder.iterdir()
        if f.is_file()
    ]

    return {
        "id": cluster_id,
        "images": images
    }


# --------------------------
# 5) Search person endpoint (optional)
# --------------------------
@app.post("/find_person")
async def find_person(image: UploadFile = File(...)):

    search_file = Path("output/search_data.pkl")
    if not search_file.exists():
        return {"error": "Run clustering first. search_data.pkl is missing."}

    # Save temp uploaded image
    temp_path = "temp_search.jpg"
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(image.file, f)

    # Detect + encode query face
    de = DetectorEncoder(device="cpu")
    faces = de.detect_faces(temp_path)

    if not faces:
        return {"error": "No face detected in uploaded image."}

    query_embedding = de.encode_faces_batch([faces[0]])[0]

    # Load stored embeddings
    import pickle
    data = pickle.load(open(search_file, "rb"))
    stored_embeddings = data["embeddings"]
    stored_labels = data["labels"]

    clusters_root = Path("output/clusters")

    matches = []
    seen_images = set()   # prevent duplicates

    for emb, label in zip(stored_embeddings, stored_labels):
        sim = cosine_similarity(query_embedding, emb)

        if sim < 0.60:
            continue

        person_folder = clusters_root / f"Person_{label}"
        if not person_folder.exists():
            continue

        for img_file in person_folder.iterdir():
            if not img_file.is_file():
                continue

            key = f"{label}_{img_file.name}"
            if key in seen_images:
                continue

            seen_images.add(key)

            url = f"http://127.0.0.1:5000/static/clusters/Person_{label}/{img_file.name}"

            matches.append({
                "score": round(sim, 3),
                "photo": url,
                "label": label
            })

    if not matches:
        return {"matches": []}

    matches.sort(key=lambda x: x["score"], reverse=True)

    return {
        "best_match": matches[0],
        "matches": matches
    }

# --------------------------
# Run the API
# --------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5000)
