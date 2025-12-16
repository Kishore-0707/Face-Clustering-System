Face Clustering & Person Search System

An AI-powered face clustering and person search system that automatically groups photos of the same individual and allows searching for a person using a single image.
Built using Python, FastAPI, PyTorch, FaceNet, DBSCAN, and React.

🚀 Features
  ✅ Automatic face detection from images
  ✅ Face embedding generation using deep learning
  ✅ Unsupervised clustering of faces (DBSCAN)
  ✅ Organizes images into person-wise folders
  ✅ Preview images for each cluster
  ✅ Search for a person using a query image
  ✅ REST API backend with FastAPI
  ✅ React frontend for visualization
  ✅ Supports fresh clustering (no cache)

🧠 Technologies Used
Backend
  Python 3.9+
  FastAPI
  PyTorch
  facenet-pytorch (MTCNN + InceptionResnetV1)
  Scikit-learn (DBSCAN)
  NumPy, Pillow
  
Frontend
  React (Create React App)
  Axios
  HTML, CSS

📂 Project Structure
face_clustering_kishore/
│
├── backend_api.py              # FastAPI backend
├── face_cluster_singlefile.py  # Face detection, embedding, clustering logic
├── search_person_client.py     # CLI tool to search a person
├── start_backend.bat           # Start backend automatically
├── Guide.txt                   # Setup & usage guide
│
├── images/                     # Uploaded images
├── output/
│   ├── clusters/               # Clustered person folders
│   ├── face_previews/          # Cluster preview images
│   └── search_data.pkl         # Stored embeddings for search
│
├── face-ui/                    # React frontend
└── README.md

⚙️ Backend Setup (Python)
1️⃣ Create Conda Environment
conda create -n faceenv python=3.9
conda activate faceenv

2️⃣ Install Dependencies
pip install fastapi uvicorn torch torchvision facenet-pytorch scikit-learn pillow numpy tqdm

3️⃣ Run Backend
python backend_api.py

Backend will run at:
http://127.0.0.1:5000

API docs:
http://127.0.0.1:5000/docs

🎨 Frontend Setup (React)
1️⃣ Install Dependencies
cd face-ui
npm install

2️⃣ Start Frontend
npm start

Frontend runs at:
http://localhost:3000
