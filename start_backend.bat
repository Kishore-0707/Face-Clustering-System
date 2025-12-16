@echo off
cd /d "C:\Users\kishore\OneDrive\Desktop\New project\face_clustering_kishore"

call "C:\Users\kishore\anaconda3\condabin\conda.bat" activate faceenv

python backend_api.py

pause
