import requests
import argparse

API_URL = "http://127.0.0.1:5000/find_person"   # Backend endpoint

def search_person(image_path: str):
    print("\n🔍 Sending image to backend API for face search...\n")

    with open(image_path, "rb") as f:
        files = {"image": ("search.jpg", f, "image/jpeg")}

        response = requests.post(API_URL, files=files)

    if response.status_code != 200:
        print(f"❌ Error: {response.status_code}")
        print(response.text)
        return

    data = response.json()

    if "error" in data:
        print("❌ Backend error:", data["error"])
        return

    matches = data.get("matches", [])

    if not matches:
        print("❌ No matching person found.")
        return

    print("🎯 BEST MATCH FOUND:")
    best = matches[0]
    print(f"Photo: {best['photo']}")
    print(f"Similarity Score: {best['score']:.3f}\n")

    print("📌 ALL MATCHES:")
    for m in matches:
        print(f"{m['photo']}  (score={m['score']:.3f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search a person using backend API")
    parser.add_argument("--image", "-i", required=True, help="Path to the query face image")
    args = parser.parse_args()

    search_person(args.image)
import requests
import argparse

API_URL = "http://127.0.0.1:5000/find_person"   # Backend endpoint

def search_person(image_path: str):
    print("\n🔍 Sending image to backend API for face search...\n")

    with open(image_path, "rb") as f:
        files = {"image": ("search.jpg", f, "image/jpeg")}

        response = requests.post(API_URL, files=files)

    if response.status_code != 200:
        print(f"❌ Error: {response.status_code}")
        print(response.text)
        return

    data = response.json()

    if "error" in data:
        print("❌ Backend error:", data["error"])
        return

    matches = data.get("matches", [])

    if not matches:
        print("❌ No matching person found.")
        return

    print("🎯 BEST MATCH FOUND:")
    best = matches[0]
    print(f"Photo: {best['photo']}")
    print(f"Similarity Score: {best['score']:.3f}\n")

    print("📌 ALL MATCHES:")
    for m in matches:
        print(f"{m['photo']}  (score={m['score']:.3f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search a person using backend API")
    parser.add_argument("--image", "-i", required=True, help="Path to the query face image")
    args = parser.parse_args()

    search_person(args.image)
