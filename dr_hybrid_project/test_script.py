import requests

url = "http://127.0.0.1:5001/scanner"
file_path = r"d:\all mini projects(codes)\dr_hybrid_project\dr_hybrid_project\data\test_images\0a0780ad3395.png"

print("Uploading test image...")
try:
    with open(file_path, "rb") as f:
        files = {"file": f}
        response = requests.post(url, files=files)
        
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        if "Inference error" in response.text:
            print("FAILED: Inference error flashed on the page.")
        elif "No file uploaded" in response.text:
            print("FAILED: No file uploaded.")
        elif 'src="/outputs/' in response.text:
            print("SUCCESS: Image outputs successfully rendered on the diagnostic page.")
        else:
            print("FAILED: Reached page, but output images not found in the HTML.")
    else:
        print("FAILED: Status code != 200")
except Exception as e:
    print(f"Connection error: {e}")
