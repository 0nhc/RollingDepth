import requests

url = "http://lamb.mech.northwestern.edu:7000/process_video"
file_path = "demo_video.mp4"

# Open file in binary mode
with open(file_path, "rb") as f:
    files = {"video": f}
    data = {"preset": "paper"} # Optional
    
    print("Sending video...")
    response = requests.post(url, files=files, data=data, stream=True)

if response.status_code == 200:
    output_filename = "received_depth.npy"
    with open(output_filename, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Depth saved to {output_filename}")
else:
    print(f"Error: {response.text}")