import requests

# Upload the file
with open('sample_upload.txt', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8000/upload', files=files)
    print(response.json())
