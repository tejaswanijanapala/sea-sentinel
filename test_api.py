import requests

try:
    r = requests.post(
        'http://127.0.0.1:8000/api/analyze',
        json={'image_path': r'C:\Users\jaish\.gemini\antigravity-ide\scratch\sea-sentinel\backend\tests\test_images\sample_sonar.jpg'}
    )
    data = r.json()
    if "evaluation_metrics" in data:
        print("evaluation_metrics IS PRESENT")
        print("Metrics keys:", data["evaluation_metrics"].keys())
    else:
        print("evaluation_metrics IS MISSING")
        print(data.keys())
except Exception as e:
    print(f"Error: {e}")
