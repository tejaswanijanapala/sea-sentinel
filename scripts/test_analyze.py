import urllib.request
import json
import os
import glob

barge_files = glob.glob('backend/**/Barge*.png', recursive=True) + glob.glob('**/*Barge*.png', recursive=True)
target_path = os.path.abspath(barge_files[0] if barge_files else r'backend/datasets/samples/noaa_h11584_gulf_sample.tif')
print('Target path:', target_path, 'Exists:', os.path.exists(target_path))

payload = json.dumps({'image_path': target_path, 'mode': 'balanced'}).encode('utf-8')
req = urllib.request.Request('http://localhost:8000/api/analyze', data=payload, headers={'Content-Type': 'application/json'})
res = urllib.request.urlopen(req)
result = json.loads(res.read().decode('utf-8'))

print('survey_id:', result.get('survey_id'))
print('center_wgs84:', result.get('center_wgs84'))
print('bbox_wgs84:', result.get('bbox_wgs84'))
detections = result.get('detections', [])
print(f'Detected targets count: {len(detections)}')
for idx, d in enumerate(detections):
    print(f"Target #{idx+1}: id={d.get('object_id')}, target_id={d.get('target_id')}, class={d.get('class') or d.get('class_name')}, lat={d.get('latitude')}, lon={d.get('longitude')}, priority={d.get('priority_level')}")
