import urllib.request
import json

def test_api_pipeline():
    url = "http://localhost:8000/api/analyze"
    req_body = {
        "image_path": "backend/datasets/samples/china_offshore_dongying_engine.jpg",
        "mode": "balanced",
        "frame_idx": 1
    }
    data = json.dumps(req_body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        
    print("Status:", res.get("status"))
    print("Analysis ID:", res.get("analysis_id"))
    print("Detections count:", len(res.get("detections", [])))
    print("Raw Image URL:", res.get("raw_image_url"))
    print("Enhanced Image URL:", res.get("enhanced_image_url"))
    print("Center WGS84:", res.get("center_wgs84"))
    
    assert res.get("status") == "success", "status must be success"
    assert len(res.get("detections", [])) > 0, "must detect targets"
    
    # Check if raw and enhanced URLs are accessible
    base_url = "http://localhost:8000"
    if res.get("raw_image_url"):
        raw_full = base_url + res.get("raw_image_url")
        with urllib.request.urlopen(raw_full) as r_img:
            assert r_img.status == 200, f"Raw image {raw_full} not 200"
            print("Raw Image stream HTTP 200 OK")
            
    if res.get("enhanced_image_url"):
        enh_full = base_url + res.get("enhanced_image_url")
        with urllib.request.urlopen(enh_full) as e_img:
            assert e_img.status == 200, f"Enhanced image {enh_full} not 200"
            print("Enhanced Image stream HTTP 200 OK")

    # Check GIS map data endpoint
    gis_url = "http://localhost:8000/api/gis/map-data"
    with urllib.request.urlopen(gis_url) as g_resp:
        g_data = json.loads(g_resp.read().decode("utf-8"))
        print("GIS Database targets count:", len(g_data.get("targets", [])))
        assert len(g_data.get("targets", [])) >= len(res.get("detections", [])), "GIS database should store targets"

    print("\nALL BACKEND & GIS PIPELINE VERIFICATIONS PASSED!")

if __name__ == "__main__":
    test_api_pipeline()
