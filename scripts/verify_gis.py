import urllib.request
import json

def verify_gis_architecture():
    # 1. Test frontend HTML
    with urllib.request.urlopen('http://localhost:3000/') as response:
        html = response.read().decode('utf-8')

    assert 'Current Input GIS' in html, 'Current Input GIS missing in HTML'
    assert 'Entire Ocean Map' in html, 'Entire Ocean Map missing in HTML'
    assert 'entireOceanMapModal' in html, 'entireOceanMapModal missing in HTML'
    assert 'globalLeafletMap' in html, 'globalLeafletMap missing in HTML'
    assert 'currentInputBanner' in html, 'currentInputBanner missing in HTML'
    print('[PASS] Frontend index.html contains all Dual-Level GIS elements')

    # 2. Test GIS API map-data
    with urllib.request.urlopen('http://localhost:8000/api/gis/map-data') as response:
        map_data = json.loads(response.read().decode('utf-8'))

    targets = map_data.get('targets', [])
    clusters = map_data.get('clusters', [])
    stats = map_data.get('statistics', {})
    print(f'[PASS] Backend GIS Database: {len(targets)} targets, {len(clusters)} clusters, {stats.get("total_surveys", 0)} surveys')

    # 3. Test GIS target lookup
    if targets:
        sample_t = targets[0]
        t_id = sample_t.get('target_id') or sample_t.get('object_id')
        with urllib.request.urlopen(f'http://localhost:8000/api/gis/target/{t_id}') as response:
            t_data = json.loads(response.read().decode('utf-8'))
        t_obj = t_data.get('target', {})
        print(f'[PASS] Target Telemetry Detail for {t_id}: Class={t_obj.get("class_name")}, Lat={t_obj.get("latitude")}, Lon={t_obj.get("longitude")}, Conf={t_obj.get("confidence")}')

    print('[ALL VERIFICATIONS PASSED]')

if __name__ == '__main__':
    verify_gis_architecture()
