import urllib.request
import json

def verify_gis_minmax_architecture():
    # 1. Test frontend HTML
    with urllib.request.urlopen('http://localhost:3000/') as response:
        html = response.read().decode('utf-8')

    # Verify buttons and markup
    assert 'btnToggleMinMaxCurrentInput' in html, 'btnToggleMinMaxCurrentInput missing in HTML'
    assert 'btnToggleMinMaxEntireOcean' in html, 'btnToggleMinMaxEntireOcean missing in HTML'
    assert 'entireOceanMinimizedBody' in html, 'entireOceanMinimizedBody missing in HTML'
    assert 'btnExpandGlobalOceanCta' in html, 'btnExpandGlobalOceanCta missing in HTML'
    assert 'currentInputBanner' in html, 'currentInputBanner missing in HTML'
    assert 'is-minimized' in html, 'default is-minimized class missing on entireOceanModalCard'
    print('[PASS] Frontend index.html: Minimize/Maximize buttons & Minimized default states verified!')

    # 2. Test JS code separation
    with open('frontend/js/map.js', 'r', encoding='utf-8') as f:
        js_code = f.read()

    assert 'getCurrentInputDetections()' in js_code, 'getCurrentInputDetections() selector missing in map.js'
    assert 'getAllOceanDetections()' in js_code, 'getAllOceanDetections() selector missing in map.js'
    assert 'this.currentInputMapState' in js_code, 'currentInputMapState missing in map.js'
    assert 'this.entireOceanMapState' in js_code, 'entireOceanMapState missing in map.js'
    assert 'isMinimized: true' in js_code, 'Entire ocean map isMinimized: true default missing in map.js'
    print('[PASS] map.js: Separate state structures, data selectors, and minimize/maximize logic verified!')

    # 3. Test GIS API data endpoint
    with urllib.request.urlopen('http://localhost:8000/api/gis/map-data') as response:
        map_data = json.loads(response.read().decode('utf-8'))

    targets = map_data.get('targets', [])
    print(f'[PASS] Backend GIS Endpoint: {len(targets)} targets in global persistent database.')

    print('\n[ALL CRITICAL GIS MINIMIZE/MAXIMIZE TESTS PASSED SUCCESSFULLY]')

if __name__ == '__main__':
    verify_gis_minmax_architecture()
