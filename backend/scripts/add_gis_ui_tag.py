import re

path = 'frontend/index.html'
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

if 'js/gis_ui.js' not in text:
    text = text.replace(
        '<script src="js/app.js?v=3.2"></script>',
        '<script src="js/app.js?v=3.2"></script>\n  <script src="js/gis_ui.js?v=3.2"></script>'
    )
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    print("Added gis_ui.js to index.html")
else:
    print("gis_ui.js already in index.html")
