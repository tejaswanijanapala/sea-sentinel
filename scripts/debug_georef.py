import glob, json, os, sys
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("backend"))
from backend.agent.orchestrator import SIHPipelineAgent
from backend.ai.geospatial.metadata_service import MetadataService
from backend.ai.geospatial.georeferencing_engine import GeoreferencingEngine
from backend.duplicate_detection.spatial_matcher import TargetMatchingService
from backend.database.local_db import LocalDatabase

agent = SIHPipelineAgent()
meta_service = MetadataService()
georef = GeoreferencingEngine()
db = LocalDatabase()
matcher = TargetMatchingService(db)

img_path = os.path.abspath('backend/outputs/uploads/175a3f8c_Barge_No_1_01.png')
res = agent.analyze_image(img_path)
meta = meta_service.extract_metadata(img_path)

print('Meta:', meta.latitude, meta.longitude, meta.sonar_range, meta.altitude, meta.heading)
for d in res['detections']:
    bbox = d.get('bbox')
    print('Raw d bbox:', bbox, type(bbox))
    g = georef.georeference_detection(bbox, (640, 640), meta)
    print(f"Detection {d.get('object_id')}: lat={g.latitude}, lon={g.longitude}, ground_range={g.ground_range_m}, bearing={g.bearing_deg}")
