from fastapi import APIRouter

from backend.debris_detection import router as detection_router
from backend.sonar_image_processing import router as processing_router
from backend.geolocation.routes.geo_routes import router as geo_router
from backend.debris_risk_scoring import router as risk_router
from backend.natural_manmade_classification import router as classification_router
from backend.duplicate_detection import router as duplicate_router
from backend.debris_density import router as density_router
from backend.sonar_quality import router as quality_router
from backend.visualization.routes.viz_routes import router as viz_router

api_router = APIRouter(prefix="/api/v2")

api_router.include_router(detection_router)
api_router.include_router(processing_router)
api_router.include_router(geo_router)
api_router.include_router(risk_router)
api_router.include_router(classification_router)
api_router.include_router(duplicate_router)
api_router.include_router(density_router)
api_router.include_router(quality_router)
api_router.include_router(viz_router)
