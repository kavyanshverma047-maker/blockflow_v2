from fastapi import APIRouter

router = APIRouter()

@router.get('/metrics/test')
def test_metrics():
    return {'status': 'ok'}
from fastapi import APIRouter

router = APIRouter()

@router.get("/metrics/test")
def test_metrics():
    return {"status": "ok"}
