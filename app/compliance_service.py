from fastapi import APIRouter

router = APIRouter()

@router.get("/compliance/test")
def compliance_test():
    return {"status": "ok"}
