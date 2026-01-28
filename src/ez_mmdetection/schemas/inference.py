from typing import List
from pydantic import BaseModel, Field

class Prediction(BaseModel):
    """A single object detection prediction."""
    label: int
    score: float
    bbox: List[float] = Field(description="Bounding box in [x1, y1, x2, y2] format")

class InferenceResult(BaseModel):
    """The collection of all predictions for a single image."""
    predictions: List[Prediction]
    
    @classmethod
    def from_mmdet(cls, mmdet_result: dict) -> "InferenceResult":
        """Converts MMDetection raw result to a structured InferenceResult."""
        # DetInferencer result format: 
        # {'predictions': [{'labels': [...], 'scores': [...], 'bboxes': [[...]]}]}
        raw_preds = mmdet_result.get("predictions", [])
        if not raw_preds:
            return cls(predictions=[])
            
        first_pred = raw_preds[0]
        labels = first_pred.get("labels", [])
        scores = first_pred.get("scores", [])
        bboxes = first_pred.get("bboxes", [])
        
        preds = []
        for l, s, b in zip(labels, scores, bboxes):
            preds.append(Prediction(label=l, score=s, bbox=b))
            
        return cls(predictions=preds)
