from pydantic import BaseModel, Field
from typing import List, Optional


class DetectionPrediction(BaseModel):
    """A single object detection prediction."""

    label: int
    score: float
    bbox: List[float] = Field(description="Bounding box in [x1, y1, x2, y2] format")


class InferenceResult(BaseModel):
    """The collection of all object detection predictions for a single image."""

    predictions: List[DetectionPrediction]

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
            preds.append(DetectionPrediction(label=l, score=s, bbox=b))

        return cls(predictions=preds)


class PosePrediction(BaseModel):
    """A single pose estimation prediction (keypoints for one person/object)."""

    keypoints: List[List[float]] = Field(
        description="Keypoint coordinates in [[x1, y1], [x2, y2], ...] format"
    )
    keypoint_scores: List[float] = Field(
        description="Confidence scores for each keypoint"
    )
    bbox: Optional[List[float]] = Field(
        None, description="Optional bounding box in [x1, y1, x2, y2] format"
    )
    score: float = Field(
        description="Overall confidence score for this pose prediction"
    )


class PoseInferenceResult(BaseModel):
    """The collection of all pose predictions for a single image."""

    predictions: List[PosePrediction]

    @classmethod
    def from_mmpose(cls, mmpose_result: List[dict]) -> "PoseInferenceResult":
        """Converts MMPose raw result to a structured PoseInferenceResult."""
        preds = []
        for res in mmpose_result:
            keypoints = res.get("keypoints", [])
            scores = res.get("keypoint_scores", [])

            if hasattr(keypoints, "tolist"):
                keypoints = keypoints.tolist()
            if hasattr(scores, "tolist"):
                scores = scores.tolist()

            bbox = None
            if "bbox" in res:
                raw_bbox = res["bbox"]
                if isinstance(raw_bbox, list) and len(raw_bbox) > 0:
                    if hasattr(raw_bbox[0], "tolist"):
                        bbox = raw_bbox[0].tolist()
                    else:
                        bbox = raw_bbox[0]

            preds.append(
                PosePrediction(
                    keypoints=keypoints,
                    keypoint_scores=scores,
                    bbox=bbox,
                    score=float(res.get("score", 0.0)),
                )
            )

        return cls(predictions=preds)

