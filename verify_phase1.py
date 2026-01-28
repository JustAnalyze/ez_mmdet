from ez_mmdetection import RTMDet
from ez_mmdetection.schemas.inference import InferenceResult

detector = RTMDet(model_name="rtmdet_tiny")
print(f"Detector initialized: {detector.model_name}")
print(f"Predict method available: {callable(detector.predict)}")
