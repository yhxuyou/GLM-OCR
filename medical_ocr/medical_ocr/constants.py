"""Medical OCR - Constants.

Global constants and default configurations.
"""

# Medical document labels
MEDICAL_LABELS = [
    "table",
    "chart",
    "diagram",
    "signature",
    "stamp",
    "header",
    "footer",
    "medical_record",
]

# OCR confidence thresholds
DEFAULT_YOLO_CONFIDENCE = 0.5
DEFAULT_YOLO_NMS_THRESHOLD = 0.45
DEFAULT_DOCUMENT_PADDING = 10

# RapidOCR thresholds
DEFAULT_RAPIDOCR_ANGLE_THRESHOLD = 5.0

# Processing limits
DEFAULT_MAX_IMAGE_PIXELS = 71372800
DEFAULT_MIN_IMAGE_PIXELS = 12544

# Cache settings
DEFAULT_CACHE_MAX_SIZE = 1000
DEFAULT_CACHE_TTL = 3600

# Pipeline pool defaults
DEFAULT_POOL_SIZE = 4
DEFAULT_SINGLE_GPU_ID = 0
DEFAULT_GPU_MEMORY_FRACTION = None

# Result formatter
DEFAULT_OUTPUT_FORMAT = "both"

# Server settings
DEFAULT_SERVER_HOST = "0.0.0.0"
DEFAULT_SERVER_PORT = 8080
DEFAULT_METRICS_PORT = 8001
