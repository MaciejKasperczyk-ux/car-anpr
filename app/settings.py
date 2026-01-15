import os


WORKDIR = os.environ.get("ANPR_WORKDIR", "/app")
DEFAULT_EVALUATE_CMD = os.environ.get("ANPR_EVALUATE_CMD", "python -u evaluate.py")
MAX_LOG_LINES_IN_MEMORY = int(os.environ.get("ANPR_MAX_LOG_LINES", "5000"))