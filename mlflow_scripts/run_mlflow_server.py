#!/usr/bin/env python
"""
Khởi động MLflow server trực tiếp mà không sử dụng CLI
"""

import os
import logging
import sys
from mlflow.server import app
import gunicorn.app.base
import argparse

# Thêm đường dẫn hiện tại vào PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Định nghĩa hàm add_healthcheck_route trực tiếp trong file này
def add_healthcheck_route():
    """Thêm route healthcheck vào MLflow server"""
    from flask import jsonify
    
    @app.route('/health', methods=['GET'])
    def health():
        return jsonify({"status": "ok", "service": "MLflow Server"})
    
    logger.info("Đã thêm healthcheck route vào MLflow server")
    return app

class StandaloneApplication(gunicorn.app.base.BaseApplication):
    """Ứng dụng Gunicorn độc lập để chạy MLflow server"""
    
    def __init__(self, app, options=None):
        self.options = options or {}
        self.application = app
        super().__init__()
    
    def load_config(self):
        for key, value in self.options.items():
            if key in self.cfg.settings and value is not None:
                self.cfg.set(key.lower(), value)
    
    def load(self):
        return self.application

def main():
    """Chạy MLflow server"""
    parser = argparse.ArgumentParser(description="Khởi động MLflow server")
    
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", default=5002, type=int, help="Port to listen on")
    parser.add_argument("--backend-store-uri", default="./mlflow_data/mlflow.db",
                        help="URI đến storage backend (SQLite, MySQL, etc.)")
    parser.add_argument("--default-artifact-root", default="./mlflow_data/artifacts",
                        help="Đường dẫn đến thư mục lưu artifacts")
    
    args = parser.parse_args()
    
    # Nếu backend URI là đường dẫn tương đối, thêm sqlite:/// vào trước
    backend_uri = args.backend_store_uri
    if not backend_uri.startswith("sqlite://") and not backend_uri.startswith("mysql:"):
        # Đảm bảo đường dẫn tuyệt đối
        backend_path = os.path.abspath(backend_uri)
        # Tạo thư mục nếu không tồn tại
        os.makedirs(os.path.dirname(backend_path), exist_ok=True)
        backend_uri = f"sqlite:///{backend_path}"
    
    # Thư mục artifacts
    artifact_root = os.path.abspath(args.default_artifact_root)
    # Tạo thư mục artifacts nếu chưa tồn tại
    os.makedirs(artifact_root, exist_ok=True)
    
    # Thiết lập biến môi trường cho MLflow
    os.environ['MLFLOW_TRACKING_URI'] = backend_uri
    os.environ['MLFLOW_DEFAULT_ARTIFACT_ROOT'] = artifact_root
    
    logger.info(f"Khởi động MLflow server với backend: {backend_uri}")
    logger.info(f"Artifacts sẽ được lưu tại: {artifact_root}")
    
    # Thêm healthcheck route
    flask_app = add_healthcheck_route()
    
    options = {
        'bind': f"{args.host}:{args.port}",
        'workers': 1,
        'timeout': 120,  # Tăng timeout
        'accesslog': '-',
    }
    
    logger.info(f"MLflow server đang chạy tại: http://{args.host}:{args.port}")
    logger.info("Nhấn Ctrl+C để dừng server")
    
    # Khởi động gunicorn server với app đã thêm route healthcheck
    StandaloneApplication(flask_app, options).run()

if __name__ == "__main__":
    main() 