#!/bin/bash

# Màu sắc cho terminal
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Khởi động lại MLflow hoàn toàn ===${NC}"
echo ""

# Dừng MLflow server nếu đang chạy
echo -e "Đang dừng MLflow server nếu đang chạy..."
lsof -ti:5002 | xargs kill -9 2>/dev/null || true

# Gỡ cài đặt thư viện
echo -e "Gỡ cài đặt SQLAlchemy và MLflow..."
pip uninstall -y sqlalchemy mlflow || true

# Cài đặt lại thư viện với version tương thích
echo -e "Cài đặt lại SQLAlchemy và MLflow với version tương thích..."
echo "${GREEN}===== Cài đặt lại các thư viện cần thiết =====${NC}"
pip install --no-deps "sqlalchemy>=1.4.0,<2.0.0" "mlflow>=2.0.0,<2.8.0"
# Cài đặt các dependencies cần thiết
pip install cloudpickle packaging protobuf flask gunicorn sqlparse markdown

if [ $? -ne 0 ]; then
    echo "${RED}Lỗi: Không thể cài đặt thư viện${NC}"
    exit 1
fi

# Xóa thư mục MLflow data cũ
echo -e "Xóa dữ liệu MLflow cũ..."
rm -rf mlruns mlflow_data

# Khởi tạo MLflow mới
echo -e "Khởi tạo MLflow mới..."
if python mlflow_config.py; then
    echo -e "${GREEN}Khởi tạo MLflow thành công!${NC}"
else  
    echo -e "${RED}Lỗi khi khởi tạo MLflow!${NC}"
    exit 1
fi

# Kiểm tra thư mục MLflow data mới
if [ -d "mlflow_data" ]; then
    echo -e "${GREEN}Dữ liệu MLflow mới đã được tạo!${NC}"
else
    echo -e "${RED}Không tìm thấy thư mục dữ liệu MLflow. Có thể có lỗi!${NC}"
    exit 1
fi

# Khởi động MLflow server
echo -e "${GREEN}Khởi động MLflow server...${NC}"
echo "MLflow đang chạy tại: http://localhost:5002"
echo "Nhấn Ctrl+C để dừng"

# Sử dụng script mới để khởi động MLflow server
python run_mlflow_server.py --host 127.0.0.1 --port 5002 \
    --backend-store-uri "./mlflow_data/mlflow.db" \
    --default-artifact-root "./mlflow_data/artifacts" 