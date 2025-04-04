#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# Màu sắc cho thông báo
YELLOW='\033[0;33m'
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Hàm kiểm tra không gian đĩa
check_disk_space() {
  local available_space=$(df -m / | awk 'NR==2 {print $4}')
  if [ "$available_space" -lt 1000 ]; then
    echo -e "${YELLOW}CẢNH BÁO: Chỉ còn ${available_space}MB không gian trống. Cần ít nhất 1GB để xây dựng images.${NC}"
    echo -e "${YELLOW}Đề xuất: Làm sạch Docker trước khi tiếp tục:${NC}"
    echo -e "${YELLOW}  docker system prune -af --volumes${NC}"
    read -p "Bạn có muốn tiếp tục không? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
      exit 1
    fi
  fi
}

# Kiểm tra Docker có đang chạy không
check_docker_running() {
  docker info > /dev/null 2>&1
  if [ $? -ne 0 ]; then
    echo -e "${RED}Docker không chạy. Vui lòng khởi động Docker và thử lại.${NC}"
    exit 1
  fi
}

# Chạy kiểm tra
check_docker_running
check_disk_space

echo -e "${GREEN}=== MLOps Lab02 - Docker Compose Build & Run ===${NC}"

# Dừng các container đang chạy
echo -e "${YELLOW}Stopping any running containers...${NC}"
docker compose -f docker/docker-compose.yml down 2>/dev/null || true

# Xây dựng và khởi động
echo -e "${GREEN}Building and starting services...${NC}"
docker compose -f docker/docker-compose.yml up --build -d

# Kiểm tra trạng thái container
echo -e "${YELLOW}Checking container status...${NC}"
sleep 5

# Kiểm tra trạng thái cụ thể của từng dịch vụ
mlflow_status=$(docker compose -f docker/docker-compose.yml ps -q mlflow)
flask_status=$(docker compose -f docker/docker-compose.yml ps -q flask-app)

if [ -n "$mlflow_status" ] && [ -n "$flask_status" ]; then
  echo -e "${GREEN}Tất cả containers đã khởi động thành công!${NC}"
else
  echo -e "${RED}CẢNH BÁO: Một số container có thể không khởi động đúng. Kiểm tra logs để biết chi tiết.${NC}"
  echo -e "${YELLOW}Xem logs với: docker compose -f docker/docker-compose.yml logs${NC}"
fi

echo -e "${GREEN}===== Dịch vụ đã khởi động =====${NC}"
echo -e "${GREEN}MLflow UI:${NC} http://localhost:5002"
echo -e "${GREEN}Flask app:${NC} http://localhost:5001"
echo -e "${GREEN}Health check:${NC} http://localhost:5001/health"
echo ""
echo -e "${YELLOW}Để xem logs:${NC} docker compose -f docker/docker-compose.yml logs -f"
echo -e "${YELLOW}Để xem logs của MLflow:${NC} docker compose -f docker/docker-compose.yml logs -f mlflow"
echo -e "${YELLOW}Để xem logs của Flask:${NC} docker compose -f docker/docker-compose.yml logs -f flask-app" 
echo -e "${YELLOW}Để dừng dịch vụ:${NC} docker compose -f docker/docker-compose.yml down" 