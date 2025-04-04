#!/bin/bash
# Script chạy kiểm thử cho ứng dụng Docker

# Set màu sắc cho output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Thông báo bắt đầu
echo -e "${YELLOW}====== Bắt đầu kiểm thử ứng dụng Docker ======${NC}"
echo "$(date)"
echo

# Kiểm tra Docker đang chạy
echo -e "${YELLOW}Kiểm tra Docker đang chạy...${NC}"
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}Docker không chạy hoặc không có quyền truy cập. Vui lòng khởi động Docker và thử lại.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Docker đang chạy${NC}"
echo

# Kiểm tra nếu docker-compose có sẵn
echo -e "${YELLOW}Kiểm tra docker-compose...${NC}"
if ! command -v docker-compose > /dev/null; then
    if ! command -v docker > /dev/null || ! docker compose version > /dev/null 2>&1; then
        echo -e "${RED}docker-compose không có sẵn. Vui lòng cài đặt docker-compose hoặc sử dụng Docker Desktop với compose plugin.${NC}"
        exit 1
    else
        echo -e "${GREEN}✓ Docker Compose plugin có sẵn${NC}"
        COMPOSE_CMD="docker compose"
    fi
else
    echo -e "${GREEN}✓ docker-compose có sẵn${NC}"
    COMPOSE_CMD="docker-compose"
fi
echo

# Chạy docker build và docker-compose up trước tiên để đảm bảo mọi thứ up-to-date
echo -e "${YELLOW}Cập nhật và khởi động các container...${NC}"
if $COMPOSE_CMD build && $COMPOSE_CMD up -d; then
    echo -e "${GREEN}✓ Containers đã được khởi động thành công${NC}"
else
    echo -e "${RED}Lỗi khi khởi động containers. Kiểm tra logs với 'docker-compose logs'.${NC}"
    exit 1
fi
echo

# Đợi dịch vụ sẵn sàng
echo -e "${YELLOW}Đợi dịch vụ khởi động (10 giây)...${NC}"
sleep 10
echo -e "${GREEN}✓ Đã đợi đủ thời gian khởi động${NC}"
echo

# Chạy các tests
echo -e "${YELLOW}Chạy các tests Docker...${NC}"
PYTHONPATH=.. python test_docker_integration.py -v

TEST_RESULT=$?

echo
if [ $TEST_RESULT -eq 0 ]; then
    echo -e "${GREEN}====== Tất cả tests đã thành công! ======${NC}"
else
    echo -e "${RED}====== Một số tests thất bại. Kiểm tra logs để biết thêm chi tiết. ======${NC}"
fi

echo
echo -e "${YELLOW}Các containers sẽ tiếp tục chạy cho việc phát triển.${NC}"
echo -e "${YELLOW}Để dừng chúng, chạy: docker-compose down${NC}"
echo

exit $TEST_RESULT 