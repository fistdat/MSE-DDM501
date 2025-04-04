# Hướng dẫn triển khai với Docker

## Tổng quan

Dự án này sử dụng Docker và Docker Compose để dễ dàng triển khai trên bất kỳ môi trường nào. Cấu trúc Docker bao gồm 2 dịch vụ chính:

1. **MLflow Server**: Quản lý thí nghiệm và lưu trữ các mô hình ML
2. **Flask App**: Ứng dụng web phục vụ người dùng cuối

## Yêu cầu hệ thống

- Docker (phiên bản 20.10+)
- Docker Compose (phiên bản 2.0+)
- Ít nhất 1GB RAM cho mỗi container
- 2GB dung lượng ổ đĩa trống

## Cấu trúc Docker

```
MLOps-Lab02/
├── docker-compose.yml        # Cấu hình Docker Compose
├── Dockerfile.flask          # Dockerfile cho Flask App
├── Dockerfile.mlflow         # Dockerfile cho MLflow Server
├── flask-requirements.txt    # Các gói Python cho Flask App
├── mlflow-requirements.txt   # Các gói Python cho MLflow
└── build-and-run.sh          # Script để xây dựng và chạy
```

## Volumes

Dự án sử dụng ba volume Docker để lưu trữ dữ liệu:

- **mlflow_data**: Lưu trữ cơ sở dữ liệu và artifact của MLflow
- **models**: Lưu trữ các mô hình đã huấn luyện
- **tuning_results**: Lưu trữ kết quả tinh chỉnh siêu tham số

## Cách sử dụng

### Phương pháp 1: Sử dụng script tự động

Chúng tôi cung cấp script `build-and-run.sh` để tự động hóa quá trình:

```bash
# Cấp quyền thực thi cho script
chmod +x build-and-run.sh

# Chạy script
./build-and-run.sh
```

### Phương pháp 2: Sử dụng Docker Compose thủ công

```bash
# Xây dựng và khởi động các container
docker compose up --build -d

# Kiểm tra trạng thái
docker compose ps

# Xem logs
docker compose logs -f

# Dừng các container
docker compose down
```

## Truy cập các dịch vụ

Sau khi khởi động thành công, bạn có thể truy cập:

- **Flask App**: [http://localhost:5001](http://localhost:5001)
- **MLflow UI**: [http://localhost:5002](http://localhost:5002)
- **Health check**: [http://localhost:5001/health](http://localhost:5001/health)

## Xử lý sự cố

1. **Kiểm tra logs**:
   ```bash
   docker compose logs -f
   docker compose logs -f flask-app
   docker compose logs -f mlflow
   ```

2. **Khởi động lại dịch vụ**:
   ```bash
   docker compose restart flask-app
   docker compose restart mlflow
   ```

3. **Xóa volumes và bắt đầu lại**:
   ```bash
   docker compose down -v
   docker compose up --build -d
   ```

4. **Kiểm tra tài nguyên hệ thống**:
   ```bash
   docker stats
   ```

## Tùy chỉnh

Bạn có thể tùy chỉnh cấu hình Docker bằng cách:

1. **Chỉnh sửa biến môi trường**: Tạo file `.env` hoặc sửa trực tiếp trong `docker-compose.yml`
2. **Điều chỉnh tài nguyên**: Thay đổi cấu hình giới hạn bộ nhớ trong `docker-compose.yml`
3. **Thay đổi port**: Sửa mapping port trong `docker-compose.yml` 