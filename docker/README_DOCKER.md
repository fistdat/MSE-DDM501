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

### Phương pháp 2: Thủ công

1. Xây dựng images:
```bash
docker-compose build
```

2. Chạy containers:
```bash
docker-compose up -d
```

3. Kiểm tra trạng thái:
```bash
docker-compose ps
```

4. Xem logs:
```bash
docker-compose logs -f
```

5. Dừng và xóa containers:
```bash
docker-compose down
```

## Truy cập các dịch vụ

- **Flask App**: http://localhost:5000
- **MLflow UI**: http://localhost:5001

## Troubleshooting

1. Nếu gặp lỗi về quyền truy cập volume:
```bash
sudo chown -R $USER:$USER ./mlflow_data ./models ./tuning_results
```

2. Nếu cần xóa và xây dựng lại images:
```bash
docker-compose down --rmi all
docker-compose build --no-cache
```

3. Nếu cần xóa tất cả dữ liệu và bắt đầu lại:
```bash
docker-compose down -v
rm -rf mlflow_data/* models/* tuning_results/*
``` 