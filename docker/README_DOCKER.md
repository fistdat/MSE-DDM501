# Hướng dẫn triển khai với Docker

## Tổng quan

Dự án này sử dụng Docker để dễ dàng triển khai ứng dụng Flask và MLflow server. Cấu trúc Docker đơn giản với một container chính:

- **mlops-flask**: Container chứa cả Flask App và MLflow Server

## Yêu cầu hệ thống

- Docker (phiên bản 20.10+)
- Ít nhất 1GB RAM cho container
- 1GB dung lượng ổ đĩa trống

## Xây dựng và chạy

### Xây dựng image

```bash
# Sử dụng Makefile
make docker-build

# hoặc sử dụng lệnh Docker trực tiếp
docker build -t mlops-flask .
```

### Chạy container

```bash
# Sử dụng Makefile
make docker-run

# hoặc sử dụng lệnh Docker trực tiếp
docker run -p 5001:5001 -p 5002:5002 mlops-flask
```

### Truy cập ứng dụng

- **Flask App**: http://localhost:5001
- **MLflow UI**: http://localhost:5002

## Cập nhật tệp trong container

Nếu bạn sửa đổi mã nguồn và muốn cập nhật tệp trong container đang chạy, sử dụng lệnh:

```bash
# Sử dụng Makefile (sẽ copy index.html và app.py, sau đó khởi động lại container)
make docker-cp-files

# hoặc sử dụng lệnh Docker trực tiếp
docker cp templates/index.html mlops-flask:/app/templates/
docker cp app.py mlops-flask:/app/
docker restart mlops-flask
```

## Xử lý lỗi

1. Nếu không thể kết nối đến ứng dụng:
```bash
# Kiểm tra container có đang chạy không
docker ps

# Xem logs
docker logs mlops-flask
```

2. Nếu cần khởi động lại container:
```bash
docker restart mlops-flask
```

3. Nếu cần xây dựng lại image:
```bash
docker build --no-cache -t mlops-flask .
```

## Ghi chú

- Ứng dụng Flask chạy ở cổng 5001
- MLflow Server chạy ở cổng 5002
- Dữ liệu được lưu trong container, nên sẽ bị mất khi container bị xóa
- Sử dụng lệnh `docker cp` để lấy dữ liệu hoặc mô hình ra khỏi container nếu cần 