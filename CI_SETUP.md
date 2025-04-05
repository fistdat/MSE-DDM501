# Thiết lập Tích hợp Liên tục (CI/CD) với CircleCI

Dự án này sử dụng CircleCI để tự động hóa quá trình build và deploy Docker image lên Docker Hub mỗi khi có thay đổi được push lên nhánh `main`.

## Cấu trúc CI/CD

- **GitHub Repository**: [fistdat/MSE-DDM501](https://github.com/fistdat/MSE-DDM501)
- **Docker Hub Repository**: [fistdat/mlops-flask](https://hub.docker.com/repository/docker/fistdat/mlops-flask/general)
- **CI/CD Service**: CircleCI

## Cách thiết lập

### 1. Tạo tài khoản CircleCI

1. Đăng ký tài khoản tại [CircleCI](https://circleci.com/) bằng tài khoản GitHub
2. Kết nối CircleCI với GitHub repository của bạn (fistdat/MSE-DDM501)

### 2. Cấu hình Docker Hub Credentials trong CircleCI

1. Đăng nhập vào CircleCI
2. Chọn Organization Settings (hoặc Project Settings)
3. Chọn "Contexts"
4. Tạo mới context với tên "docker-hub-credentials"
5. Thêm hai biến môi trường:
   - DOCKER_USERNAME: tên đăng nhập Docker Hub của bạn
   - DOCKER_PASSWORD: mật khẩu hoặc token truy cập của bạn (khuyến nghị sử dụng Personal Access Token)

### 3. Tệp Cấu hình CircleCI

Dự án đã được định cấu hình với tệp `.circleci/config.yml` để thực hiện các bước sau:

- Checkout code từ GitHub
- Build Docker image từ Dockerfile.flask
- Đẩy image lên Docker Hub với tag `latest`
- Tạo tag version mới dựa trên số build của CircleCI (ví dụ: v1.42)
- Đẩy image được tag version lên Docker Hub

## Quy trình CI/CD

1. Developer push code lên nhánh `main` của GitHub repository
2. CircleCI tự động phát hiện thay đổi và trigger workflow
3. CircleCI build Docker image dựa trên Dockerfile.flask
4. CircleCI đăng nhập vào Docker Hub bằng credentials được cấu hình
5. Image được push lên Docker Hub dưới hai tag:
   - `fistdat/mlops-flask:latest`
   - `fistdat/mlops-flask:v1.X` (X là số build của CircleCI)

## Kiểm tra và Theo dõi

- Theo dõi quá trình build trên CircleCI Dashboard
- Kiểm tra xem image đã được đẩy lên Docker Hub chưa
- Sử dụng lệnh `docker pull fistdat/mlops-flask:latest` để kéo image mới nhất về

## Xử lý Lỗi Phổ biến

1. **Lỗi xác thực Docker Hub**:
   - Kiểm tra lại DOCKER_USERNAME và DOCKER_PASSWORD trong context
   - Đảm bảo token truy cập vẫn còn hiệu lực

2. **Lỗi build Docker**:
   - Kiểm tra logs trong CircleCI 
   - Đảm bảo Dockerfile không có lỗi

3. **Workflow không được kích hoạt**:
   - Kiểm tra cấu hình branch filters
   - Đảm bảo đang push lên nhánh `main`

## Cải tiến Tương lai

- Thêm kiểm thử tự động trước khi build
- Thêm kiểm tra bảo mật cho Docker image
- Tích hợp thông báo qua Slack/Email khi build thành công/thất bại 