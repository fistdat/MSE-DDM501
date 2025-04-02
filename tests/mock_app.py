"""
Mock của app.py để sử dụng trong testing
"""

import os
import json
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session, abort

from tests.mock_run_tuning import mock_run_tuning_command

# Tạo app
app = Flask(__name__, template_folder='../templates')
app.secret_key = 'test_secret_key'
app.config['TESTING'] = True
app.config['WTF_CSRF_ENABLED'] = False

# Các đường dẫn thư mục
TUNING_RESULTS_DIR = 'tuning_results'
MODELS_DIR = 'models'

# Đảm bảo thư mục tồn tại
os.makedirs(TUNING_RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Sử dụng mock_run_tuning_command
run_tuning_command = mock_run_tuning_command

# Hàm lấy danh sách kết quả tuning
def get_tuning_results():
    """
    Lấy và sắp xếp các kết quả tuning theo thời gian
    
    Returns:
        List các kết quả tuning đã sắp xếp
    """
    results = []
    if not os.path.exists(TUNING_RESULTS_DIR):
        return results
    
    for filename in os.listdir(TUNING_RESULTS_DIR):
        if filename.endswith('.json'):
            file_path = os.path.join(TUNING_RESULTS_DIR, filename)
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    data['file_name'] = filename
                    results.append(data)
            except json.JSONDecodeError:
                print(f"Lỗi đọc file {filename}")
    
    # Sắp xếp kết quả theo timestamp giảm dần (nếu có)
    results.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    return results

# API endpoint kiểm tra sức khỏe hệ thống
@app.route('/health')
def health_check():
    """API endpoint kiểm tra sức khỏe"""
    # Kiểm tra xem có model đã train chưa
    model_trained = os.path.exists(os.path.join(MODELS_DIR, 'best_model.joblib'))
    
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_trained': model_trained
    })

# Route hiển thị trang chính
@app.route('/')
def index():
    """Hiển thị trang chính với danh sách kết quả tuning"""
    tuning_results = get_tuning_results()
    return render_template('index.html', tuning_results=tuning_results)

# Route chạy tuning với tham số đơn giản
@app.route('/run_simple_tuning', methods=['POST'])
def run_simple_tuning():
    """Chạy tuning với tham số đơn giản từ form"""
    model_type = request.form.get('model_type', 'random_forest')
    param_space = request.form.get('param_space', 'small')
    samples = request.form.get('samples', '1000')
    features = request.form.get('features', '20')
    cv = request.form.get('cv', '5')
    
    command = f"python simple_hyperparam_tuning.py --model {model_type} --space {param_space} --samples {samples} --features {features} --cv {cv}"
    
    success, output, result_file = run_tuning_command(command, {
        'model_type': model_type,
        'param_space': param_space,
        'samples': int(samples),
        'features': int(features),
        'cv': int(cv)
    })
    
    if success:
        flash('Tuning thành công', 'success')
    else:
        flash(f'Tuning thất bại: {output}', 'danger')
    
    return redirect(url_for('index'))

# Route chạy tuning với tham số tùy chỉnh
@app.route('/run_custom_tuning', methods=['POST'])
def run_custom_tuning():
    """Chạy tuning với tham số tùy chỉnh từ form"""
    model_type = request.form.get('model_type', 'random_forest')
    param_args = ""
    params = {
        'model_type': model_type,
        'param_space': 'custom'
    }
    
    # Xây dựng câu lệnh dựa trên model được chọn
    if model_type == 'random_forest':
        n_estimators = request.form.get('n_estimators', '')
        max_depth = request.form.get('max_depth', '')
        min_samples_split = request.form.get('min_samples_split', '')
        min_samples_leaf = request.form.get('min_samples_leaf', '')
        
        param_args = f"--n_estimators {n_estimators} "
        if max_depth:
            param_args += f"--max_depth {max_depth} "
        if min_samples_split:
            param_args += f"--min_samples_split {min_samples_split} "
        if min_samples_leaf:
            param_args += f"--min_samples_leaf {min_samples_leaf} "
    
    elif model_type == 'gradient_boosting':
        n_estimators = request.form.get('n_estimators', '')
        learning_rate = request.form.get('learning_rate', '')
        max_depth = request.form.get('max_depth', '')
        
        param_args = f"--n_estimators {n_estimators} "
        if learning_rate:
            param_args += f"--learning_rate {learning_rate} "
        if max_depth:
            param_args += f"--max_depth {max_depth} "
    
    else:
        flash(f"Loại mô hình {model_type} không được hỗ trợ", 'danger')
        return redirect(url_for('index'))
    
    # Thêm các tham số chung
    samples = request.form.get('samples', '1000')
    features = request.form.get('features', '20')
    cv = request.form.get('cv', '5')
    
    params.update({
        'samples': int(samples),
        'features': int(features),
        'cv': int(cv)
    })
    
    # Xây dựng câu lệnh đầy đủ
    command = f"python custom_hyperparam_tuning.py --model {model_type} {param_args} --samples {samples} --features {features} --cv {cv}"
    
    success, output, result_file = run_tuning_command(command, params)
    
    if success:
        flash('Tuning tùy chỉnh thành công', 'success')
    else:
        flash(f'Tuning thất bại: {output}', 'danger')
    
    return redirect(url_for('index'))

# Route xem chi tiết kết quả
@app.route('/view_result/<filename>')
def view_result(filename):
    """Hiển thị chi tiết kết quả tuning từ file"""
    file_path = os.path.join(TUNING_RESULTS_DIR, filename)
    
    if not os.path.exists(file_path):
        flash(f'File kết quả không tồn tại: {filename}', 'danger')
        return redirect(url_for('index'))
    
    try:
        with open(file_path, 'r') as f:
            result = json.load(f)
        return render_template('result_detail.html', result=result, filename=filename)
    except Exception as e:
        flash(f'Lỗi khi đọc file kết quả: {str(e)}', 'danger')
        return redirect(url_for('index'))

# Route để dự đoán
@app.route('/predict', methods=['POST'])
def predict():
    """Thực hiện dự đoán từ form"""
    if request.method == 'POST':
        features = request.form.get('features')
        
        try:
            # Kiểm tra xem có best model không
            model_info_path = os.path.join(MODELS_DIR, 'model_info.json')
            if not os.path.exists(model_info_path):
                flash('Chưa có model để dự đoán. Vui lòng train model trước.', 'danger')
                return redirect(url_for('index'))
                
            # Đọc thông tin model
            with open(model_info_path, 'r') as f:
                model_info = json.load(f)
            
            # Giả lập kết quả dự đoán
            result = {
                'predicted_class': 1,
                'probability': 0.85,
                'model_type': model_info.get('model_type', 'unknown'),
                'model_accuracy': model_info.get('accuracy', 'N/A'),
                'model_f1': model_info.get('f1_score', 'N/A')
            }
            
            return render_template('prediction_result.html', result=result, features=features)
        except Exception as e:
            flash(f'Lỗi khi thực hiện dự đoán: {str(e)}', 'danger')
            return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, port=5000) 