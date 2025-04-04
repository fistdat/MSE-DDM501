// Main JavaScript for MLOps UI

document.addEventListener('DOMContentLoaded', function() {
    console.log('MLOps UI loaded successfully');

    // Validate form before submission
    const tuningForms = document.querySelectorAll('.tuning-form form');
    tuningForms.forEach(form => {
        form.addEventListener('submit', function(event) {
            const nSamples = parseInt(form.querySelector('input[name="n_samples"]').value);
            const nFeatures = parseInt(form.querySelector('input[name="n_features"]').value);
            
            if (nSamples <= 0 || nFeatures <= 0) {
                event.preventDefault();
                alert('Số lượng mẫu và tính năng phải lớn hơn 0');
            }
            
            if (nSamples > 10000) {
                const confirm = window.confirm('Số lượng mẫu lớn có thể làm quá trình huấn luyện chậm hơn. Bạn có muốn tiếp tục?');
                if (!confirm) {
                    event.preventDefault();
                }
            }
        });
    });

    // Fetch latest models from API
    fetchModels();

    // Setup prediction form if exists
    setupPredictionForm();
});

// Fetch models from API
function fetchModels() {
    const modelListEl = document.getElementById('model-list');
    if (!modelListEl) return;

    fetch('/api/models')
        .then(response => response.json())
        .then(data => {
            if (data && data.length > 0) {
                const modelListHTML = data.map(model => `
                    <div class="col-md-6 mb-3">
                        <div class="card h-100">
                            <div class="card-body">
                                <h5 class="card-title">${model.name}</h5>
                                <h6 class="card-subtitle mb-2 text-muted">Version: ${model.version}</h6>
                                <p class="card-text">Accuracy: ${model.accuracy.toFixed(4)}</p>
                            </div>
                            <div class="card-footer">
                                <button class="btn btn-primary btn-sm use-model" data-model="${model.name}">Sử dụng mô hình này</button>
                            </div>
                        </div>
                    </div>
                `).join('');
                
                modelListEl.innerHTML = modelListHTML;
                
                // Add event listeners to "Use Model" buttons
                document.querySelectorAll('.use-model').forEach(btn => {
                    btn.addEventListener('click', function() {
                        const modelName = this.getAttribute('data-model');
                        document.getElementById('selected-model').textContent = modelName;
                        
                        // Scroll to prediction form
                        document.getElementById('prediction-section').scrollIntoView({
                            behavior: 'smooth'
                        });
                    });
                });
            } else {
                modelListEl.innerHTML = `<div class="col-12"><div class="alert alert-info">Không tìm thấy mô hình nào. Hãy huấn luyện mô hình trước.</div></div>`;
            }
        })
        .catch(error => {
            console.error('Error fetching models:', error);
            modelListEl.innerHTML = `<div class="col-12"><div class="alert alert-danger">Lỗi khi tải danh sách mô hình: ${error.message}</div></div>`;
        });
}

// Setup prediction form
function setupPredictionForm() {
    const predictionForm = document.getElementById('prediction-form');
    if (!predictionForm) return;

    predictionForm.addEventListener('submit', function(event) {
        event.preventDefault();
        
        // Get form data
        const formData = new FormData(predictionForm);
        const jsonData = {};
        
        for (let [key, value] of formData.entries()) {
            jsonData[key] = parseFloat(value) || value;
        }
        
        // Add selected model
        const selectedModel = document.getElementById('selected-model').textContent;
        if (!selectedModel || selectedModel === 'None') {
            alert('Vui lòng chọn một mô hình để dự đoán');
            return;
        }
        
        jsonData.model = selectedModel;
        
        // Submit prediction request
        fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(jsonData)
        })
        .then(response => response.json())
        .then(result => {
            displayPredictionResult(result);
        })
        .catch(error => {
            console.error('Error making prediction:', error);
            alert('Lỗi khi thực hiện dự đoán: ' + error.message);
        });
    });
}

// Display prediction result
function displayPredictionResult(result) {
    const resultContainer = document.getElementById('prediction-result');
    if (!resultContainer) return;
    
    // Create result HTML
    let resultHTML = `
        <div class="prediction-result">
            <h4>Kết quả dự đoán</h4>
            <div class="badge bg-${result.class === 'positive' ? 'success' : 'danger'} prediction-badge">
                ${result.class === 'positive' ? 'Tích cực' : 'Tiêu cực'}
            </div>
            <div class="row mt-3">
                <div class="col-12">
                    <h5>Xác suất:</h5>
                    <div class="progress mb-2" style="height: 25px;">
                        <div class="progress-bar bg-success" role="progressbar" 
                             style="width: ${result.prediction[0] * 100}%;" 
                             aria-valuenow="${result.prediction[0] * 100}" aria-valuemin="0" aria-valuemax="100">
                            ${(result.prediction[0] * 100).toFixed(2)}%
                        </div>
                    </div>
                    <div class="progress" style="height: 25px;">
                        <div class="progress-bar bg-danger" role="progressbar" 
                             style="width: ${result.prediction[1] * 100}%;" 
                             aria-valuenow="${result.prediction[1] * 100}" aria-valuemin="0" aria-valuemax="100">
                            ${(result.prediction[1] * 100).toFixed(2)}%
                        </div>
                    </div>
                </div>
            </div>
            <div class="mt-3">
                <p class="text-muted">Mô hình: ${document.getElementById('selected-model').textContent} (Phiên bản: ${result.model_version})</p>
            </div>
        </div>
    `;
    
    resultContainer.innerHTML = resultHTML;
    
    // Scroll to result
    resultContainer.scrollIntoView({ behavior: 'smooth' });
} 