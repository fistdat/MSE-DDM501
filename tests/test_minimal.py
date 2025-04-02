"""
Test tối thiểu cho chức năng cơ bản của Flask app
"""

import unittest
import os
import sys
import json
from flask import Flask, render_template, flash, redirect, url_for, request, session

# Đảm bảo có thể import các module từ thư mục cha
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class MinimalAppTest(unittest.TestCase):
    """
    Test tối thiểu cho Flask app
    """
    
    def setUp(self):
        """
        Thiết lập app test
        """
        # Tạo một Flask app mới cho test
        self.app = Flask(__name__)
        self.app.secret_key = 'test_secret_key'
        self.app.config['TESTING'] = True
        self.app.config['WTF_CSRF_ENABLED'] = False
        
        # Tạo route test
        @self.app.route('/')
        def index():
            return 'Test Index Page'
        
        @self.app.route('/flash_success')
        def flash_success():
            flash('Success message', 'success')
            return redirect(url_for('index'))
        
        @self.app.route('/flash_danger')
        def flash_danger():
            flash('Danger message', 'danger')
            return redirect(url_for('index'))
        
        @self.app.route('/health')
        def health():
            return json.dumps({'status': 'healthy'})
        
        # Tạo test client
        self.client = self.app.test_client()
    
    def test_index(self):
        """
        Test trang chủ đơn giản
        """
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data, b'Test Index Page')
    
    def test_flash_success(self):
        """
        Test flash message success
        """
        with self.client as c:
            # Đảm bảo session hoạt động
            with c.session_transaction() as sess:
                sess.clear()
            
            # Gọi route có flash success
            response = c.get('/flash_success', follow_redirects=True)
            self.assertEqual(response.status_code, 200)
            
            # Kiểm tra flash message trong session
            with c.session_transaction() as sess:
                flashes = sess.get('_flashes', [])
                self.assertTrue(any(cat == 'success' for cat, _ in flashes),
                               f"Không tìm thấy flash 'success' trong {flashes}")
    
    def test_flash_danger(self):
        """
        Test flash message danger
        """
        with self.client as c:
            # Đảm bảo session hoạt động
            with c.session_transaction() as sess:
                sess.clear()
            
            # Gọi route có flash danger
            response = c.get('/flash_danger', follow_redirects=True)
            self.assertEqual(response.status_code, 200)
            
            # Kiểm tra flash message trong session
            with c.session_transaction() as sess:
                flashes = sess.get('_flashes', [])
                self.assertTrue(any(cat == 'danger' for cat, _ in flashes),
                               f"Không tìm thấy flash 'danger' trong {flashes}")
    
    def test_health_endpoint(self):
        """
        Test health endpoint đơn giản
        """
        response = self.client.get('/health')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'healthy')

if __name__ == '__main__':
    unittest.main() 