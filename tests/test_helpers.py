"""
Các hàm helper hỗ trợ test cho ứng dụng Flask
"""

from flask import template_rendered, session, get_flashed_messages
from contextlib import contextmanager
import re

@contextmanager
def captured_templates(app):
    """
    Context manager để bắt các template đã render
    """
    recorded = []
    def record(sender, template, context, **extra):
        recorded.append((template, context))
    template_rendered.connect(record, app)
    try:
        yield recorded
    finally:
        template_rendered.disconnect(record, app)

def mock_flash_messages(app, messages=None):
    """
    Tạo flash message giả lập để test
    
    Args:
        app: Flask app
        messages: Dict các flash message {category: message}
    """
    if messages is None:
        messages = {}
    
    original_flash = app.flash
    
    def mock_flash(message, category='message'):
        # Lưu message vào session như Flask thực sự làm
        session.setdefault('_flashes', []).append((category, message))
    
    app.flash = mock_flash
    return original_flash

def check_flash_in_response(response_data, category=None, message_pattern=None):
    """
    Kiểm tra flash message trong HTML response
    
    Args:
        response_data: Dữ liệu phản hồi từ request
        category: Loại flash message cần kiểm tra (success, danger, ...)
        message_pattern: Chuỗi regex để so khớp với nội dung thông báo
        
    Returns:
        True nếu tìm thấy flash message phù hợp, False nếu không tìm thấy
    """
    response_text = response_data.decode('utf-8')
    
    # Tìm tất cả các flash message trong HTML response
    if category:
        pattern = rf'<div[^>]*class="[^"]*alert[^"]*alert-{category}[^"]*"[^>]*>(.*?)</div>'
    else:
        pattern = r'<div[^>]*class="[^"]*alert[^"]*"[^>]*>(.*?)</div>'
    
    flash_divs = re.findall(pattern, response_text, re.DOTALL)
    
    if not flash_divs:
        return False
    
    # Nếu không cần kiểm tra message pattern, chỉ cần tìm thấy div là đủ
    if not message_pattern:
        return True
    
    # Kiểm tra xem có nội dung phù hợp không
    for content in flash_divs:
        if re.search(message_pattern, content, re.DOTALL):
            return True
    
    return False 