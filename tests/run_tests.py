#!/usr/bin/env python
"""
Script chạy tất cả các test với báo cáo bao phủ
"""

import os
import sys
import unittest
import coverage
import argparse
from termcolor import colored

def setup_test_environment():
    """
    Chuẩn bị môi trường test
    """
    # Đảm bảo thư mục tuning_results tồn tại
    if not os.path.exists('tuning_results'):
        os.makedirs('tuning_results')
    
    # Đảm bảo thư mục templates tồn tại
    if not os.path.exists('templates'):
        os.makedirs('templates')

def fix_index_template():
    """
    Sửa lỗi trong template index.html nếu cần
    """
    try:
        from test_error_fix import fix_template_index_html
        fix_template_index_html()
        print(colored("✓ Đã sửa lỗi template index.html", "green"))
    except:
        print(colored("! Không thể sửa lỗi template index.html", "yellow"))

def run_tests(verbosity=2, coverage_report=False):
    """
    Tìm và chạy tất cả các test
    
    Args:
        verbosity (int): Mức độ chi tiết của báo cáo
        coverage_report (bool): Có tạo báo cáo bao phủ không
    """
    print(colored("\n=== BẮT ĐẦU CHẠY TEST ===", "cyan"))
    
    # Thiết lập coverage
    cov = None
    if coverage_report:
        cov = coverage.Coverage(source=['app.py', 'templates'])
        cov.start()
    
    # Tìm tất cả test case
    loader = unittest.TestLoader()
    
    # Tìm các test từ các file test_*.py
    test_files = []
    for file in os.listdir('.'):
        if file.startswith('test_') and file.endswith('.py'):
            test_files.append(file[:-3])  # Cắt đuôi .py
    
    suite = unittest.TestSuite()
    
    # Thêm test từ mỗi file
    for test_file in test_files:
        try:
            tests = loader.loadTestsFromName(test_file)
            suite.addTests(tests)
            print(colored(f"✓ Đã tìm thấy {tests.countTestCases()} test case trong {test_file}", "green"))
        except Exception as e:
            print(colored(f"! Lỗi khi tải test từ {test_file}: {str(e)}", "red"))
    
    # Chạy tất cả test
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    # Tạo báo cáo coverage nếu được yêu cầu
    if coverage_report and cov:
        cov.stop()
        cov.save()
        print(colored("\n=== BÁO CÁO BAO PHỦ ===", "cyan"))
        cov.report()
        
        # Tạo báo cáo HTML
        cov.html_report(directory='coverage_html')
        print(colored(f"Đã tạo báo cáo HTML chi tiết trong thư mục: coverage_html", "green"))
    
    # In bản tóm tắt
    print(colored("\n=== KẾT QUẢ TEST ===", "cyan"))
    print(f"Runs: {result.testsRun}")
    print(colored(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}", "green"))
    
    if result.failures:
        print(colored(f"Failures: {len(result.failures)}", "red"))
    else:
        print(colored(f"Failures: 0", "green"))
    
    if result.errors:
        print(colored(f"Errors: {len(result.errors)}", "red"))
    else:
        print(colored(f"Errors: 0", "green"))
    
    # Return 0 nếu tất cả test đều pass, 1 nếu có bất kỳ test nào fail
    return 0 if result.wasSuccessful() else 1

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Chạy test cho ứng dụng MLOps Flask")
    parser.add_argument('-v', '--verbose', action='store_true', 
                       help='Hiển thị kết quả test chi tiết')
    parser.add_argument('-c', '--coverage', action='store_true', 
                       help='Tạo báo cáo bao phủ code')
    parser.add_argument('-f', '--fix-templates', action='store_true', 
                       help='Sửa lỗi trong template trước khi chạy test')
    
    args = parser.parse_args()
    
    # Thiết lập môi trường test
    setup_test_environment()
    
    # Sửa template nếu yêu cầu
    if args.fix_templates:
        fix_index_template()
    
    # Chạy test
    verbosity = 2 if args.verbose else 1
    exit_code = run_tests(verbosity=verbosity, coverage_report=args.coverage)
    
    # Kết thúc với mã tương ứng
    sys.exit(exit_code) 