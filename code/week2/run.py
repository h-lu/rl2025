#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
运行Streamlit应用程序的入口脚本
"""

import os
import sys
import subprocess

def main():
    """
    主函数，运行Streamlit应用
    """
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 应用程序主文件路径
    app_path = os.path.join(current_dir, "app.py")
    
    print(f"启动交互式课件应用...")
    print(f"应用路径: {app_path}")
    print("请在浏览器中访问显示的URL (通常是 http://localhost:8501)")
    
    # 启动Streamlit应用
    subprocess.run(["streamlit", "run", app_path], check=True)

if __name__ == "__main__":
    main() 