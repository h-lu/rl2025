import streamlit as st
import introduction
import q_table_visualization
import dynamic_pricing
import exercises
import utils

# 配置matplotlib字体以支持中文显示
utils.configure_matplotlib_fonts()

def main():
    st.set_page_config(
        page_title="Q-Learning 交互式课件",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.sidebar.title("Q-Learning 交互式课件")
    
    # 使用emoji图标使侧边栏更直观
    menu = st.sidebar.radio(
        "选择章节", 
        ["📚 简介", "📊 Q表可视化", "💰 动态定价案例", "✏️ 交互式练习"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 学习建议")
    st.sidebar.markdown("""
    1. 按照从上到下的顺序学习每个章节
    2. 尝试交互式组件来加深理解
    3. 在练习部分测试你的理解
    """)
    
    if menu == "📚 简介":
        introduction.show()
    elif menu == "📊 Q表可视化":
        q_table_visualization.show()
    elif menu == "💰 动态定价案例":
        dynamic_pricing.show()
    elif menu == "✏️ 交互式练习":
        exercises.show()
        
if __name__ == "__main__":
    main() 