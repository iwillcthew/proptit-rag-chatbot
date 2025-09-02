# pages/2_Logs.py
import streamlit as st

st.set_page_config(page_title="Chi tiết Logs", page_icon="📝", layout="wide")

st.markdown("# 📝 Chi tiết Logs")
st.write(
    """
    Trang này hiển thị chi tiết các bước xử lý của RAG pipeline cho lượt truy vấn cuối cùng.
    """
)

if "logs" not in st.session_state or not st.session_state.logs:
    st.info("Chưa có log nào được ghi lại. Hãy bắt đầu một cuộc trò chuyện trong trang Chatbot.")
else:
    logs = st.session_state.logs
    
    st.write(f"**Câu hỏi:** `{logs.get('query')}`")
    
    classification = logs.get('classification', 'N/A')
    st.write(f"**Phân loại câu hỏi:** `{classification}`")

    if not logs.get('rag_enabled', False):
        st.success("Đây là một cuộc hội thoại thông thường, không sử dụng RAG pipeline.")
        st.write("**Phản hồi của Bot:**")
        st.info(logs.get('response', 'N/A'))
    else:
        st.success("Câu hỏi được xác định liên quan đến PROPTIT, RAG pipeline đã được kích hoạt.")
        
        # Hiển thị các bước của RAG
        with st.expander("1. Vector Search - Lấy tài liệu ứng viên", expanded=True):
            vector_logs = logs.get('vector_search', {})
            st.metric("Số tài liệu từ Vector Search", vector_logs.get('retrieved_count', 0))
            retrieved_ids = vector_logs.get('retrieved_ids', [])
            st.write("**Các ID tài liệu được tìm thấy:**")
            st.json({"retrieved_ids": retrieved_ids})

        with st.expander("2. LLM Reranker - Sắp xếp lại và chọn lọc", expanded=True):
            rerank_logs = logs.get('rerank', {})
            reranked_ids = rerank_logs.get('reranked_ids', [])
            st.metric("Số tài liệu được chọn sau khi Rerank", len(reranked_ids))
            st.write("**Các ID tài liệu được chọn (theo thứ tự ưu tiên):**")
            st.json({"reranked_ids": reranked_ids})

        with st.expander("3. Generation - Tạo câu trả lời", expanded=True):
            st.write("**Ngữ cảnh (Context) cuối cùng được gửi đến LLM:**")
            st.text_area("Context", value=logs.get('final_context', 'N/A'), height=300, disabled=True)
            
            st.write("**Phản hồi cuối cùng của Bot:**")
            st.info(logs.get('response', 'N/A'))
