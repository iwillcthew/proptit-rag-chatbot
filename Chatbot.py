# Chatbot.py
# Đây là file chính của ứng dụng Streamlit, được đổi tên từ app_test.py
import streamlit as st
from pipeline import RAGPipeline
import time

# --- PAGE CONFIG ---
st.set_page_config(page_title="PROPTIT RAG Chatbot", page_icon="🤖", layout="centered")

# --- SESSION STATE ---
# Khởi tạo pipeline trong session state để tránh tải lại model mỗi lần re-run
if "rag_pipeline" not in st.session_state:
    with st.spinner("🚀 Khởi động hệ thống, vui lòng chờ..."):
        st.session_state.rag_pipeline = RAGPipeline()

# Khởi tạo lịch sử chat và logs
if "messages" not in st.session_state:
    st.session_state.messages = []
if "logs" not in st.session_state:
    st.session_state.logs = {}

# --- UI ---
# Tiêu đề và logo
st.image("logo pro@8x.png", width=100)
st.title("PROPTIT RAG Chatbot")
st.caption("Trợ lý AI thông minh của CLB Lập trình PTIT")

# Hiển thị lịch sử chat
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=message.get("avatar")):
        st.markdown(message["content"])

# --- CHAT LOGIC ---
if prompt := st.chat_input("Bạn có câu hỏi gì về CLB Lập trình PTIT?"):
    # Thêm câu hỏi của người dùng vào UI
    st.session_state.messages.append({"role": "user", "content": prompt, "avatar": "🧑‍💻"})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    # Tạo phản hồi từ bot
    with st.chat_message("assistant", avatar="🤖"):
        message_placeholder = st.empty()
        full_response = ""
        
        with st.spinner("🤔 Bot đang suy nghĩ..."):
            # Lấy lịch sử chat để đưa vào model
            chat_history_for_model = [
                {"user": msg["content"], "assistant": st.session_state.messages[i+1]["content"]}
                for i, msg in enumerate(st.session_state.messages[:-1])
                if msg["role"] == "user"
            ]
            
            # Gọi pipeline để lấy câu trả lời và logs
            pipeline: RAGPipeline = st.session_state.rag_pipeline
            response, logs = pipeline.get_response(prompt, chat_history_for_model)
            
            # Lưu logs vào session state để trang Logs có thể truy cập
            st.session_state.logs = logs

        # Hiệu ứng gõ chữ
        for chunk in response.split():
            full_response += chunk + " "
            time.sleep(0.05)
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)

    # Thêm phản hồi của bot vào lịch sử chat
    st.session_state.messages.append({"role": "assistant", "content": full_response, "avatar": "🤖"})
