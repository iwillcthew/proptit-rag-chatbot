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

# Hiển thị gợi ý nếu chưa có tin nhắn nào
if len(st.session_state.messages) == 0:
    st.markdown("### 💡 Gợi ý câu hỏi:")
    
    # Tạo các cột để hiển thị gợi ý đẹp hơn
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📍 CLB có những ban nào?", use_container_width=True):
            st.session_state.suggested_query = "CLB có những ban nào?"
            st.rerun()
        if st.button("👥 Làm thế nào để gia nhập CLB?", use_container_width=True):
            st.session_state.suggested_query = "Làm thế nào để gia nhập CLB?"
            st.rerun()
        if st.button("📅 CLB có những hoạt động gì?", use_container_width=True):
            st.session_state.suggested_query = "CLB có những hoạt động gì?"
            st.rerun()
        if st.button("🏆 CLB đã đạt được những thành tích gì?", use_container_width=True):
            st.session_state.suggested_query = "CLB đã đạt được những thành tích gì?"
            st.rerun()
    
    with col2:
        if st.button("💻 CLB có những khóa học lập trình nào?", use_container_width=True):
            st.session_state.suggested_query = "CLB có khóa học lập trình nào?"
            st.rerun()
        if st.button("🔧 CLB dạy những ngôn ngữ lập trình gì?", use_container_width=True):
            st.session_state.suggested_query = "CLB dạy những ngôn ngữ lập trình gì?"
            st.rerun()
        if st.button("🎯 Quy trình tuyển thành viên như thế nào?", use_container_width=True):
            st.session_state.suggested_query = "Quy trình tuyển thành viên như thế nào?"
            st.rerun()
        if st.button("📚 Quyền lợi khi là thành viên chính thức?", use_container_width=True):
            st.session_state.suggested_query = "Quyền lợi khi trở thành thành viên chính thức là gì?"
            st.rerun()
    
    st.markdown("---")

# Hiển thị lịch sử chat
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=message.get("avatar")):
        st.markdown(message["content"])

# --- CHAT LOGIC ---
# Luôn hiển thị ô chat input ở cuối trang
prompt_from_input = st.chat_input("Bạn có câu hỏi gì về CLB Lập trình PTIT?")

# Xử lý nếu có một gợi ý được chọn
prompt_from_suggestion = None
if "suggested_query" in st.session_state:
    prompt_from_suggestion = st.session_state.suggested_query
    del st.session_state.suggested_query

# Ưu tiên xử lý gợi ý trước, sau đó đến input của người dùng
prompt = prompt_from_suggestion or prompt_from_input

if prompt:
    # Thêm câu hỏi của người dùng vào UI
    st.session_state.messages.append({"role": "user", "content": prompt, "avatar": "🧑‍💻"})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    # Tạo phản hồi từ bot
    with st.chat_message("assistant", avatar="🤖"):
        message_placeholder = st.empty()
        full_response = ""
        
        with st.spinner("🤔 Bot đang suy nghĩ..."):
            # Lấy lịch sử chat để đưa vào model - chỉ lấy các cặp hội thoại hoàn chỉnh
            chat_history_for_model = []
            # Lấy tất cả tin nhắn TRỪ tin nhắn cuối cùng (là câu hỏi hiện tại)
            messages_for_history = st.session_state.messages[:-1]
            
            # Nhóm thành các cặp (user, assistant)
            for i in range(0, len(messages_for_history), 2):
                if i + 1 < len(messages_for_history) and messages_for_history[i]["role"] == "user" and messages_for_history[i+1]["role"] == "assistant":
                    chat_history_for_model.append({
                        "user": messages_for_history[i]["content"],
                        "assistant": messages_for_history[i+1]["content"]
                    })
            
            # Gọi pipeline để lấy câu trả lời và logs
            pipeline: RAGPipeline = st.session_state.rag_pipeline
            response, logs = pipeline.get_response(prompt, chat_history_for_model)
            
            # Lưu logs vào session state để trang Logs có thể truy cập
            st.session_state.logs = logs

        # Hiệu ứng gõ chữ với định dạng markdown
        # Tách theo từng từ nhưng giữ nguyên cấu trúc xuống dòng và khoảng trắng
        words = []
        current_word = ""
        for char in response:
            if char in [' ', '\n', '\t']:
                if current_word:
                    words.append(current_word)
                    current_word = ""
                words.append(char)  # Giữ nguyên khoảng trắng và xuống dòng
            else:
                current_word += char
        if current_word:
            words.append(current_word)
        
        for word in words:
            full_response += word
            time.sleep(0.03)  # Tăng tốc độ gõ một chút
            # Cập nhật lại giao diện ở mỗi bước để tạo hiệu ứng gõ chữ
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)

    # Thêm phản hồi của bot vào lịch sử chat
    st.session_state.messages.append({"role": "assistant", "content": full_response, "avatar": "🤖"})
    
    # Nếu câu hỏi đến từ gợi ý, chạy lại script để xóa trạng thái và chỉ hiển thị chat input
    if prompt_from_suggestion:
        st.rerun()
