# pipeline.py
import os
import json
import time
import hashlib
from typing import List, Dict, Any, Tuple, Union
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

# Tải biến môi trường từ file .env
load_dotenv()

# --- 1. NVIDIA NIM CLIENT ---
class NIMClient:
    """Client để tương tác với NVIDIA NIM qua API tương thích OpenAI."""
    def __init__(self, api_key: str = "", base_url: str = "", model: str = "meta/llama-3.1-405b-instruct"):
        api_key = api_key or os.environ.get("NIM_API_KEY", "")
        base_url = base_url or os.environ.get("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
        if not api_key:
            print("⚠️ NIM_API_KEY không được tìm thấy. Chatbot sẽ chạy ở chế độ demo.")
            # Thay vì raise error, chúng ta sẽ sử dụng một API key demo hoặc fallback
            api_key = "demo_key"  # Placeholder - bạn có thể điều chỉnh logic này
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.is_demo_mode = api_key == "demo_key"

    def chat(self, messages, temperature: float = 0.1, max_tokens: int = 1024, top_p: float = 0.7, **kwargs) -> str:
        """Gửi yêu cầu chat và nhận phản hồi."""
        if self.is_demo_mode:
            return "Xin lỗi, chatbot đang ở chế độ demo do thiếu cấu hình API key. Vui lòng liên hệ admin để được hỗ trợ."
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    stream=False,
                    **kwargs
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"❌ API call failed after {max_retries} attempts: {e}")
                    return f"Xin lỗi, đã có lỗi xảy ra khi xử lý yêu cầu của bạn: {str(e)}"
                print(f"⚠️ API call failed (attempt {attempt + 1}/{max_retries}), retrying...")
                time.sleep(2 ** attempt)
        return "Xin lỗi, không thể xử lý yêu cầu của bạn lúc này."

# --- 2. EMBEDDING MODEL ---
class Embeddings:
    """Tải và sử dụng model embedding từ HuggingFace."""
    def __init__(self, model_path="iwillcthew/vietnamese-embedding-PROPTIT-domain-ft"):
        print(f"🚀 Loading embedding model: {model_path}")
        self.model = SentenceTransformer(model_path)
        self.model.max_seq_length = 2048
        print("✅ Embedding model loaded.")

    def encode(self, text: Union[str, List[str]]):
        """Tạo embedding cho văn bản."""
        return self.model.encode(text)

# --- 3. VECTOR DATABASE ---
class VectorDatabaseAtlas:
    """Kết nối và truy vấn MongoDB Atlas Vector Search."""
    def __init__(self, mongodb_uri: str = ""):
        mongodb_uri = mongodb_uri or os.environ.get("MONGODB_URI", "")
        if not mongodb_uri:
            print("⚠️ MONGODB_URI không được tìm thấy. Vector search sẽ không khả dụng.")
            self.client = None
            self.db = None
            self.collection = None
            self.is_demo_mode = True
            return
        
        try:
            print("🔗 Connecting to MongoDB Atlas...")
            self.client = MongoClient(mongodb_uri)
            self.db = self.client.get_database("vector_db")
            self.collection = self.db["information"]
            self.is_demo_mode = False
            print("✅ MongoDB Atlas connected.")
        except Exception as e:
            print(f"⚠️ Failed to connect to MongoDB: {e}")
            self.client = None
            self.db = None
            self.collection = None
            self.is_demo_mode = True

    def vector_search(self, query_vector: List[float], limit: int = 30) -> List[Dict]:
        """Thực hiện vector search."""
        if self.is_demo_mode or self.collection is None:
            print("⚠️ Vector search không khả dụng trong chế độ demo.")
            return []
        
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "queryVector": query_vector,
                    "path": "embedding",
                    "numCandidates": max(200, limit * 10),
                    "limit": limit
                }
            }
        ]
        try:
            return list(self.collection.aggregate(pipeline, maxTimeMS=30000))
        except Exception as e:
            print(f"⚠️ Vector search failed: {e}")
            return []

# --- 4. LLM RERANKER ---
class LLMReranker:
    """Sử dụng LLM để re-rank kết quả tìm kiếm."""
    def __init__(self, nim_client: NIMClient):
        self.nim = nim_client

    def rerank(self, query: str, results: List[Dict[str, Any]], top_k: int = 7) -> Tuple[List[Dict[str, Any]], List[int]]:
        """Re-rank và chọn ra top_k tài liệu liên quan nhất."""
        if not results:
            return [], []

        doc_ids = [int(str(r.get("title", "Document 0")).split()[-1]) for r in results]
        docs_text = "".join(f"[Document {doc_id}]\n{result.get('information', '')}\n\n" for doc_id, result in zip(doc_ids, results))

        rerank_prompt = f"""Bạn là một chuyên gia Re-ranker AI. Nhiệm vụ của bạn là sắp xếp lại và chọn ra `{top_k}` tài liệu liên quan nhất từ danh sách cho trước để trả lời câu hỏi.
CÂU HỎI: {query}
DANH SÁCH TÀI LIỆU:
{docs_text}
---
YÊU CẦU:
1. Phân tích câu hỏi và đánh giá từng tài liệu.
2. Sắp xếp các tài liệu theo mức độ liên quan giảm dần và chọn ra {top_k} tài liệu tốt nhất.
3. Trả về CHỈ một JSON object chứa danh sách các ID tài liệu đã được sắp xếp lại.
FORMAT OUTPUT (TUÂN THỦ NGHIÊM NGẶT):
{{"selected_indices": [15, 3, 27, 8, 12, 1, 5]}}
"""
        messages = [
            {"role": "system", "content": f"Bạn là một chuyên gia Re-ranker AI. Chỉ trả về JSON object với format yêu cầu."},
            {"role": "user", "content": rerank_prompt}
        ]

        try:
            response = self.nim.chat(messages, temperature=0.0, max_tokens=200)
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start != -1 and json_end != -1:
                json_str = response[json_start:json_end]
                parsed = json.loads(json_str)
                selected_doc_ids = parsed.get('selected_indices', [])
            else:
                raise ValueError("Invalid JSON response")

            reranked_results = []
            seen_doc_ids = set()
            original_results_map = {int(str(r.get("title", "Document 0")).split()[-1]): r for r in results}

            for doc_id in selected_doc_ids:
                if doc_id in original_results_map and doc_id not in seen_doc_ids:
                    reranked_results.append(original_results_map[doc_id])
                    seen_doc_ids.add(doc_id)
            
            # Fill remaining spots if LLM returns fewer than top_k
            if len(reranked_results) < top_k:
                for doc_id, result in original_results_map.items():
                    if doc_id not in seen_doc_ids:
                        reranked_results.append(result)
                        if len(reranked_results) == top_k:
                            break
            
            final_results = reranked_results[:top_k]
            final_doc_ids = [int(str(r.get("title", "Document 0")).split()[-1]) for r in final_results]
            return final_results, final_doc_ids
        except Exception as e:
            print(f"❌ Reranking failed: {e}. Falling back to original order.")
            return results[:top_k], doc_ids[:top_k]

# --- 5. QUERY CLASSIFIER ---
class QueryClassifier:
    """Phân loại câu hỏi để quyết định có sử dụng RAG hay không."""
    def __init__(self, nim_client: NIMClient):
        self.nim = nim_client
        self.proptit_keywords = [
            "proptit", "clb", "câu lạc bộ", "lập trình ptit", "ptit",
            "tuyển thành viên", "ctv", "cộng tác viên", "thành viên",
            "team", "ban", "dự án", "đào tạo", "training", "phỏng vấn",
            "sự kiện", "event", "workshop", "cuộc thi"
        ]

    def is_proptit_related(self, query: str) -> bool:
        """Kiểm tra nhanh bằng từ khóa."""
        query_lower = query.lower()
        if any(keyword in query_lower for keyword in self.proptit_keywords):
            return True
        
        # Nếu không có từ khóa, dùng LLM để chắc chắn
        prompt = f"""Bạn là một bộ phân loại văn bản. Nhiệm vụ của bạn là xác định xem câu hỏi của người dùng có liên quan đến "Câu lạc bộ Lập trình PTIT (ProPTIT)" hay không.
Chỉ trả lời "yes" hoặc "no".
Câu hỏi: "{query}"
Câu hỏi này có liên quan đến CLB Lập trình PTIT không?
"""
        messages = [
            {"role": "system", "content": "Bạn là một bộ phân loại văn bản. Chỉ trả lời 'yes' hoặc 'no'."},
            {"role": "user", "content": prompt}
        ]
        try:
            response = self.nim.chat(messages, temperature=0.0, max_tokens=5).lower()
            return "yes" in response
        except Exception as e:
            print(f"⚠️ Query classification failed: {e}. Defaulting to RAG.")
            return True # Mặc định dùng RAG nếu có lỗi

# --- 6. RAG PIPELINE ---
class RAGPipeline:
    """Orchestrates the entire RAG pipeline."""
    def __init__(self):
        self.nim_client = NIMClient()
        self.embedding_model = Embeddings()
        self.vector_db = VectorDatabaseAtlas()
        self.reranker = LLMReranker(self.nim_client)
        self.classifier = QueryClassifier(self.nim_client)
        self.rag_prompt_template = """Bạn là một trợ lý AI chuyên cung cấp thông tin về Câu lạc bộ Lập trình ProPTIT.
Bạn sẽ nhận được dữ liệu ngữ cảnh (context) từ một hệ thống Retrieval-Augmented Generation (RAG) chứa các thông tin chính xác về CLB.
NGUYÊN TẮC TRẢ LỜI BẮT BUỘC:
1. CHỈ sử dụng thông tin từ context được cung cấp để trả lời. KHÔNG được thêm thông tin ngoài context.
2. Trả lời CHÍNH XÁC, và TRỰC TIẾP vào câu hỏi.
3. KHÔNG được thêm lời chào hỏi, cảm ơn, hoặc câu xã giao không cần thiết.
4. KHÔNG được nói "Xin lỗi", "Tôi không biết", "Không có thông tin" - PHẢI trả lời dựa trên context có sẵn.
5. Nếu context không đủ, hãy suy luận LOGIC từ thông tin có sẵn mà KHÔNG bịa thêm.
6. Tập trung trả lời CÂU HỎI CHÍNH, bỏ qua thông tin không liên quan.
7. Sử dụng ngôn ngữ tự nhiên, dễ hiểu, phù hợp với phong cách trả lời của con người.
8. Ưu tiên xưng là "CLB" khi nói về tổ chức.
9. Không được thêm câu dẫn như "Dựa trên thông tin từ ngữ cảnh, dưới đây là...", trả lời trực tiếp vào câu hỏi
{context}
Dựa vào thông tin trên, hãy trả lời câu hỏi sau:
Câu hỏi: {query}
"""

    def get_response(self, query: str, chat_history: List[Dict[str, str]]) -> Tuple[str, Dict]:
        """
        Main function to get a response for a user query.
        Returns the response string and a log dictionary.
        """
        logs = {"query": query, "rag_enabled": False}

        # Kiểm tra nếu đang ở demo mode
        if hasattr(self.vector_db, 'is_demo_mode') and self.vector_db.is_demo_mode:
            logs["classification"] = "Demo Mode"
            response = "Xin chào! Chatbot đang chạy ở chế độ demo do thiếu cấu hình database. Để sử dụng đầy đủ tính năng RAG, vui lòng cấu hình MONGODB_URI và NIM_API_KEY trong secrets của Hugging Face Space."
            logs["response"] = response
            return response, logs

        if not self.classifier.is_proptit_related(query):
            logs["classification"] = "General Conversation"
            response = self.get_general_response(query, chat_history)
            logs["response"] = response
            return response, logs

        logs["rag_enabled"] = True
        logs["classification"] = "PROPTIT-related"

        # 1. Vector Search
        k_retrieval = 30
        query_embedding = self.embedding_model.encode(query).tolist()
        
        vector_results = self.vector_db.vector_search(query_embedding, limit=k_retrieval)
        
        retrieved_ids = [int(str(r.get("title", "Doc 0")).split()[-1]) for r in vector_results]
        logs["vector_search"] = {
            "retrieved_ids": retrieved_ids,
            "retrieved_count": len(vector_results)
        }

        # Nếu không có kết quả tìm kiếm, trả lời mặc định
        if not vector_results:
            logs["response"] = "Xin lỗi, hiện tại tôi không thể tìm thấy thông tin liên quan đến câu hỏi của bạn. Vui lòng thử lại sau."
            return logs["response"], logs

        # 2. Rerank
        k_final = 7
        reranked_results, reranked_ids = self.reranker.rerank(query, vector_results, top_k=k_final)
        logs["rerank"] = {"reranked_ids": reranked_ids}

        # 3. Generate Response
        context = "\n\n".join([f"Trích đoạn từ tài liệu {reranked_ids[i]}:\n{doc.get('information', '')}" for i, doc in enumerate(reranked_results)])
        
        final_prompt = self.rag_prompt_template.format(context=context, query=query)
        
        messages = self._build_chat_history(final_prompt, chat_history)
        
        response = self.nim_client.chat(messages)
        logs["response"] = response
        logs["final_context"] = context

        return response, logs

    def get_general_response(self, query: str, chat_history: List[Dict[str, str]]) -> str:
        """Generate a response for non-RAG questions."""
        prompt = f"""Bạn là một trợ lý AI thân thiện của CLB Lập trình PTIT.
Hãy trả lời câu hỏi của người dùng một cách tự nhiên.
Câu hỏi: {query}
"""
        messages = self._build_chat_history(prompt, chat_history)
        return self.nim_client.chat(messages, temperature=0.5)

    def _build_chat_history(self, new_prompt: str, chat_history: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Builds the message list for the LLM, including history."""
        messages = [{"role": "system", "content": "Bạn là một trợ lý AI của CLB Lập trình PTIT."}]
        for message in chat_history[-4:]: # Lấy 4 cặp hội thoại gần nhất
            messages.append({"role": "user", "content": message["user"]})
            messages.append({"role": "assistant", "content": message["assistant"]})
        messages.append({"role": "user", "content": new_prompt})
        return messages

# --- Main execution for testing ---
if __name__ == '__main__':
    print("Testing RAG Pipeline...")
    pipeline = RAGPipeline()
    
    test_query = "CLB có những ban nào?"
    print(f"\n--- Testing with query: '{test_query}' ---")
    response, logs = pipeline.get_response(test_query, [])
    print(f"Response: {response}")
    print(f"Logs: {json.dumps(logs, indent=2, ensure_ascii=False)}")

    test_query_general = "Trời hôm nay đẹp quá!"
    print(f"\n--- Testing with query: '{test_query_general}' ---")
    response, logs = pipeline.get_response(test_query_general, [])
    print(f"Response: {response}")
    print(f"Logs: {json.dumps(logs, indent=2, ensure_ascii=False)}")
