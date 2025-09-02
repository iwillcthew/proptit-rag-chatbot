#metric.py
import os
import time
import json
import math
import hashlib
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from openai import OpenAI
from dotenv import load_dotenv
import re
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

load_dotenv()

class NIMClient:
    """NVIDIA NIM qua OpenAI-compatible API."""
    def __init__(self,
                 api_key: str = "",
                 base_url: str = "",
                 model: str = "meta/llama-3.1-405b-instruct"):
        api_key = api_key or os.environ.get("NIM_API_KEY", "")
        base_url = base_url or os.environ.get("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
        if not api_key:
            raise RuntimeError("❌ NIM_API_KEY chưa được thiết lập. Set env var NIM_API_KEY hoặc truyền trực tiếp.")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def chat(self,
             messages,
             temperature: float = 0.0,
             max_tokens: int = 512,
             top_p: float = 0.7,
             **kwargs) -> str:
        """Gọi API với retry logic."""
        max_retries = 5
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
                    raise e
                print(f"⚠️ API call failed (attempt {attempt + 1}/{max_retries}), retrying...")
                time.sleep(2 ** attempt)

class Embeddings:
    def __init__(self, model_path="iwillcthew/vietnamese-embedding-PROPTIT-domain-ft"):
        """Load embedding model from HuggingFace."""
        self.model_path = model_path
        self.model = None

        print(f"🚀 Loading embedding model: {model_path}")
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_path)

        self.model.max_seq_length = 2048
        print("✅ Embedding model loaded.")

    def encode(self, text):
        """Encode text(s) for documents."""
        if isinstance(text, str):
            return self.model.encode([text])[0]
        return self.model.encode(text)

    def encode_query(self, query):
        """Encode query for search."""
        return self.model.encode([query])[0]

from pymongo import MongoClient

class VectorDatabaseAtlas:
    def __init__(self, mongodb_uri: str = ""):
        mongodb_uri = mongodb_uri or os.environ.get("MONGODB_URI", "")
        if not mongodb_uri:
            raise RuntimeError("❌ MONGODB_URI not found. Please set it in .env file or pass directly.")
        print("🔗 Connecting to MongoDB Atlas...")
        self.client = MongoClient(mongodb_uri)
        self.db = self.client.get_database("vector_db")
        print("✅ MongoDB Atlas connected.")

    def insert_document(self, collection_name: str, document: dict):
        self.db[collection_name].insert_one(document)

    def query(self, collection_name: str, query_vector: List[float], limit: int = 5):
        num_candidates = max(200, limit * 10)
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "queryVector": query_vector,
                    "path": "embedding",
                    "numCandidates": num_candidates,
                    "limit": limit
                }
            }
        ]

        attempts = 3
        backoff = 1
        max_time_ms = 30000  

        for attempt in range(1, attempts + 1):
            try:
                cursor = self.db[collection_name].aggregate(pipeline, maxTimeMS=max_time_ms)
                results = list(cursor)
                return results
            except Exception as e:
                print(f"⚠️ Vector query attempt {attempt}/{attempts} failed: {e}")
                if attempt == attempts:
                    print("❌ Vector query failed after retries — returning empty list as fallback.")
                    return []
                num_candidates = max(50, num_candidates // 2)
                pipeline[0]["$vectorSearch"]["numCandidates"] = num_candidates
                max_time_ms = max(5000, int(max_time_ms * 0.5))
                time.sleep(backoff)
                backoff *= 2

    def count_documents(self, collection_name: str) -> int:
        return self.db[collection_name].count_documents({})

    def drop_collection(self, collection_name: str):
        self.db[collection_name].drop()

class LLMReranker:
    def __init__(self, nim_client: NIMClient):
        self.nim = nim_client
        print("🚀 Initialized LLM-based reranker")

    def rerank_results(self, query: str, results: List[Dict[str, Any]], top_k: int = None):
        """Use LLM to rerank and select the most relevant documents."""
        if not results or len(results) <= 1:
            return results

        print(f"🔄 Reranking {len(results)} results...")

        doc_ids = []
        for result in results:
            if "title" in result:
                try:
                    doc_id = int(str(result["title"]).split()[-1])
                    doc_ids.append(doc_id)
                except:
                    doc_ids.append(0)
            else:
                doc_ids.append(0)

        docs_text = ""
        for i, (result, doc_id) in enumerate(zip(results, doc_ids)):
            doc_content = result.get("information", "")
            docs_text += f"[Document {doc_id}]\n{doc_content}\n\n"
        rerank_prompt = f"""Bạn là một chuyên gia Re-ranker AI hàng đầu. Nhiệm vụ của bạn là sắp xếp lại và chọn ra `{top_k}` tài liệu liên quan nhất từ một danh sách cho trước để trả lời câu hỏi của người dùng.

BƯỚC RERANK: Phân tích và chọn ra {top_k} tài liệu PHÙ HỢP NHẤT để trả lời câu hỏi.

CÂU HỎI: {query}

DANH SÁCH TÀI LIỆU:
{docs_text}
---
Bạn phải thực hiện theo quy trình sau:
- **Phân tích câu hỏi:** Hiểu rõ ý định, các thực thể chính và thông tin mà người dùng đang tìm kiếm.
- **Đánh giá từng tài liệu:** Đối chiếu nội dung mỗi tài liệu với câu hỏi dựa trên các tiêu chí bên dưới.
- **Sắp xếp và lựa chọn:** Sắp xếp các tài liệu theo thứ tự ưu tiên giảm dần và chọn ra {top_k} tài liệu tốt nhất.

QUY TẮC ĐÁNH GIÁ (Relevance Rubric 0–4, sử dụng để xếp hạng – KHÔNG in ra điểm):
- 4 = Trả lời trực tiếp/bao trùm câu hỏi; thông tin chính xác, cụ thể, có thể dùng độc lập để trả lời.
- 3 = Liên quan mạnh, chứa phần lớn thông tin cần thiết hoặc mảnh ghép quan trọng để hoàn thiện câu trả lời.
- 2 = Liên quan vừa; có thông tin nền/gián tiếp hữu ích (định nghĩa, bối cảnh, ví dụ) để hỗ trợ trả lời.
- 1 = Liên quan yếu; đề cập khái quát đến chủ đề hoặc có từ khóa liên quan đến câu hỏi nhưng thiếu chi tiết hữu ích.
- 0 = Không liên quan/ngoài phạm vi.

HƯỚNG DẪN LỰA CHỌN
1) Ưu tiên tài liệu điểm 4, sau đó 3, rồi 2. Tránh chọn 0–1 trừ khi không có lựa chọn tốt hơn.
2) Với câu hỏi đa khía cạnh, có thể chọn nhiều tài liệu bổ trợ, miễn tổng thể giúp trả lời đầy đủ.
3) Không bị ảnh hưởng bởi thứ tự xuất hiện của tài liệu.
4) Sắp xếp tài liệu theo mức độ liên quan giảm dần (tài liệu liên quan hơn đứng trước) (lấy chính xác {top_k} tài liệu liên quan nhất)
---
YÊU CẦU FORMAT OUTPUT (TUÂN THỦ NGHIÊM NGẶT):
Trả về CHỈ JSON object với format:
{{"selected_indices": [5,12,18,27,33]}}

Trong đó:
- selected_indices: Mảng các ID tài liệu THỰC TẾ (không phải chỉ số mảng)
- Chỉ trả về JSON, không có text khác
- Không giải thích, không thêm ký tự nào khác

Ví dụ output đúng:
{{"selected_indices": [15,3,27,8,12]}}"""

        messages = [
            {
                "role": "system",
                "content": f"Bạn là một chuyên gia Re-ranker AI hàng đầu. Nhiệm vụ của bạn là sắp xếp lại và chọn ra `{top_k}` tài liệu liên quan nhất từ một danh sách tài liệu cho trước để trả lời câu hỏi của người dùng. Luôn trả về JSON format chính xác như yêu cầu. Không thêm text nào khác."
            },
            {
                "role": "user",
                "content": rerank_prompt
            }
        ]

        try:
            response = self.nim.chat(messages, temperature=0.0, max_tokens=1000)

            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start != -1 and json_end != -1:
                json_str = response[json_start:json_end]
                parsed = json.loads(json_str)
                selected_doc_ids = parsed.get('selected_indices', [])
            else:
                parsed = json.loads(response.strip())
                selected_doc_ids = parsed.get('selected_indices', [])

            reranked_results = []
            used_doc_ids = set()

            for selected_id in selected_doc_ids:
                if isinstance(selected_id, int):
                    for i, (result, doc_id) in enumerate(zip(results, doc_ids)):
                        if doc_id == selected_id and selected_id not in used_doc_ids:
                            reranked_results.append(result)
                            used_doc_ids.add(selected_id)
                            break

            if len(reranked_results) < (top_k or len(results)):
                for i, (result, doc_id) in enumerate(zip(results, doc_ids)):
                    if doc_id not in used_doc_ids:
                        reranked_results.append(result)
                        used_doc_ids.add(doc_id)
                        if len(reranked_results) >= (top_k or len(results)):
                            break

            if top_k:
                reranked_results = reranked_results[:top_k]

            print(f"✅ Reranking completed. Selected {len(reranked_results)} documents")
            return reranked_results

        except Exception as e:
            print(f"❌ Reranking failed, using original order")
            return results[:top_k] if top_k else results

# Search cache và các utility functions
_search_cache: Dict[str, Dict[str, Any]] = {}

def clear_search_cache():
    global _search_cache
    _search_cache = {}
    print("🧹 Cleared search cache.")

def extract_doc_ids(results: List[Dict[str, Any]]) -> List[int]:
    out = []
    for r in results:
        if "title" in r:
            try:
                out.append(int(str(r["title"]).split()[-1]))
            except:
                pass
    return out

def vector_search_only(query: str, embedding: Embeddings, vector_db, k: int = 5, nim_client: NIMClient = None):
    """Pure vector search without keyword search."""
    user_emb = embedding.encode_query(query)
    initial_k = 35
    raw = vector_db.query("information", user_emb.tolist(), limit=initial_k)

    if nim_client and len(raw) > k:
        reranked = LLMReranker(nim_client).rerank_results(query, raw, top_k=k)
    else:
        reranked = raw[:k]

    return user_emb, reranked

def get_cached_search_results(query: str, embedding: Embeddings, vector_db, k: int = 7, nim_client: NIMClient = None):
    global _search_cache
    key = hashlib.md5((query + f"_k{k}").encode("utf-8")).hexdigest()
    if key in _search_cache:
        return _search_cache[key]
    user_embedding, results = vector_search_only(query, embedding, vector_db, k, nim_client)
    retrieved_docs = extract_doc_ids(results)
    data = {"user_embedding": user_embedding, "results": results, "retrieved_docs": retrieved_docs}
    _search_cache[key] = data
    return data

# Metric primitives
def dcg_at_k(relevances, k):
    relevances = np.array(relevances)[:k]
    return float(np.sum((2**relevances - 1) / np.log2(np.arange(2, len(relevances) + 2))))

def ndcg_at_k(relevances, k):
    best = dcg_at_k(sorted(relevances, reverse=True), k)
    if best == 0:
        return 0.0
    return dcg_at_k(relevances, k) / best

def cosine01(a, b):
    a = np.array(a); b = np.array(b)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float((np.dot(a, b) / (na * nb) + 1.0) / 2.0)

def _parse_ground_truth_docs(ground_truth_field) -> List[int]:
    docs = []
    if isinstance(ground_truth_field, str):
        for x in ground_truth_field.split(","):
            try:
                docs.append(int(x))
            except:
                pass
    else:
        try:
            docs.append(int(ground_truth_field))
        except:
            pass
    return docs

def hit_k(df_clb, df_train, embedding, vector_db, k=5, nim_client: NIMClient = None):
    hits = 0
    n = len(df_train)

    for idx, row in enumerate(df_train.iterrows()):
        row = row[1]
        query = row["Query"]
        gt_docs = _parse_ground_truth_docs(row["Ground truth document"])

        search = get_cached_search_results(query, embedding, vector_db, k, nim_client)
        retrieved_docs = search["retrieved_docs"][:k]

        hit_docs = [d for d in retrieved_docs if d in gt_docs]
        is_hit = len(hit_docs) > 0
        hits += int(is_hit)

    hit_rate = hits / n if n > 0 else 0.0
    print(f"🎯 Hit@{k}: {hits}/{n} = {hit_rate:.4f}")
    return hit_rate

def recall_k(df_clb, df_train, embedding, vector_db, k=5, nim_client: NIMClient = None):
    total = 0.0
    n = len(df_train)

    for idx, row in enumerate(df_train.iterrows()):
        row = row[1]
        query = row["Query"]
        gt_docs = _parse_ground_truth_docs(row["Ground truth document"])

        search = get_cached_search_results(query, embedding, vector_db, k, nim_client)
        retrieved_docs = search["retrieved_docs"][:k]

        hit_docs = [d for d in retrieved_docs if d in gt_docs]
        recall = len(hit_docs) / len(gt_docs) if gt_docs else 0.0
        total += recall

    mean_recall = total / n if n > 0 else 0.0
    print(f"🎯 Recall@{k}: {mean_recall:.4f}")
    return mean_recall

def precision_k(df_clb, df_train, embedding, vector_db, k=5, nim_client: NIMClient = None):
    total = 0.0
    n = len(df_train)

    for idx, row in enumerate(df_train.iterrows()):
        row = row[1]
        query = row["Query"]
        gt_docs = _parse_ground_truth_docs(row["Ground truth document"])

        search = get_cached_search_results(query, embedding, vector_db, k, nim_client)
        retrieved_docs = search["retrieved_docs"][:k]

        hit_docs = [d for d in retrieved_docs if d in gt_docs]
        precision = len(hit_docs) / k if k > 0 else 0.0
        total += precision

    mean_precision = total / n if n > 0 else 0.0
    print(f"🎯 Precision@{k}: {mean_precision:.4f}")
    return mean_precision

def f1_k(df_clb, df_train, embedding, vector_db, k=5, nim_client: NIMClient = None):
    total_f1 = 0.0
    n = len(df_train)

    for idx, row in enumerate(df_train.iterrows()):
        row = row[1]
        query = row["Query"]
        gt_docs = _parse_ground_truth_docs(row["Ground truth document"])

        search = get_cached_search_results(query, embedding, vector_db, k, nim_client)
        retrieved_docs = search["retrieved_docs"][:k]

        hit_docs = [d for d in retrieved_docs if d in gt_docs]
        precision = len(hit_docs) / k if k > 0 else 0.0
        recall = len(hit_docs) / len(gt_docs) if gt_docs else 0.0
        f1 = 0.0 if (precision + recall) == 0 else (2 * precision * recall) / (precision + recall)
        total_f1 += f1

    mean_f1 = total_f1 / n if n > 0 else 0.0
    print(f"🎯 F1@{k}: {mean_f1:.4f}")
    return mean_f1

def map_k(df_clb, df_train, embedding, vector_db, k=5, nim_client: NIMClient = None):
    total_map = 0.0
    n = len(df_train)

    for idx, row in enumerate(df_train.iterrows()):
        row = row[1]
        query = row["Query"]
        gt_docs = _parse_ground_truth_docs(row["Ground truth document"])

        search = get_cached_search_results(query, embedding, vector_db, k, nim_client)
        retrieved_docs = search["retrieved_docs"][:k]

        hits = 0
        ap = 0.0
        for i, doc in enumerate(retrieved_docs):
            if doc in gt_docs:
                hits += 1
                ap += hits / (i + 1)

        if hits > 0:
            ap /= hits
        total_map += ap

    mean_map = total_map / n if n > 0 else 0.0
    print(f"🎯 MAP@{k}: {mean_map:.4f}")
    return mean_map

def mrr_k(df_clb, df_train, embedding, vector_db, k=5, nim_client: NIMClient = None):
    total_mrr = 0.0
    n = len(df_train)

    for idx, row in enumerate(df_train.iterrows()):
        row = row[1]
        query = row["Query"]
        gt_docs = _parse_ground_truth_docs(row["Ground truth document"])

        search = get_cached_search_results(query, embedding, vector_db, k, nim_client)
        retrieved_docs = search["retrieved_docs"][:k]

        mrr = 0.0
        for i, doc in enumerate(retrieved_docs):
            if doc in gt_docs:
                mrr = 1.0 / (i + 1)
                break
        total_mrr += mrr

    mean_mrr = total_mrr / n if n > 0 else 0.0
    print(f"🎯 MRR@{k}: {mean_mrr:.4f}")
    return mean_mrr

def ndcg_k(df_clb, df_train, embedding, vector_db, k=5, nim_client: NIMClient = None):
    total_ndcg = 0.0
    n = len(df_train)

    for idx, row in enumerate(df_train.iterrows()):
        row = row[1]
        query = row["Query"]
        gt_docs = _parse_ground_truth_docs(row["Ground truth document"])

        search = get_cached_search_results(query, embedding, vector_db, k, nim_client)
        retrieved_docs = search["retrieved_docs"][:k]
        user_emb = search["user_embedding"]

        rels = []
        for doc_id in retrieved_docs:
            if doc_id in gt_docs:
                try:
                    doc_text = df_clb.loc[doc_id - 1, 'Văn bản']
                    doc_emb = embedding.encode(doc_text)
                    s = cosine01(user_emb, doc_emb)
                    if s > 0.9: rels.append(3)
                    elif s > 0.7: rels.append(2)
                    elif s > 0.5: rels.append(1)
                    else: rels.append(0)
                except Exception as e:
                    rels.append(1)
            else:
                rels.append(0)

        ndcg_score = ndcg_at_k(rels, k)
        total_ndcg += ndcg_score

    mean_ndcg = total_ndcg / n if n > 0 else 0.0
    print(f"🎯 NDCG@{k}: {mean_ndcg:.4f}")
    return mean_ndcg

# LLM-judged retrieval context metrics
def context_precision_k(df_clb, df_train, embedding, vector_db, k=5, nim: NIMClient = None):
    if nim is None:
        raise RuntimeError("❌ NIM client is required for context_precision_k.")
    total_precision = 0.0
    n = len(df_train)

    for idx, row in enumerate(df_train.iterrows()):
        row = row[1]
        query = row["Query"]
        search = get_cached_search_results(query, embedding, vector_db, k, nim)
        results = search["results"][:k]
        messages = [
            {
                "role": "system",
                "content": """Bạn là một trợ lý AI chuyên cung cấp thông tin về Câu lạc bộ Lập trình ProPTIT.
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

Ví dụ tham khảo về phong cách trả lời:
- Hiện tại CLB ProPTIT có 6 team dự án: Team AI, Team Mobile, Team Data, Team Game, Team Web, Team Backend. Các em sẽ vào team dự án sau khi đã hoàn thành khóa học Java.
- Quá trình tuyển thành viên gồm ba vòng: đơn đăng ký, phỏng vấn và thử thách thực tế. Trong vòng training, các anh chị sẽ đánh giá dựa trên thái độ, tinh thần học hỏi và khả năng làm việc nhóm của bạn.
- CLB là nơi giao lưu, đào tạo các môn lập trình và phát triển kỹ năng mềm. Thành viên được tham gia các hoạt động học tập, dự án, sự kiện và các chương trình giao lưu với CLB khác.
-Trong giai đoạn training, CLB chú trọng đánh giá tinh thần học hỏi, khả năng làm việc nhóm và sự tuân thủ nội quy. Việc hoàn thành tốt nghĩa vụ sẽ giúp bạn ghi điểm cao.
NHIỆM VỤ:
- Trả lời các câu hỏi về CLB Lập trình ProPTIT.
- Luôn trả lời bằng tiếng Việt, trừ khi câu hỏi yêu cầu khác.
- Giữ câu trả lời trong phạm vi 2-4 câu, không lan man."""
            }
        ]
        context = "Content từ các tài liệu liên quan:\n"
        context += "\n".join([result["information"] for result in results])

        messages.append({
            "role": "user",
            "content": context + "\n\nCâu hỏi: " + query
        })

        reply = nim.chat(messages, temperature=0.0, max_tokens=512)

        batch_judge_messages = [
            {
                "role": "system",
                "content": """Bạn là một chuyên gia AI đánh giá Context Precision trong hệ thống RAG. Nhiệm vụ của bạn là đánh giá TẤT CẢ các ngữ cảnh được cung cấp cùng một lúc.

QUY TRÌNH ĐÁNH GIÁ CHI TIẾT:
1. Đọc kỹ câu hỏi của người dùng
2. Đọc kỹ câu trả lời từ mô hình AI
3. Đánh giá từng ngữ cảnh một cách độc lập
4. Xác định mức độ hỗ trợ của từng ngữ cảnh cho câu trả lời

TIÊU CHÍ ĐÁNH GIÁ CHO TỪNG NGỮ CẢNH:
- ĐÁNH GIÁ = 1 nếu ngữ cảnh CUNG CẤP ĐỦ THÔNG TIN hoặc MỘT PHẦN THÔNG TIN để trả lời câu hỏi
- ĐÁNH GIÁ = 0 nếu ngữ cảnh KHÔNG CUNG CẤP THÔNG TIN hoặc KHÔNG LIÊN QUAN đến câu hỏi
- Ngay cả khi ngữ cảnh chỉ cung cấp một phần thông tin hữu ích, vẫn đánh giá = 1
- Chỉ đánh giá = 0 khi ngữ cảnh hoàn toàn không liên quan hoặc không có thông tin gì hữu ích

ĐỊNH DẠNG OUTPUT BẮT BUỘC:
Trả về CHỈ danh sách các số 0 hoặc 1, ngăn cách bởi dấu phẩy, KHÔNG có text khác:
Ví dụ: 1,0,1,1,0

LƯU Ý QUAN TRỌNG:
- Số lượng kết quả PHẢI BẰNG số lượng ngữ cảnh được đánh giá
- Thứ tự kết quả PHẢI tương ứng với thứ tự các ngữ cảnh
- KHÔNG giải thích, KHÔNG thêm text nào khác ngoài danh sách số"""
            }
        ]

        batch_content = f"CÂU HỎI: {query}\n\nCÂU TRẢ LỜI: {reply}\n\n"
        for i, r in enumerate(results):
            snippet = r.get("information", "")
            batch_content += f"NGỮ CẢNH {i+1}:\n{snippet}\n\n"

        batch_judge_messages.append({
            "role": "user",
            "content": batch_content
        })

        batch_judged = nim.chat(batch_judge_messages, temperature=0.0, max_tokens=50).strip()

        try:
            import re
            numbers = re.findall(r'[01]', batch_judged)
            if len(numbers) == len(results):
                judged_results = [int(num) for num in numbers]
            else:
                judged_results = [1] * len(results)
        except Exception as e:
            judged_results = [1] * len(results)

        hits = sum(judged_results)
        precision = hits / k if k > 0 else 0
        total_precision += precision

    mean_precision = total_precision / n if n > 0 else 0.0
    print(f"Context Precision@{k}: {mean_precision:.3f}")
    return mean_precision

def context_recall_k(df_clb, df_train, embedding, vector_db, k=5, nim: NIMClient = None):
    if nim is None:
        raise RuntimeError("❌ NIM client is required for context_recall_k.")
    total_recall = 0.0
    n = len(df_train)

    for idx, row in enumerate(df_train.iterrows()):
        row = row[1]
        query = row["Query"]
        gt_answer = row.get("Ground truth answer", "")
        search = get_cached_search_results(query, embedding, vector_db, k, nim)
        results = search["results"][:k]

        batch_judge_messages = [
            {
                "role": "system",
                "content": """Bạn là một chuyên gia AI đánh giá Context Recall trong hệ thống RAG. Nhiệm vụ của bạn là đánh giá TẤT CẢ các ngữ cảnh được cung cấp cùng một lúc.

QUY TRÌNH ĐÁNH GIÁ CHI TIẾT:
1. Đọc kỹ câu hỏi của người dùng
2. Đọc kỹ câu trả lời chính xác (Ground Truth)
3. Đánh giá từng ngữ cảnh một cách độc lập
4. Xác định mức độ hỗ trợ của từng ngữ cảnh cho câu trả lời chính xác

TIÊU CHÍ ĐÁNH GIÁ CHO TỪNG NGỮ CẢNH:
- ĐÁNH GIÁ = 1 nếu ngữ cảnh CUNG CẤP ĐỦ THÔNG TIN hoặc MỘT PHẦN THÔNG TIN để trả lời câu hỏi
- ĐÁNH GIÁ = 0 nếu ngữ cảnh KHÔNG CUNG CẤP THÔNG TIN hoặc KHÔNG LIÊN QUAN đến câu hỏi
- Ngay cả khi ngữ cảnh chỉ cung cấp một phần thông tin hữu ích, vẫn đánh giá = 1
- Chỉ đánh giá = 0 khi ngữ cảnh hoàn toàn không liên quan hoặc không có thông tin gì hữu ích

ĐỊNH DẠNG OUTPUT BẮT BUỘC:
Trả về CHỈ danh sách các số 0 hoặc 1, ngăn cách bởi dấu phẩy, KHÔNG có text khác:
Ví dụ: 1,0,1,1,0

LƯU Ý QUAN TRỌNG:
- Số lượng kết quả PHẢI BẰNG số lượng ngữ cảnh được đánh giá
- Thứ tự kết quả PHẢI tương ứng với thứ tự các ngữ cảnh
- KHÔNG giải thích, KHÔNG thêm text nào khác ngoài danh sách số"""
            }
        ]
        batch_content = f"CÂU HỎI: {query}\n\nCÂU TRẢ LỜI CHÍNH XÁC: {gt_answer}\n\n"
        for i, r in enumerate(results):
            snippet = r.get("information", "")
            batch_content += f"NGỮ CẢNH {i+1}:\n{snippet}\n\n"

        batch_judge_messages.append({
            "role": "user",
            "content": batch_content
        })

        batch_judged = nim.chat(batch_judge_messages, temperature=0.0, max_tokens=50).strip()

        try:
            import re
            numbers = re.findall(r'[01]', batch_judged)
            if len(numbers) == len(results):
                judged_results = [int(num) for num in numbers]
            else:
                judged_results = [1] * len(results)
        except Exception as e:
            judged_results = [1] * len(results)

        hits = sum(judged_results)
        recall = hits / k if k > 0 else 0
        total_recall += recall

    mean_recall = total_recall / n if n > 0 else 0.0
    print(f"Context Recall@{k}: {mean_recall:.3f}")
    return mean_recall

def context_entities_recall_k(df_clb, df_train, embedding, vector_db, k=5, nim: NIMClient = None):
    if nim is None:
        raise RuntimeError("❌ NIM client is required for context_entities_recall_k.")
    total_recall = 0.0
    n = len(df_train)
    print(f"🔍 Computing Context Entities Recall@{k} for {n} queries...")

    for idx, row in enumerate(df_train.iterrows()):
        row = row[1]
        gt_answer = row.get("Ground truth answer", "")
        query = row["Query"]
        print(f"🔍 Query {idx+1:3d}/{n}: {query[:50]}...")

        search = get_cached_search_results(query, embedding, vector_db, k, nim)
        results = search["results"][:k]
        messages_entities = [
            {
                "role": "system",
                "content": """Bạn là một trợ lý AI chuyên trích xuất các thực thể từ câu trả lời. Bạn sẽ được cung cấp một câu trả lời và nhiệm vụ của bạn là trích xuất các thực thể từ câu trả lời đó. Các thực thể có thể là tên người, địa điểm, tổ chức, sự kiện, v.v. Hãy trả lời dưới dạng một danh sách các thực thể.
                Ví dụ:
                Câu trả lời: Nếu bạn thuộc ngành khác bạn vẫn có thể tham gia CLB chúng mình. Nếu định hướng của bạn hoàn toàn là theo CNTT thì CLB chắc chắn là nơi phù hợp nhất để các bạn phát triển. Trở ngại lớn nhất sẽ là do bạn theo một hướng khác nữa nên sẽ phải tập trung vào cả 2 mảng nên sẽ cần cố gắng nhiều hơn.
                ["ngành khác", "CLB", "CNTT", "mảng]
                Câu trả lời: Câu lạc bộ Lập Trình PTIT (Programming PTIT), tên viết tắt là PROPTIT được thành lập ngày 9/10/2011. Với phương châm hoạt động "Chia sẻ để cùng nhau phát triển", câu lạc bộ là nơi giao lưu, đào tạo các môn lập trình và các môn học trong trường, tạo điều kiện để sinh viên trong Học viện có môi trường học tập năng động sáng tạo. Slogan: Lập Trình PTIT - Lập trình từ trái tim.
                ["Câu lạc bộ Lập Trình PTIT (Programming PTIT)", "PROPTIT", "9/10/2011", "Chia sẻ để cùng nhau phát triển", "sinh viên", "Học viện", "Lập Trình PTIT - Lập trình từ trái tim"]"""
            },
            {
                "role": "user",
                "content": f"Câu trả lời: {gt_answer}"
            }
        ]
        entities_text = nim.chat(messages_entities, temperature=0.0, max_tokens=256)

        # Parse entities
        entities = []
        try:
            start_idx = entities_text.find('[')
            end_idx = entities_text.rfind(']') + 1
            if start_idx != -1 and end_idx != -1:
                entities_str = entities_text[start_idx:end_idx]
                entities = eval(entities_str)
            else:
                entities = []
        except Exception as e:
            entities = []

        # Count presence of entities in the retrieved contexts
        tmp = len(entities)
        hits = 0
        if tmp == 0:
            recall = 0.0
        else:
            contexts = [r.get("information", "") for r in results]
            full_context = "\n".join(contexts)
            for entity in entities:
                if entity.strip() in full_context:
                    hits += 1
        recall = hits / tmp if tmp > 0 else 0.0
        total_recall += recall
        print(f"✅ Context Entities Recall: {recall:.3f}")

    mean_recall = total_recall / n if n > 0 else 0.0
    print(f"[context_entities_recall@k] Mean: {mean_recall:.3f}")
    return mean_recall

# 10) LLM Answer Quality Metrics
def string_presence_k(df_clb, df_train, embedding, vector_db, k=5, nim: NIMClient = None, use_hybrid: bool = True, hybrid_alpha: float = 0.7):
    print(f"🔍 [string_presence@k] Starting string_presence@k calculation with k={k}, hybrid={use_hybrid}, alpha={hybrid_alpha}")
    if nim is None:
        raise RuntimeError("❌ NIM client is required for string_presence_k.")
    total_presence = 0.0
    n = len(df_train)
    print(f"📊 [string_presence@k] Processing {n} queries total")

    for idx, row in enumerate(df_train.iterrows()):
        row = row[1]
        query = row["Query"]
        gt_answer = row.get("Ground truth answer", "")

        print(f"🔍 [string_presence@k] Query {idx+1:3d}/{n}: '{query[:50]}...'")
        print(f"📋 [string_presence@k] GT Answer: '{gt_answer[:100]}...'")

        search = get_cached_search_results(query, embedding, vector_db, k, nim, use_hybrid, hybrid_alpha)
        results = search["results"][:k]

        print(f"🎯 [string_presence@k] Retrieved {len(results)} documents")

        context = "Content từ các tài liệu liên quan:\n"
        context += "\n".join([result["information"] for result in results])

        messages = [
            {
                "role": "system",
                "content": """Bạn là một trợ lý AI chuyên cung cấp thông tin về Câu lạc bộ Lập trình ProPTIT.
Bạn sẽ nhận được dữ liệu ngữ cảnh (context) từ một hệ thống Retrieval-Augmented Generation (RAG) chứa các thông tin chính xác về CLB.

NGUYÊN TẮC TRẢ LỜI BẮT BUỘC:
1. CHỈ sử dụng thông tin từ context được cung cấp để trả lời. KHÔNG được thêm thông tin ngoài context.
2. Trả lời NGẮN GỌN, CHÍNH XÁC, và TRỰC TIẾP vào câu hỏi.
3. KHÔNG được thêm lời chào hỏi, cảm ơn, hoặc câu xã giao không cần thiết.
4. KHÔNG được nói "Xin lỗi", "Tôi không biết", "Không có thông tin" - PHẢI trả lời dựa trên context có sẵn.
5. Nếu context không đủ, hãy suy luận LOGIC từ thông tin có sẵn mà KHÔNG bịa thêm.
6. Tập trung trả lời CÂU HỎI CHÍNH, bỏ qua thông tin không liên quan.
7. Sử dụng ngôn ngữ tự nhiên, dễ hiểu, phù hợp với phong cách trả lời của con người.
8. Ưu tiên xưng là "CLB" khi nói về tổ chức.
9. Sử dụng CÁC TỪ KHÓA và THỰC THỂ chính xác từ context để đảm bảo tính chính xác.
10. Trả lời bằng tiếng Việt, trừ khi câu hỏi yêu cầu khác.

Ví dụ tham khảo về phong cách trả lời:
- CLB luôn khuyến khích các bạn thuộc ngành khác tham gia nếu có đam mê.
- Trong quá trình training, CLB đánh giá cao sự tiến bộ và tinh thần học hỏi của các thành viên.
- Thành viên cần tuân thủ nội quy về trang phục, giờ giấc và thái độ trong các buổi sinh hoạt.
- CLB thường xuyên tổ chức các hoạt động giao lưu, hợp tác để mở rộng mạng lưới và chia sẻ kinh nghiệm.
- CLB thu thập phản hồi để không ngừng cải thiện chất lượng hoạt động.
- Khi tổ chức sự kiện, CLB chuẩn bị kỹ lưỡng từ nội dung đến hậu cần để đảm bảo thành công.

NHIỆM VỤ:
- Trả lời các câu hỏi về CLB Lập trình ProPTIT.
- Giữ câu trả lời trong phạm vi 2-4 câu, không lan man.
- Sử dụng thông tin chính xác từ context để đạt độ chính xác cao."""
            }
        ]
        messages.append({
            "role": "user",
            "content": context + "\n\nCâu hỏi: " + query
        })

        response = nim.chat(messages, temperature=0.1, max_tokens=512)  # Tăng temperature và max_tokens để cải thiện quality
        print(f"🤖 [string_presence@k] LLM Response: '{response[:100]}...'")

        # Extract entities from GT answer
        messages_entities = [
            {
                "role": "system",
                "content": """Bạn là một trợ lý AI chuyên trích xuất các thực thể từ câu trả lời. Bạn sẽ được cung cấp một câu trả lời và nhiệm vụ của bạn là trích xuất các thực thể từ câu trả lời đó. Các thực thể có thể là tên người, địa điểm, tổ chức, sự kiện, v.v. Hãy trả lời dưới dạng một danh sách các thực thể.
                Ví dụ:
                Câu trả lời: Nếu bạn thuộc ngành khác bạn vẫn có thể tham gia CLB chúng mình. Nếu định hướng của bạn hoàn toàn là theo CNTT thì CLB chắc chắn là nơi phù hợp nhất để các bạn phát triển. Trở ngại lớn nhất sẽ là do bạn theo một hướng khác nữa nên sẽ phải tập trung vào cả 2 mảng nên sẽ cần cố gắng nhiều hơn.
                ["ngành khác", "CLB", "CNTT", "mảng]
                Câu trả lời: Câu lạc bộ Lập Trình PTIT (Programming PTIT), tên viết tắt là PROPTIT được thành lập ngày 9/10/2011. Với phương châm hoạt động "Chia sẻ để cùng nhau phát triển", câu lạc bộ là nơi giao lưu, đào tạo các môn lập trình và các môn học trong trường, tạo điều kiện để sinh viên trong Học viện có môi trường học tập năng động sáng tạo. Slogan: Lập Trình PTIT - Lập trình từ trái tim.
                ["Câu lạc bộ Lập Trình PTIT (Programming PTIT)", "PROPTIT", "9/10/2011", "Chia sẻ để cùng nhau phát triển", "sinh viên", "Học viện", "Lập Trình PTIT - Lập trình từ trái tim"]"""
            },
            {
                "role": "user",
                "content": f"Câu trả lời: {gt_answer}"
            }
        ]

        entities_text = nim.chat(messages_entities, temperature=0.1, max_tokens=512)
        entities = []
        try:
            start_idx = entities_text.find('[')
            end_idx = entities_text.rfind(']') + 1
            if start_idx != -1 and end_idx != -1:
                entities_str = entities_text[start_idx:end_idx]
                entities = eval(entities_str)
            else:
                entities = []
        except Exception as e:
            print(f"⚠️ [string_presence@k] Error parsing entities: {e}")
            entities = []

        print(f"🔍 [string_presence@k] Extracted entities: {entities}")

        # Count entity presence in response
        hits = 0
        for entity in entities:
            if entity.strip() in response:
                hits += 1
                print(f"✅ [string_presence@k] Entity found: '{entity.strip()}'")

        presence = hits / len(entities) if len(entities) > 0 else 0
        print(f"📈 [string_presence@k] Presence score: {presence:.4f} (Found: {hits}/{len(entities)})")

        total_presence += presence
        running_avg = total_presence / (idx + 1)
        print(f"📊 [string_presence@k] Running average: {running_avg:.4f}")
        print()

    mean_presence = total_presence / n if n > 0 else 0.0
    print(f"🎯 [string_presence@k] FINAL RESULT: Mean presence = {mean_presence:.4f}")
    return mean_presence

def rouge_l_k(df_clb, df_train, embedding, vector_db, k=5, nim: NIMClient = None, use_hybrid: bool = True, hybrid_alpha: float = 0.7):
    print(f"🔍 [rouge_l@k] Starting ROUGE-L@k calculation with k={k}, hybrid={use_hybrid}, alpha={hybrid_alpha}")
    if nim is None:
        raise RuntimeError("❌ NIM client is required for rouge_l_k.")
    rouge = Rouge()
    total_rouge_l = 0.0
    n = len(df_train)
    print(f"📊 [rouge_l@k] Processing {n} queries total")

    for idx, row in enumerate(df_train.iterrows()):
        row = row[1]
        query = row["Query"]
        gt_answer = row.get("Ground truth answer", "")

        print(f"🔍 [rouge_l@k] Query {idx+1:3d}/{n}: '{query[:50]}...'")
        print(f"📋 [rouge_l@k] GT Answer: '{gt_answer[:100]}...'")

        search = get_cached_search_results(query, embedding, vector_db, k, nim, use_hybrid, hybrid_alpha)
        results = search["results"][:k]

        print(f"🎯 [rouge_l@k] Retrieved {len(results)} documents")

        context = "Content từ các tài liệu liên quan:\n"
        context += "\n".join([result["information"] for result in results])

        messages = [
            {
                "role": "system",
                "content": """Bạn là một trợ lý AI chuyên cung cấp thông tin về Câu lạc bộ Lập trình ProPTIT.
Bạn sẽ nhận được dữ liệu ngữ cảnh (context) từ một hệ thống Retrieval-Augmented Generation (RAG) chứa các thông tin chính xác về CLB.

NGUYÊN TẮC TRẢ LỜI BẮT BUỘC:
1. CHỈ sử dụng thông tin từ context được cung cấp để trả lời. KHÔNG được thêm thông tin ngoài context.
2. Trả lời NGẮN GỌN, CHÍNH XÁC, và TRỰC TIẾP vào câu hỏi.
3. KHÔNG được thêm lời chào hỏi, cảm ơn, hoặc câu xã giao không cần thiết.
4. KHÔNG được nói "Xin lỗi", "Tôi không biết", "Không có thông tin" - PHẢI trả lời dựa trên context có sẵn.
5. Nếu context không đủ, hãy suy luận LOGIC từ thông tin có sẵn mà KHÔNG bịa thêm.
6. Tập trung trả lời CÂU HỎI CHÍNH, bỏ qua thông tin không liên quan.
7. Sử dụng ngôn ngữ tự nhiên, dễ hiểu, phù hợp với phong cách trả lời của con người.
8. Ưu tiên xưng là "CLB" khi nói về tổ chức.
9. Sử dụng CÁC TỪ KHÓA và THỰC THỂ chính xác từ context để đảm bảo tính chính xác.
10. Trả lời bằng tiếng Việt, trừ khi câu hỏi yêu cầu khác.

Ví dụ tham khảo về phong cách trả lời:
- CLB luôn khuyến khích các bạn thuộc ngành khác tham gia nếu có đam mê.
- Trong quá trình training, CLB đánh giá cao sự tiến bộ và tinh thần học hỏi của các thành viên.
- Thành viên cần tuân thủ nội quy về trang phục, giờ giấc và thái độ trong các buổi sinh hoạt.
- CLB thường xuyên tổ chức các hoạt động giao lưu, hợp tác để mở rộng mạng lưới và chia sẻ kinh nghiệm.
- CLB thu thập phản hồi để không ngừng cải thiện chất lượng hoạt động.
- Khi tổ chức sự kiện, CLB chuẩn bị kỹ lưỡng từ nội dung đến hậu cần để đảm bảo thành công.

NHIỆM VỤ:
- Trả lời các câu hỏi về CLB Lập trình ProPTIT.
- Giữ câu trả lời trong phạm vi 2-4 câu, không lan man.
- Sử dụng thông tin chính xác từ context để đạt độ chính xác cao."""
            }
        ]
        messages.append({
            "role": "user",
            "content": context + "\n\nCâu hỏi: " + query
        })

        response = nim.chat(messages, temperature=0.1, max_tokens=512)  # Tăng temperature và max_tokens để cải thiện quality
        print(f"🤖 [rouge_l@k] LLM Response: '{response[:100]}...'")

        # Calculate ROUGE-L
        scores = rouge.get_scores(response, gt_answer)
        rouge_l = scores[0]['rouge-l']['f']
        print(f"📊 [rouge_l@k] ROUGE-L calculation:")
        print(f"   - Response length: {len(response)} chars")
        print(f"   - GT Answer length: {len(gt_answer)} chars")
        print(f"   - ROUGE-L F1: {rouge_l:.4f}")

        total_rouge_l += rouge_l
        running_avg = total_rouge_l / (idx + 1)
        print(f"📊 [rouge_l@k] Running average: {running_avg:.4f}")
        print()

    mean_rouge_l = total_rouge_l / n if n > 0 else 0.0
    print(f"🎯 [rouge_l@k] FINAL RESULT: Mean ROUGE-L = {mean_rouge_l:.4f}")
    return mean_rouge_l

def bleu_4_k(df_clb, df_train, embedding, vector_db, k=5, nim: NIMClient = None, use_hybrid: bool = True, hybrid_alpha: float = 0.7):
    print(f"🔍 [bleu_4@k] Starting BLEU-4@k calculation with k={k}, hybrid={use_hybrid}, alpha={hybrid_alpha}")
    if nim is None:
        raise RuntimeError("❌ NIM client is required for bleu_4_k.")
    total_bleu_4 = 0.0
    n = len(df_train)
    smoothing_function = SmoothingFunction().method1
    print(f"📊 [bleu_4@k] Processing {n} queries total")

    for idx, row in enumerate(df_train.iterrows()):
        row = row[1]
        query = row["Query"]
        gt_answer = row.get("Ground truth answer", "")

        print(f"🔍 [bleu_4@k] Query {idx+1:3d}/{n}: '{query[:50]}...'")
        print(f"📋 [bleu_4@k] GT Answer: '{gt_answer[:100]}...'")

        search = get_cached_search_results(query, embedding, vector_db, k, nim, use_hybrid, hybrid_alpha)
        results = search["results"][:k]

        print(f"🎯 [bleu_4@k] Retrieved {len(results)} documents")

        context = "Content từ các tài liệu liên quan:\n"
        context += "\n".join([result["information"] for result in results])

        messages = [
            {
                "role": "system",
                "content": """Bạn là một trợ lý AI chuyên cung cấp thông tin về Câu lạc bộ Lập trình ProPTIT.
Bạn sẽ nhận được dữ liệu ngữ cảnh (context) từ một hệ thống Retrieval-Augmented Generation (RAG) chứa các thông tin chính xác về CLB.

NGUYÊN TẮC TRẢ LỜI BẮT BUỘC:
1. CHỈ sử dụng thông tin từ context được cung cấp để trả lời. KHÔNG được thêm thông tin ngoài context.
2. Trả lời NGẮN GỌN, CHÍNH XÁC, và TRỰC TIẾP vào câu hỏi.
3. KHÔNG được thêm lời chào hỏi, cảm ơn, hoặc câu xã giao không cần thiết.
4. KHÔNG được nói "Xin lỗi", "Tôi không biết", "Không có thông tin" - PHẢI trả lời dựa trên context có sẵn.
5. Nếu context không đủ, hãy suy luận LOGIC từ thông tin có sẵn mà KHÔNG bịa thêm.
6. Tập trung trả lời CÂU HỎI CHÍNH, bỏ qua thông tin không liên quan.
7. Sử dụng ngôn ngữ tự nhiên, dễ hiểu, phù hợp với phong cách trả lời của con người.
8. Ưu tiên xưng là "CLB" khi nói về tổ chức.
9. Sử dụng CÁC TỪ KHÓA và THỰC THỂ chính xác từ context để đảm bảo tính chính xác.
10. Trả lời bằng tiếng Việt, trừ khi câu hỏi yêu cầu khác.

Ví dụ tham khảo về phong cách trả lời:
- CLB luôn khuyến khích các bạn thuộc ngành khác tham gia nếu có đam mê.
- Trong quá trình training, CLB đánh giá cao sự tiến bộ và tinh thần học hỏi của các thành viên.
- Thành viên cần tuân thủ nội quy về trang phục, giờ giấc và thái độ trong các buổi sinh hoạt.
- CLB thường xuyên tổ chức các hoạt động giao lưu, hợp tác để mở rộng mạng lưới và chia sẻ kinh nghiệm.
- CLB thu thập phản hồi để không ngừng cải thiện chất lượng hoạt động.
- Khi tổ chức sự kiện, CLB chuẩn bị kỹ lưỡng từ nội dung đến hậu cần để đảm bảo thành công.

NHIỆM VỤ:
- Trả lời các câu hỏi về CLB Lập trình ProPTIT.
- Giữ câu trả lời trong phạm vi 2-4 câu, không lan man.
- Sử dụng thông tin chính xác từ context để đạt độ chính xác cao."""
            }
        ]
        messages.append({
            "role": "user",
            "content": context + "\n\nCâu hỏi: " + query
        })

        response = nim.chat(messages, temperature=0.1, max_tokens=512)  # Tăng temperature và max_tokens để cải thiện quality
        print(f"🤖 [bleu_4@k] LLM Response: '{response[:100]}...'")

        # Calculate BLEU-4
        reference = [gt_answer.split()]
        candidate = response.split()
        bleu_score = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing_function)

        print(f"📊 [bleu_4@k] BLEU-4 calculation:")
        print(f"   - Reference tokens: {len(reference[0])}")
        print(f"   - Candidate tokens: {len(candidate)}")
        print(f"   - BLEU-4 score: {bleu_score:.4f}")

        total_bleu_4 += bleu_score
        running_avg = total_bleu_4 / (idx + 1)
        print(f"📊 [bleu_4@k] Running average: {running_avg:.4f}")
        print()

    mean_bleu_4 = total_bleu_4 / n if n > 0 else 0.0
    print(f"🎯 [bleu_4@k] FINAL RESULT: Mean BLEU-4 = {mean_bleu_4:.4f}")
    return mean_bleu_4

def groundedness_k(df_clb, df_train, embedding, vector_db, k=5, nim: NIMClient = None, use_hybrid: bool = True, hybrid_alpha: float = 0.7):
    print(f"🔍 [groundedness@k] Starting groundedness@k calculation with k={k}, hybrid={use_hybrid}, alpha={hybrid_alpha}")
    if nim is None:
        raise RuntimeError("❌ NIM client is required for groundedness_k.")
    total_groundedness = 0.0
    n = len(df_train)
    print(f"📊 [groundedness@k] Processing {n} queries total")

    for idx, row in enumerate(df_train.iterrows()):
        row = row[1]
        query = row["Query"]
        gt_answer = row.get("Ground truth answer", "")

        print(f"🔍 [groundedness@k] Query {idx+1:3d}/{n}: '{query[:50]}...'")
        print(f"📋 [groundedness@k] GT Answer: '{gt_answer[:100]}...'")

        search = get_cached_search_results(query, embedding, vector_db, k, nim, use_hybrid, hybrid_alpha)
        results = search["results"][:k]

        print(f"🎯 [groundedness@k] Retrieved {len(results)} documents")
        context = "Content từ các tài liệu liên quan:\n"
        context += "\n".join([result["information"] for result in results])

        messages = [
            {
                "role": "system",
                "content": """Bạn là một trợ lý AI chuyên cung cấp thông tin về Câu lạc bộ Lập trình ProPTIT.
Bạn sẽ nhận được dữ liệu ngữ cảnh (context) từ một hệ thống Retrieval-Augmented Generation (RAG) chứa các thông tin chính xác về CLB.

NGUYÊN TẮC TRẢ LỜI BẮT BUỘC:
1. CHỈ sử dụng thông tin từ context được cung cấp để trả lời. KHÔNG được thêm thông tin ngoài context.
2. Trả lời NGẮN GỌN, CHÍNH XÁC, và TRỰC TIẾP vào câu hỏi.
3. KHÔNG được thêm lời chào hỏi, cảm ơn, hoặc câu xã giao không cần thiết.
4. KHÔNG được nói "Xin lỗi", "Tôi không biết", "Không có thông tin" - PHẢI trả lời dựa trên context có sẵn.
5. Nếu context không đủ, hãy suy luận LOGIC từ thông tin có sẵn mà KHÔNG bịa thêm.
6. Tập trung trả lời CÂU HỎI CHÍNH, bỏ qua thông tin không liên quan.
7. Sử dụng ngôn ngữ tự nhiên, dễ hiểu, phù hợp với phong cách trả lời của con người.
8. Ưu tiên xưng là "CLB" khi nói về tổ chức.
9. Sử dụng CÁC TỪ KHÓA và THỰC THỂ chính xác từ context để đảm bảo tính chính xác.
10. Trả lời bằng tiếng Việt, trừ khi câu hỏi yêu cầu khác.

Ví dụ tham khảo về phong cách trả lời:
- CLB luôn khuyến khích các bạn thuộc ngành khác tham gia nếu có đam mê.
- Trong quá trình training, CLB đánh giá cao sự tiến bộ và tinh thần học hỏi của các thành viên.
- Thành viên cần tuân thủ nội quy về trang phục, giờ giấc và thái độ trong các buổi sinh hoạt.
- CLB thường xuyên tổ chức các hoạt động giao lưu, hợp tác để mở rộng mạng lưới và chia sẻ kinh nghiệm.
- CLB thu thập phản hồi để không ngừng cải thiện chất lượng hoạt động.
- Khi tổ chức sự kiện, CLB chuẩn bị kỹ lưỡng từ nội dung đến hậu cần để đảm bảo thành công.

NHIỆM VỤ:
- Trả lời các câu hỏi về CLB Lập trình ProPTIT.
- Giữ câu trả lời trong phạm vi 2-4 câu, không lan man.
- Sử dụng thông tin chính xác từ context để đạt độ chính xác cao."""
            }
        ]
        messages.append({
            "role": "user",
            "content": context + "\n\nCâu hỏi: " + query
        })

        response = nim.chat(messages, temperature=0.1, max_tokens=512)  # Tăng temperature và max_tokens để cải thiện quality
        print(f"🤖 [groundedness@k] LLM Response: '{response[:100]}...'")

        # Split response into sentences
        sentences = response.split('. ')
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            print(f"⚠️ [groundedness@k] No sentences found, skipping...")
            continue

        print(f"📝 [groundedness@k] Split into {len(sentences)} sentences for evaluation")

        batch_judge_messages = [
            {
                "role": "system",
                "content": """Bạn là một chuyên gia đánh giá Groundedness trong hệ thống RAG. Nhiệm vụ của bạn là đánh giá TẤT CẢ các câu được cung cấp cùng một lúc.

QUY TRÌNH ĐÁNH GIÁ CHI TIẾT:
1. Đọc kỹ câu hỏi của người dùng
2. Đọc kỹ ngữ cảnh được cung cấp
3. Đánh giá từng câu một cách độc lập
4. Phân loại mức độ groundedness của từng câu

TIÊU CHÍ ĐÁNH GIÁ CHO TỪNG CÂU:
- supported: Nội dung câu được ngữ cảnh hỗ trợ hoặc suy ra trực tiếp
- unsupported: Nội dung câu không được ngữ cảnh hỗ trợ, và không thể suy ra từ đó
- contradictory: Nội dung câu trái ngược hoặc mâu thuẫn với ngữ cảnh
- no_rad: Câu không yêu cầu kiểm tra thực tế (ví dụ: câu chào hỏi, ý kiến cá nhân, câu hỏi tu từ, disclaimers)

ĐỊNH DẠNG OUTPUT BẮT BUỘC:
Trả về CHỈ danh sách các nhãn, ngăn cách bởi dấu phẩy, KHÔNG có text khác:
Ví dụ: supported,unsupported,supported,no_rad,contradictory

LƯU Ý QUAN TRỌNG:
- Số lượng kết quả PHẢI BẰNG số lượng câu được đánh giá
- Thứ tự kết quả PHẢI tương ứng với thứ tự các câu
- KHÔNG giải thích, KHÔNG thêm text nào khác ngoài danh sách nhãn"""
            }
        ]
        batch_content = f"CÂU HỎI: {query}\n\nNGỮ CẢNH:\n{context}\n\n"
        for i, sentence in enumerate(sentences):
            batch_content += f"CÂU {i+1}: {sentence}\n"

        batch_judge_messages.append({
            "role": "user",
            "content": batch_content
        })

        
        batch_judged = nim.chat(batch_judge_messages, temperature=0.0, max_tokens=100).strip()
        print(f"🔍 [groundedness@k] Batch judgment result: '{batch_judged}'")

        
        try:
            # Extract labels from response
            import re
            labels = re.findall(r'\b(supported|unsupported|contradictory|no_rad)\b', batch_judged.lower())
            if len(labels) == len(sentences):
                judged_results = labels
            else:
                # Fallback: assume all are supported if parsing fails
                print(f"⚠️ [groundedness@k] Failed to parse batch results, using fallback (all supported)")
                judged_results = ["supported"] * len(sentences)
        except Exception as e:
            print(f"⚠️ [groundedness@k] Error parsing batch results: {e}, using fallback (all supported)")
            judged_results = ["supported"] * len(sentences)

        hits = sum(1 for label in judged_results if label == "supported")
        cnt = len(sentences)
        groundedness_score = hits / cnt if cnt > 0 else 0.0

        print(f"📊 [groundedness@k] Groundedness evaluation:")
        print(f"   - Total sentences: {cnt}")
        print(f"   - Supported sentences: {hits}")
        print(f"   - Groundedness score: {groundedness_score:.4f}")

        total_groundedness += groundedness_score
        running_avg = total_groundedness / (idx + 1)
        print(f"📊 [groundedness@k] Running average: {running_avg:.4f}")
        print()

    mean_groundedness = total_groundedness / n if n > 0 else 0.0
    print(f"🎯 [groundedness@k] FINAL RESULT: Mean groundedness = {mean_groundedness:.4f}")
    return mean_groundedness

def response_relevancy_k(df_clb, df_train, embedding, vector_db, k=5, nim: NIMClient = None, use_hybrid: bool = True, hybrid_alpha: float = 0.7):
    print(f"🔍 [response_relevancy@k] Starting response_relevancy@k calculation with k={k}, hybrid={use_hybrid}, alpha={hybrid_alpha}")
    if nim is None:
        raise RuntimeError("❌ NIM client is required for response_relevancy_k.")
    total_relevancy = 0.0
    n = len(df_train)
    print(f"📊 [response_relevancy@k] Processing {n} queries total")

    for idx, row in enumerate(df_train.iterrows()):
        row = row[1]
        query = row["Query"]

        print(f"🔍 [response_relevancy@k] Query {idx+1:3d}/{n}: '{query[:50]}...'")

        search = get_cached_search_results(query, embedding, vector_db, k, nim, use_hybrid, hybrid_alpha)
        results = search["results"][:k]

        print(f"🎯 [response_relevancy@k] Retrieved {len(results)} documents")

        context = "Content từ các tài liệu liên quan:\n"
        context += "\n".join([result["information"] for result in results])

        messages = [
            {
                "role": "system",
                "content": """Bạn là một trợ lý AI chuyên cung cấp thông tin về Câu lạc bộ Lập trình ProPTIT.
Bạn sẽ nhận được dữ liệu ngữ cảnh (context) từ một hệ thống Retrieval-Augmented Generation (RAG) chứa các thông tin chính xác về CLB.

NGUYÊN TẮC TRẢ LỜI BẮT BUỘC:
1. CHỈ sử dụng thông tin từ context được cung cấp để trả lời. KHÔNG được thêm thông tin ngoài context.
2. Trả lời NGẮN GỌN, CHÍNH XÁC, và TRỰC TIẾP vào câu hỏi.
3. KHÔNG được thêm lời chào hỏi, cảm ơn, hoặc câu xã giao không cần thiết.
4. KHÔNG được nói "Xin lỗi", "Tôi không biết", "Không có thông tin" - PHẢI trả lời dựa trên context có sẵn.
5. Nếu context không đủ, hãy suy luận LOGIC từ thông tin có sẵn mà KHÔNG bịa thêm.
6. Tập trung trả lời CÂU HỎI CHÍNH, bỏ qua thông tin không liên quan.
7. Sử dụng ngôn ngữ tự nhiên, dễ hiểu, phù hợp với phong cách trả lời của con người.
8. Ưu tiên xưng là "CLB" khi nói về tổ chức.
9. Sử dụng CÁC TỪ KHÓA và THỰC THỂ chính xác từ context để đảm bảo tính chính xác.
10. Trả lời bằng tiếng Việt, trừ khi câu hỏi yêu cầu khác.

Ví dụ tham khảo về phong cách trả lời:
- CLB luôn khuyến khích các bạn thuộc ngành khác tham gia nếu có đam mê.
- Trong quá trình training, CLB đánh giá cao sự tiến bộ và tinh thần học hỏi của các thành viên.
- Thành viên cần tuân thủ nội quy về trang phục, giờ giấc và thái độ trong các buổi sinh hoạt.
- CLB thường xuyên tổ chức các hoạt động giao lưu, hợp tác để mở rộng mạng lưới và chia sẻ kinh nghiệm.
- CLB thu thập phản hồi để không ngừng cải thiện chất lượng hoạt động.
- Khi tổ chức sự kiện, CLB chuẩn bị kỹ lưỡng từ nội dung đến hậu cần để đảm bảo thành công.

NHIỆM VỤ:
- Trả lời các câu hỏi về CLB Lập trình ProPTIT.
- Giữ câu trả lời trong phạm vi 2-4 câu, không lan man.
- Sử dụng thông tin chính xác từ context để đạt độ chính xác cao."""
            }
        ]
        messages.append({
            "role": "user",
            "content": context + "\n\nCâu hỏi: " + query
        })

        response = nim.chat(messages, temperature=0.1, max_tokens=512)  # Tăng temperature và max_tokens để cải thiện quality
        print(f"🤖 [response_relevancy@k] LLM Response: '{response[:100]}...'")

        # Generate related questions from the response
        messages_related = [
            {
                "role": "system",
                "content": """Bạn là một trợ lý AI chuyên tạo ra các câu hỏi liên quan từ một câu trả lời. Bạn sẽ được cung cấp một câu trả lời và nhiệm vụ của bạn là tạo ra các câu hỏi liên quan đến câu trả lời đó. Hãy tạo ra ít nhất 5 câu hỏi liên quan, mỗi câu hỏi nên ngắn gọn và rõ ràng. Trả lời dưới dạng list các câu hỏi như ở ví dụ dưới. LƯU Ý: Trả lời dưới dạng ["câu hỏi 1", "câu hỏi 2", "câu hỏi 3", ...], bao gồm cả dấu ngoặc vuông.
    Ví dụ:
    Câu trả lời: Câu lạc bộ Lập Trình PTIT (Programming PTIT), tên viết tắt là PROPTIT được thành lập ngày 9/10/2011. Với phương châm hoạt động "Chia sẻ để cùng nhau phát triển", câu lạc bộ là nơi giao lưu, đào tạo các môn lập trình và các môn học trong trường, tạo điều kiện để sinh viên trong Học viện có môi trường học tập năng động sáng tạo. Slogan: Lập Trình PTIT - Lập trình từ trái tim.
    Output của bạn: "["CLB Lập Trình PTIT được thành lập khi nào?", "Slogan của CLB là gì?", "Mục tiêu của CLB là gì?"]"
    Câu trả lời: Nếu bạn thuộc ngành khác bạn vẫn có thể tham gia CLB chúng mình. Nếu định hướng của bạn hoàn toàn là theo CNTT thì CLB chắc chắn là nơi phù hợp nhất để các bạn phát triển. Trở ngại lớn nhất sẽ là do bạn theo một hướng khác nữa nên sẽ phải tập trung vào cả 2 mảng nên sẽ cần cố gắng nhiều hơn.
    Output của bạn: "["Ngành nào có thể tham gia CLB?", "CLB phù hợp với những ai?", "Trở ngại lớn nhất khi tham gia CLB là gì?"]"""
            },
            {
                "role": "user",
                "content": f"Câu trả lời: {response}"
            }
        ]

        related_questions_text = nim.chat(messages_related, temperature=0.0, max_tokens=256)
        print(f"🔍 [response_relevancy@k] Related questions response: '{related_questions_text[:100]}...'")

        # Parse related questions
        try:
            start_idx = related_questions_text.find('[')
            end_idx = related_questions_text.rfind(']') + 1
            if start_idx != -1 and end_idx != -1:
                questions_str = related_questions_text[start_idx:end_idx]
                related_questions = eval(questions_str)
            else:
                related_questions = []
        except Exception as e:
            print(f"⚠️ [response_relevancy@k] Error parsing related questions: {e}")
            related_questions = []

        print(f"📝 [response_relevancy@k] Generated {len(related_questions)} related questions:")
        for i, q in enumerate(related_questions):
            print(f"   {i+1}. {q}")

        # Evaluate if original query can answer the related questions using batch processing
        if not related_questions:
            print(f"⚠️ [response_relevancy@k] No related questions found, skipping...")
            continue

        batch_judge_messages = [
            {
                "role": "system",
                "content": """Bạn là một chuyên gia AI đánh giá Response Relevancy. Nhiệm vụ của bạn là đánh giá TẤT CẢ các câu hỏi liên quan cùng một lúc.

QUY TRÌNH ĐÁNH GIÁ CHI TIẾT:
1. Đọc kỹ câu trả lời gốc được cung cấp
2. Đánh giá từng câu hỏi liên quan một cách độc lập
3. Xác định xem câu trả lời gốc có thể trả lời được câu hỏi đó hay không

TIÊU CHÍ ĐÁNH GIÁ CHO TỪNG CÂU HỎI:
- ĐÁNH GIÁ = 1 nếu câu trả lời gốc CÓ THỂ trả lời được câu hỏi liên quan
- ĐÁNH GIÁ = 0 nếu câu trả lời gốc KHÔNG THỂ trả lời được câu hỏi liên quan
- Câu trả lời có thể trả lời trực tiếp hoặc suy luận logic từ nội dung

ĐỊNH DẠNG OUTPUT BẮT BUỘC:
Trả về CHỈ danh sách các số 0 hoặc 1, ngăn cách bởi dấu phẩy, KHÔNG có text khác:
Ví dụ: 1,0,1,1,0

LƯU Ý QUAN TRỌNG:
- Số lượng kết quả PHẢI BẰNG số lượng câu hỏi liên quan được đánh giá
- Thứ tự kết quả PHẢI tương ứng với thứ tự các câu hỏi
- KHÔNG giải thích, KHÔNG thêm text nào khác ngoài danh sách số"""
            }
        ]
        batch_content = f"CÂU TRẢ LỜI GỐC: {response}\n\n"
        for i, q in enumerate(related_questions):
            batch_content += f"CÂU HỎI LIÊN QUAN {i+1}: {q}\n"

        batch_judge_messages.append({
            "role": "user",
            "content": batch_content
        })

        batch_judged = nim.chat(batch_judge_messages, temperature=0.0, max_tokens=50).strip()
        print(f"🔍 [response_relevancy@k] Batch judgment result: '{batch_judged}'")

        
        try:
            # Extract numbers from response
            import re
            numbers = re.findall(r'[01]', batch_judged)
            if len(numbers) == len(related_questions):
                judged_results = [int(num) for num in numbers]
            else:
                # Fallback: assume all are answerable if parsing fails
                print(f"⚠️ [response_relevancy@k] Failed to parse batch results, using fallback (all 1s)")
                judged_results = [1] * len(related_questions)
        except Exception as e:
            print(f"⚠️ [response_relevancy@k] Error parsing batch results: {e}, using fallback (all 1s)")
            judged_results = [1] * len(related_questions)

        hits = sum(judged_results)
        relevancy_score = hits / len(related_questions) if len(related_questions) > 0 else 0.0

        print(f"📊 [response_relevancy@k] Relevancy evaluation:")
        print(f"   - Total related questions: {len(related_questions)}")
        print(f"   - Answerable questions: {hits}")
        print(f"   - Relevancy score: {relevancy_score:.4f}")

        total_relevancy += relevancy_score
        running_avg = total_relevancy / (idx + 1)
        print(f"📊 [response_relevancy@k] Running average: {running_avg:.4f}")
        print()

    mean_relevancy = total_relevancy / n if n > 0 else 0.0
    print(f"🎯 [response_relevancy@k] FINAL RESULT: Mean relevancy = {mean_relevancy:.4f}")
    return mean_relevancy

def noise_sensitivity_k(df_clb, df_train, embedding, vector_db, k=5, nim: NIMClient = None, use_hybrid: bool = True, hybrid_alpha: float = 0.7):
    print(f"🔍 [noise_sensitivity@k] Starting noise_sensitivity@k calculation with k={k}, hybrid={use_hybrid}, alpha={hybrid_alpha}")
    if nim is None:
        raise RuntimeError("❌ NIM client is required for noise_sensitivity_k.")
    total_sensitivity = 0.0
    n = len(df_train)
    print(f"📊 [noise_sensitivity@k] Processing {n} queries total")

    for idx, row in enumerate(df_train.iterrows()):
        row = row[1]
        query = row["Query"]
        gt_answer = row.get("Ground truth answer", "")

        print(f"🔍 [noise_sensitivity@k] Query {idx+1:3d}/{n}: '{query[:50]}...'")
        print(f"📋 [noise_sensitivity@k] GT Answer: '{gt_answer[:100]}...'")

        search = get_cached_search_results(query, embedding, vector_db, k, nim, use_hybrid, hybrid_alpha)
        results = search["results"][:k]

        print(f"🎯 [noise_sensitivity@k] Retrieved {len(results)} documents")

        context = "Content từ các tài liệu liên quan:\n"
        context += "\n".join([result["information"] for result in results])

        messages = [
            {
                "role": "system",
                "content": """Bạn là một trợ lý AI chuyên cung cấp thông tin về Câu lạc bộ Lập trình ProPTIT.
Bạn sẽ nhận được dữ liệu ngữ cảnh (context) từ một hệ thống Retrieval-Augmented Generation (RAG) chứa các thông tin chính xác về CLB.

NGUYÊN TẮC TRẢ LỜI BẮT BUỘC:
1. CHỈ sử dụng thông tin từ context được cung cấp để trả lời. KHÔNG được thêm thông tin ngoài context.
2. Trả lời NGẮN GỌN, CHÍNH XÁC, và TRỰC TIẾP vào câu hỏi.
3. KHÔNG được thêm lời chào hỏi, cảm ơn, hoặc câu xã giao không cần thiết.
4. KHÔNG được nói "Xin lỗi", "Tôi không biết", "Không có thông tin" - PHẢI trả lời dựa trên context có sẵn.
5. Nếu context không đủ, hãy suy luận LOGIC từ thông tin có sẵn mà KHÔNG bịa thêm.
6. Tập trung trả lời CÂU HỎI CHÍNH, bỏ qua thông tin không liên quan.
7. Sử dụng ngôn ngữ tự nhiên, dễ hiểu, phù hợp với phong cách trả lời của con người.
8. Ưu tiên xưng là "CLB" khi nói về tổ chức.
9. Sử dụng CÁC TỪ KHÓA và THỰC THỂ chính xác từ context để đảm bảo tính chính xác.
10. Trả lời bằng tiếng Việt, trừ khi câu hỏi yêu cầu khác.

Ví dụ tham khảo về phong cách trả lời:
- CLB luôn khuyến khích các bạn thuộc ngành khác tham gia nếu có đam mê.
- Trong quá trình training, CLB đánh giá cao sự tiến bộ và tinh thần học hỏi của các thành viên.
- Thành viên cần tuân thủ nội quy về trang phục, giờ giấc và thái độ trong các buổi sinh hoạt.
- CLB thường xuyên tổ chức các hoạt động giao lưu, hợp tác để mở rộng mạng lưới và chia sẻ kinh nghiệm.
- CLB thu thập phản hồi để không ngừng cải thiện chất lượng hoạt động.
- Khi tổ chức sự kiện, CLB chuẩn bị kỹ lưỡng từ nội dung đến hậu cần để đảm bảo thành công.

NHIỆM VỤ:
- Trả lời các câu hỏi về CLB Lập trình ProPTIT.
- Giữ câu trả lời trong phạm vi 2-4 câu, không lan man.
- Sử dụng thông tin chính xác từ context để đạt độ chính xác cao."""
            }
        ]
        messages.append({
            "role": "user",
            "content": context + "\n\nCâu hỏi: " + query
        })

        response = nim.chat(messages, temperature=0.1, max_tokens=512)  # Tăng temperature và max_tokens để cải thiện quality
        print(f"🤖 [noise_sensitivity@k] LLM Response: '{response[:100]}...'")

        # Split response into sentences and evaluate each using batch processing
        sentences = response.split('. ')
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            print(f"⚠️ [noise_sensitivity@k] No sentences found, skipping...")
            continue

        print(f"📝 [noise_sensitivity@k] Split into {len(sentences)} sentences for evaluation")

        batch_judge_messages = [
            {
                "role": "system",
                "content": """Bạn là một chuyên gia đánh giá Noise Sensitivity trong hệ thống RAG. Nhiệm vụ của bạn là đánh giá TẤT CẢ các câu được cung cấp cùng một lúc.

QUY TRÌNH ĐÁNH GIÁ CHI TIẾT:
1. Đọc kỹ câu hỏi của người dùng
2. Đọc kỹ ngữ cảnh được cung cấp
3. Đánh giá từng câu một cách độc lập
4. Xác định mức độ nhạy cảm của từng câu với noise

TIÊU CHÍ ĐÁNH GIÁ CHO TỪNG CÂU:
- ĐÁNH GIÁ = 1 nếu nội dung câu được ngữ cảnh hỗ trợ hoặc suy ra trực tiếp
- ĐÁNH GIÁ = 0 nếu nội dung câu không được ngữ cảnh hỗ trợ, và không thể suy ra từ đó

ĐỊNH DẠNG OUTPUT BẮT BUỘC:
Trả về CHỈ danh sách các số 0 hoặc 1, ngăn cách bởi dấu phẩy, KHÔNG có text khác:
Ví dụ: 1,0,1,1,0

LƯU Ý QUAN TRỌNG:
- Số lượng kết quả PHẢI BẰNG số lượng câu được đánh giá
- Thứ tự kết quả PHẢI tương ứng với thứ tự các câu
- KHÔNG giải thích, KHÔNG thêm text nào khác ngoài danh sách số"""
            }
        ]
        batch_content = f"CÂU HỎI: {query}\n\nNGỮ CẢNH:\n{context}\n\n"
        for i, sentence in enumerate(sentences):
            batch_content += f"CÂU {i+1}: {sentence}\n"

        batch_judge_messages.append({
            "role": "user",
            "content": batch_content
        })

        
        batch_judged = nim.chat(batch_judge_messages, temperature=0.1, max_tokens=512).strip()
        print(f"🔍 [noise_sensitivity@k] Batch judgment result: '{batch_judged}'")

        
        try:
            # Extract numbers from response
            import re
            numbers = re.findall(r'[01]', batch_judged)
            if len(numbers) == len(sentences):
                judged_results = [int(num) for num in numbers]
            else:
                # Fallback: assume all are supported if parsing fails
                print(f"⚠️ [noise_sensitivity@k] Failed to parse batch results, using fallback (all 1s)")
                judged_results = [1] * len(sentences)
        except Exception as e:
            print(f"⚠️ [noise_sensitivity@k] Error parsing batch results: {e}, using fallback (all 1s)")
            judged_results = [1] * len(sentences)

        hits = sum(1 for result in judged_results if result == 0)  # Count unsupported sentences
        sensitivity_score = hits / len(sentences) if len(sentences) > 0 else 0.0

        print(f"📊 [noise_sensitivity@k] Sensitivity evaluation:")
        print(f"   - Total sentences: {len(sentences)}")
        print(f"   - Unsupported sentences: {hits}")
        print(f"   - Sensitivity score: {sensitivity_score:.4f}")

        total_sensitivity += sensitivity_score
        running_avg = total_sensitivity / (idx + 1)
        print(f"📊 [noise_sensitivity@k] Running average: {running_avg:.4f}")
        print()

    mean_sensitivity = total_sensitivity / n if n > 0 else 0.0
    print(f"🎯 [noise_sensitivity@k] FINAL RESULT: Mean sensitivity = {mean_sensitivity:.4f}")
    return mean_sensitivity

# Calculate all retrieval metrics 
def calculate_metrics_retrieval(df_clb, df_train, embedding, vector_db, train: bool, nim: NIMClient, compute_llm_metrics: bool = True, use_hybrid: bool = True, hybrid_alpha: float = 0.7):
    """
    Returns a DataFrame with rows for k in {3,5,7} and metrics columns as BTC's file.
    """
    clear_search_cache()

    k_values = [3, 5, 7]
    metrics = {
        "K": [],
        "hit@k": [],
        "recall@k": [],
        "precision@k": [],
        "f1@k": [],
        "map@k": [],
        "mrr@k": [],
        "ndcg@k": [],
        "context_precision@k": [],
        "context_recall@k": [],
        "context_entities_recall@k": []
    }

    for k in k_values:
        print(f"\n================  Metrics @K={k}  ================")
        metrics["K"].append(k)
        h = hit_k(df_clb, df_train, embedding, vector_db, k, nim, use_hybrid, hybrid_alpha)
        r = recall_k(df_clb, df_train, embedding, vector_db, k, nim, use_hybrid, hybrid_alpha)
        p = precision_k(df_clb, df_train, embedding, vector_db, k, nim, use_hybrid, hybrid_alpha)
        f1 = f1_k(df_clb, df_train, embedding, vector_db, k, nim, use_hybrid, hybrid_alpha)
        mp = map_k(df_clb, df_train, embedding, vector_db, k, nim, use_hybrid, hybrid_alpha)
        mrr = mrr_k(df_clb, df_train, embedding, vector_db, k, nim, use_hybrid, hybrid_alpha)
        nd = ndcg_k(df_clb, df_train, embedding, vector_db, k, nim, use_hybrid, hybrid_alpha)

        # Compute LLM-based metrics only if requested
        if compute_llm_metrics:
            cp = context_precision_k(df_clb, df_train, embedding, vector_db, k, nim=nim, use_hybrid=use_hybrid, hybrid_alpha=hybrid_alpha)
            cr = context_recall_k(df_clb, df_train, embedding, vector_db, k, nim=nim, use_hybrid=use_hybrid, hybrid_alpha=hybrid_alpha)
            cer = context_entities_recall_k(df_clb, df_train, embedding, vector_db, k, nim=nim, use_hybrid=use_hybrid, hybrid_alpha=hybrid_alpha)
        else:
            print("⏭️ Skipping LLM-based metrics (context_precision@k, context_recall@k, context_entities_recall@k)")
            cp = 0.0
            cr = 0.0
            cer = 0.0

        metrics["hit@k"].append(round(h, 2))
        metrics["recall@k"].append(round(r, 2))
        metrics["precision@k"].append(round(p, 2))
        metrics["f1@k"].append(round(f1, 2))
        metrics["map@k"].append(round(mp, 2))
        metrics["mrr@k"].append(round(mrr, 2))
        metrics["ndcg@k"].append(round(nd, 2))
        metrics["context_precision@k"].append(round(cp, 2))
        metrics["context_recall@k"].append(round(cr, 2))
        metrics["context_entities_recall@k"].append(round(cer, 2))

    df = pd.DataFrame(metrics)
    out = "metrics_retrieval_train.csv" if train else "metrics_retrieval_test.csv"
    df.to_csv(out, index=False)
    print(f"\n💾 Saved retrieval metrics to {out}")
    return df

# Calculate all LLM answer metrics
def calculate_metrics_llm_answer(df_clb, df_train, embedding, vector_db, train: bool, nim: NIMClient, compute_llm_answer_metrics: bool = True, use_hybrid: bool = True, hybrid_alpha: float = 0.7):
    """
    Returns a DataFrame with rows for k in {3,5,7} and LLM answer metrics columns.
    """
    if not compute_llm_answer_metrics:
        print("⏭️ Skipping LLM answer metrics computation")
        return pd.DataFrame()
    
    k_values = [3, 5, 7]
    metrics = {
        "K": [],
        "string_presence@k": [],
        "rouge_l@k": [],
        "bleu_4@k": [],
        "groundedness@k": [],
        "response_relevancy@k": [],
        "noise_sensitivity@k": []
    }

    for k in k_values:
        print(f"\n================  LLM Answer Metrics @K={k}  ================")
        metrics["K"].append(k)
        sp = string_presence_k(df_clb, df_train, embedding, vector_db, k, nim=nim, use_hybrid=use_hybrid, hybrid_alpha=hybrid_alpha)
        rl = rouge_l_k(df_clb, df_train, embedding, vector_db, k, nim=nim, use_hybrid=use_hybrid, hybrid_alpha=hybrid_alpha)
        b4 = bleu_4_k(df_clb, df_train, embedding, vector_db, k, nim=nim, use_hybrid=use_hybrid, hybrid_alpha=hybrid_alpha)
        gr = groundedness_k(df_clb, df_train, embedding, vector_db, k, nim=nim, use_hybrid=use_hybrid, hybrid_alpha=hybrid_alpha)
        rr = response_relevancy_k(df_clb, df_train, embedding, vector_db, k, nim=nim, use_hybrid=use_hybrid, hybrid_alpha=hybrid_alpha)
        ns = noise_sensitivity_k(df_clb, df_train, embedding, vector_db, k, nim=nim, use_hybrid=use_hybrid, hybrid_alpha=hybrid_alpha)

        metrics["string_presence@k"].append(round(sp, 2))
        metrics["rouge_l@k"].append(round(rl, 2))
        metrics["bleu_4@k"].append(round(b4, 2))
        metrics["groundedness@k"].append(round(gr, 2))
        metrics["response_relevancy@k"].append(round(rr, 2))
        metrics["noise_sensitivity@k"].append(round(ns, 2))

    df = pd.DataFrame(metrics)
    out = "metrics_llm_answer_train.csv" if train else "metrics_llm_answer_test.csv"
    df.to_csv(out, index=False)
    print(f"\n💾 Saved LLM answer metrics to {out}")
    return df

def main():
    print("🚀 NeoRAG Cup 2025 — Local Pipeline with LLM Reranker\n")

    USE_TRAIN_DATA = False
    COMPUTE_LLM_METRICS = True
    COMPUTE_LLM_ANSWER_METRICS = True

    print("📋 Loading environment variables...")
    nim_api_key = os.environ.get("NIM_API_KEY", "")
    nim_base_url = os.environ.get("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
    mongodb_uri = os.environ.get("MONGODB_URI", "")

    if not nim_api_key:
        raise RuntimeError("❌ NIM_API_KEY not found in .env file")
    if not mongodb_uri:
        raise RuntimeError("❌ MONGODB_URI not found in .env file")

    print("✅ Environment variables loaded.")

    if not os.path.exists("CLB_PROPTIT.csv"):
        raise FileNotFoundError("❌ CLB_PROPTIT.csv not found.")

    if USE_TRAIN_DATA:
        data_file = "train_data_proptit.xlsx"
        is_train = True
        print(f"📋 Using {data_file} - TRAIN MODE")
    else:
        data_file = "test_data_proptit.xlsx"
        is_train = False
        print(f"📋 Using {data_file} - TEST MODE")

    if not os.path.exists(data_file):
        raise FileNotFoundError(f"❌ {data_file} not found.")

    df_clb = pd.read_csv("CLB_PROPTIT.csv")
    df_train = pd.read_excel(data_file)

    print(f"📄 Loaded: {len(df_clb)} CLB docs, {len(df_train)} queries.")

    embedding = Embeddings()
    vector_db = VectorDatabaseAtlas(mongodb_uri)

    col_name = "information"
    try:
        count_existing = vector_db.count_documents(col_name)
    except Exception:
        count_existing = 0

    if count_existing == 0:
        print("🧠 Building document embeddings...")
        cnt = 0
        for i, row in df_clb.iterrows():
            text = str(row.get("Văn bản", "")).strip()
            if not text:
                continue
            emb = embedding.encode(text)
            cnt += 1
            vector_db.insert_document(col_name, {
                "title": f"Document {cnt}",
                "information": text,
                "embedding": emb.tolist()
            })
        print(f"✅ Stored {cnt} documents.")
    else:
        print(f"✅ Found {count_existing} existing documents.")

    nim = NIMClient(api_key=nim_api_key, base_url=nim_base_url, model="meta/llama-3.1-405b-instruct")

    print("\n🔍 Computing retrieval metrics...")
    df_metrics = calculate_metrics_retrieval(df_clb, df_train, embedding, vector_db, train=is_train, nim=nim, compute_llm_metrics=COMPUTE_LLM_METRICS)
    print("\n📊 Retrieval Metrics:\n", df_metrics)

    print("\n🤖 Computing LLM answer quality metrics...")
    df_llm_answer_metrics = calculate_metrics_llm_answer(df_clb, df_train, embedding, vector_db, train=is_train, nim=nim, compute_llm_answer_metrics=COMPUTE_LLM_ANSWER_METRICS)
    if not df_llm_answer_metrics.empty:
        print("\n📊 LLM Answer Metrics:\n", df_llm_answer_metrics)

if __name__ == "__main__":
    main()
