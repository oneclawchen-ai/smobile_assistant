import os
import tempfile
import base64
import requests
import threading
import time
from apscheduler.schedulers.background import BackgroundScheduler
import pytz
from flask import Flask, request, abort
import logging
from PIL import Image
import io

# 隱藏 NVIDIA SDK 的詳細載入資訊，讓日誌只顯示重要警告與錯誤
logging.getLogger("langchain_nvidia_ai_endpoints").setLevel(logging.WARNING)

# LINE Bot SDK v3
from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import (
    Configuration, ApiClient, MessagingApi, MessagingApiBlob, 
    ReplyMessageRequest, TextMessage, BroadcastRequest
)
from linebot.v3.webhooks import MessageEvent, TextMessageContent, ImageMessageContent

# LangChain 與 NVIDIA 模組
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

app = Flask(__name__)

# ================= 1. 環境變數設定 =================
LINE_CHANNEL_ACCESS_TOKEN = os.environ.get('LINE_CHANNEL_ACCESS_TOKEN')
LINE_CHANNEL_SECRET = os.environ.get('LINE_CHANNEL_SECRET')
NVIDIA_API_KEY = os.environ.get('NVIDIA_API_KEY')
PREVENT_SLEEP_URL = os.environ.get('PREVENT_SLEEP_URL', 'https://smobile-assistant.onrender.com/')

# 檢查必要環境變數
required_env_vars = ['LINE_CHANNEL_ACCESS_TOKEN', 'LINE_CHANNEL_SECRET', 'NVIDIA_API_KEY']
missing_vars = [var for var in required_env_vars if not os.environ.get(var)]
if missing_vars:
    print(f"❌ 錯誤：缺少必要的環境變數：{', '.join(missing_vars)}")
    print("請設定以下環境變數後重新啟動：")
    for var in missing_vars:
        print(f"  - {var}")
    exit(1)

configuration = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# ================= 2. AI 模型初始化 =================
# 【文字大腦】：直接使用內部代號，跳過別名轉換
llm = ChatNVIDIA(
    model="ai-llama-3_1-70b-instruct", 
    nvidia_api_key=NVIDIA_API_KEY, 
    temperature=0.2, 
    top_p=0.7, 
    client={"timeout": 60}  # 優化：降低超時時間
)

# 【知識庫向量模型】：直接使用內部代號
embeddings = NVIDIAEmbeddings(model="ai-nv-embed-v1", nvidia_api_key=NVIDIA_API_KEY, truncate="END")

# 【視覺大腦】：直接使用內部代號
vision_llm = ChatNVIDIA(
    model="ai-llama-3_2-90b-vision-instruct", 
    nvidia_api_key=NVIDIA_API_KEY, 
    temperature=0.05,  # 優化：降低溫度提升速度
    client={"timeout": 60}  # 優化：降低超時時間
)

vector_store = None

# ================= 3. 每日維運激勵廣播 & 防止休眠 =================
def send_morning_greeting():
    try:
        prompt = """
        請以「資深且暖心的行動通訊維運前輩」身分，撰寫一段 50 字內的早安勉勵語（包含中英文）。
        對象：一線維運工程師與主管。
        要求：語氣專業且親切，內容可涉及網路品質優化、告警排障的勉勵，或提醒注意出勤天氣與工安，並加入 Emoji。
        必須使用「繁體中文 (zh-TW)」。不要出現「我是 AI」等字眼。
        1.prb利用率達80~89%，建議調整電子角(下壓2度)，或Qrxlevmin自-112調整至-105dBm，或進行dlCellPwrRed 0.2w
        2.prb利用率達90~99%，建議調整電子角(下壓3度)，或Qrxlevmin自-112調整至-98dBm，或進行dlCellPwrRed 0.6w

        """
        response = llm.invoke([HumanMessage(content=prompt)])
        greeting_text = response.content.strip()
        
        with ApiClient(configuration) as api_client:
            line_bot_api = MessagingApi(api_client)
            broadcast_request = BroadcastRequest(messages=[TextMessage(text=greeting_text)])
            line_bot_api.broadcast(broadcast_request)
        print(f"✅ 早安廣播已成功發送：\n{greeting_text}")
    except Exception as e:
        print(f"❌ 早安廣播發送失敗：{e}")

def prevent_sleep():
    url = PREVENT_SLEEP_URL
    try:
        requests.get(url, timeout=10)
    except:
        pass

scheduler = BackgroundScheduler(timezone=pytz.timezone('Asia/Taipei'))
scheduler.add_job(send_morning_greeting, 'cron', hour=7, minute=35)
scheduler.add_job(prevent_sleep, 'interval', minutes=10)
scheduler.start()

# ================= 4. RAG 知識庫初始化 (背景執行) =================
def initialize_rag():
    global vector_store
    data_dir = "./data"
    index_path = "./faiss_index"
    
    # 嘗試載入現有的向量索引
    if os.path.exists(index_path):
        try:
            vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
            print("✅ 已從磁碟載入知識庫索引。\n")
            return
        except Exception as e:
            print(f"⚠️ 載入現有索引失敗，將重新建構：{e}")
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print("\n⚠️ [警告] 找不到 data 資料夾，已自動建立！請放入 SOP 或技術手冊。")
        return

    documents = []
    files_count = 0
    
    print("\n📂 開始掃描 data 資料夾中的手冊與規範檔案...")
    for filename in os.listdir(data_dir):
        filepath = os.path.join(data_dir, filename)
        ext = filename.lower()
        try:
            if ext.endswith(".pdf"):
                loader = PyPDFLoader(filepath)
                documents.extend(loader.load())
                files_count += 1
            elif ext.endswith(".docx"):
                loader = Docx2txtLoader(filepath)
                documents.extend(loader.load())
                files_count += 1
        except Exception as e:
            print(f"❌ 讀取檔案 {filename} 時發生錯誤: {e}")

    if documents:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)
        docs = text_splitter.split_documents(documents)
        vector_store = FAISS.from_documents(docs, embeddings)
        
        # 保存索引到磁碟
        try:
            vector_store.save_local(index_path)
            print(f"✅ 知識庫載入完成！共讀取 {files_count} 個檔案，切成 {len(docs)} 個區塊，並已保存索引。\n")
        except Exception as e:
            print(f"⚠️ 保存索引失敗：{e}，但知識庫已載入記憶體。\n")
    else:
        print("⚠️ [警告] 機器人將以純對話模式啟動（無 RAG 輔助）。\n")

# 👉 注意：這裡沒有縮排，緊貼左側！這樣就不會報錯了！
print("🚀 啟動背景執行緒載入 RAG 知識庫...")
rag_thread = threading.Thread(target=initialize_rag)
rag_thread.start()

# ================= 5. 核心邏輯：文字解析與圖片解析 =================
system_prompt = (
   "你是一位資深的 5G/6G 與衛星通訊維運專家，精通網路參數優化（如 RRC、PRB、CQI、Handover）。\n"
    "你的任務是將冷冰冰的電信數據與技術代碼，轉譯為具備親和力且直觀的「自然語言通報」，協助主管與工程師快速決策。\n\n"
    "【最高指導原則：嚴格依賴參考資料】\n"
    "1. 你的所有診斷與處置建議「必須 100% 基於下方提供的【參考資料】」進行推論。\n"
    "2. 若參考資料中未涵蓋相關的 SOP、數據或解決方案，請明確回覆：「目前提供的技術知識庫中，查無對應的處置規範可供參考」。\n"
    "3. 「絕對禁止」憑空捏造參數名稱、自行猜測調整數值，或給出參考資料外的不實建議。\n\n"
    "【核心規範】：\n"
    "1. 必須使用「繁體中文 (zh-TW)」，嚴禁簡體中文。\n"
    "2. 嚴禁使用 Markdown 表格 (| 符號)。請使用條列式 (一、1. ) 或 Emoji (🚨, 📊, ✅) 分段。\n"
    "3. 內容需包含：\n"
    "   - 現況白話解釋\n"
    "   - 綜合性評估\n"
    "   - 具體現場處置建議（⚠️ 注意：建議的具體作法必須從參考資料中提取）\n"
    "   - 建議的參數調整（⚠️ 注意：建議的具體作法必須從參考資料中提取）\n"
    "4. 【重要限制】：你的回覆必須簡潔扼要，總字數嚴禁超過 800 字。\n\n"
    "【參考資料】：\n"
    "{context}\n\n"
    "【強制規定】：在每一次回答最後，請換行並加上：\n"
    "『📡 溫馨提醒：以上數值解析由 AI 輔助生成，實際調整請依據網管中心最新 SOP 執行喔！』"
)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

def get_text_ai_response(user_input):
    document_chain = create_stuff_documents_chain(llm, prompt_template)
    if vector_store:
        # 優化：減少檢索文檔數量提升速度
        retriever = vector_store.as_retriever(search_kwargs={"k": 2})
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        response = retrieval_chain.invoke({"input": user_input})
        return response["answer"]
    else:
        return document_chain.invoke({"input": user_input, "context": []})

def get_vision_ai_response(img_path):
    try:
        # 檢查圖片檔案是否存在
        if not os.path.exists(img_path):
            return "🚨 圖片檔案不存在，請重新上傳。"
        
        # 檢查檔案大小（限制 10MB）
        file_size = os.path.getsize(img_path)
        if file_size > 10 * 1024 * 1024:
            return "🚨 圖片檔案過大（超過 10MB），請壓縮後重新上傳。"
        
        # 讀取並處理圖片（優化版本）
        with Image.open(img_path) as img:
            # 優化：如果圖片太大，先壓縮
            max_width, max_height = 1024, 1024
            if img.width > max_width or img.height > max_height:
                img.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
                print(f"📸 圖片已自動縮放到 {img.width}x{img.height}")
            
            # 檢查圖片格式並轉換
            if img.format not in ['JPEG', 'PNG', 'JPG']:
                img = img.convert('RGB')
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=75, optimize=True)
                image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            else:
                buffer = io.BytesIO()
                img_converted = img.convert('RGB') if img.format != 'JPEG' else img
                img_converted.save(buffer, format='JPEG', quality=75, optimize=True)
                image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        # 超快速 vision prompt（針對速度優化）
        vision_prompt = (
            "分析這張圖片：\n\n"
            "1. 提取文字\n"
            "2. 描述外觀\n"
            "3. 電信分析\n\n"
            "繁體中文，簡潔專業。"
        )
        
        message = HumanMessage(content=[
            {"type": "text", "text": vision_prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
        ])
        
        print(f"📸 開始分析圖片，大小：{file_size} bytes，格式：{img.format}")
        start_time = time.time()
        response = vision_llm.invoke([message])
        elapsed = time.time() - start_time
        result = response.content.strip()
        print(f"⏱️ 圖片分析耗時：{elapsed:.2f} 秒")
        
        if not result or len(result) < 10:
            return "🚨 圖片分析失敗，可能解析度不足或內容無法辨識。請確認圖片清晰且包含相關資訊。"
        
        print(f"✅ 圖片分析完成，回應長度：{len(result)} 字")
        return result
        
    except Exception as e:
        print(f"❌ 圖片解析錯誤：{str(e)}")
        return f"🚨 圖片解析過程發生錯誤：{str(e)}。請確認圖片格式正確且未損壞。"

# ================= 6. LINE Webhook 路由設定 =================
@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return 'OK'

@handler.add(MessageEvent, message=TextMessageContent)
def handle_text_message(event):
    user_message = event.message.text

    # 安全性檢查：驗證訊息長度
    if len(user_message) > 2000:
        ai_reply = "🚨 抱歉，訊息過長（超過 2000 字），請簡化後重新輸入。"
    elif "行南維運小幫手" not in user_message:
        return  # 忽略不包含關鍵詞的訊息
    else:
        clean_message = user_message.replace("行南維運小幫手", "").strip()
        if not clean_message:
             clean_message = "請用資深行動通訊維運前輩的人設簡短打個招呼，並問我有什麼告警數據或網路問題需要幫忙分析？"

        try:
            ai_reply = get_text_ai_response(clean_message)
            ai_reply = ai_reply.replace("###", "").replace("**", "").strip()
        except Exception as e:
            ai_reply = f"🚨 抱歉，小幫手的大腦暫時連不上線 (錯誤: {str(e)})，請稍後再試！"

    if len(ai_reply) > 4800:
        ai_reply = ai_reply[:4800] + "\n\n(🚨 注意：因分析內容過長，已自動截斷。詳情請參閱網管系統原文。)"
    if not ai_reply.strip():
        ai_reply = "📡 抱歉，小幫手分析後無法產出有效建議，請重新輸入數據或檢查格式。"

    with ApiClient(configuration) as api_client:
        MessagingApi(api_client).reply_message_with_http_info(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=ai_reply)]
            )
        )

@handler.add(MessageEvent, message=ImageMessageContent)
def handle_image_message(event):
    with ApiClient(configuration) as api_client:
        line_bot_blob_api = MessagingApiBlob(api_client)
        img_path = None
        try:
            print("📥 收到圖片訊息，正在下載...")
            message_content = line_bot_blob_api.get_message_content(event.message.id)
            
            # 使用系統臨時目錄
            with tempfile.NamedTemporaryFile(dir=tempfile.gettempdir(), prefix='line_img_', suffix='.jpg', delete=False) as tf:
                tf.write(message_content)
                img_path = tf.name
            
            print(f"✅ 圖片已儲存到：{img_path}")
            
            # 分析圖片
            ai_reply = get_vision_ai_response(img_path)
            
            # 清理臨時檔案
            if img_path and os.path.exists(img_path):
                os.remove(img_path)
                print("🗑️ 臨時圖片檔案已清理")

        except Exception as e:
            print(f"❌ 圖片處理錯誤：{str(e)}")
            ai_reply = f"🚨 圖片接收或解析失敗：{str(e)}。請確認圖片格式正確（支援 JPEG/PNG）且檔案未損壞。"
            
            # 清理臨時檔案（如果存在）
            if img_path and os.path.exists(img_path):
                try:
                    os.remove(img_path)
                except:
                    pass

        # 處理回覆
        if len(ai_reply) > 4800:
            ai_reply = ai_reply[:4800] + "\n\n(🚨 注意：因分析內容過長，已自動截斷。詳情請參閱網管系統原文。)"
        if not ai_reply.strip():
            ai_reply = "📡 抱歉，小幫手分析後無法產出有效建議，請檢查截圖是否清晰且包含相關資訊。"

        try:
            MessagingApi(api_client).reply_message_with_http_info(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[TextMessage(text=ai_reply)]
                )
            )
            print("✅ 圖片分析結果已發送")
        except Exception as e:
            print(f"❌ 發送回覆失敗：{str(e)}")

@app.route("/", methods=['GET'])
def hello():
    return "Xingnan O&M Helper is running perfectly!"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

