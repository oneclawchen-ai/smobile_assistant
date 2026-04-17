import os
import tempfile
import base64
import requests
import threading
from apscheduler.schedulers.background import BackgroundScheduler
import pytz
from flask import Flask, request, abort
import logging

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

configuration = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# ================= 2. AI 模型初始化 =================
# 【文字大腦】：直接使用內部代號，跳過別名轉換
llm = ChatNVIDIA(model="ai-llama-3_1-70b-instruct", nvidia_api_key=NVIDIA_API_KEY, temperature=0.2, top_p=0.7, client={"timeout": 120})

# 【知識庫向量模型】：直接使用內部代號
embeddings = NVIDIAEmbeddings(model="ai-nv-embed-v1", nvidia_api_key=NVIDIA_API_KEY, truncate="END")

# 【視覺大腦】：直接使用內部代號
vision_llm = ChatNVIDIA(model="ai-llama-3_2-90b-vision-instruct", nvidia_api_key=NVIDIA_API_KEY, temperature=0.1, client={"timeout": 120})

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
    url = "https://smobile-assistant.onrender.com/" 
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
        print(f"✅ 知識庫載入完成！共讀取 {files_count} 個檔案，切成 {len(docs)} 個區塊。\n")
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
        retriever = vector_store.as_retriever(search_kwargs={"k": 4})
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        response = retrieval_chain.invoke({"input": user_input})
        return response["answer"]
    else:
        return document_chain.invoke({"input": user_input, "context": []})

def get_vision_ai_response(img_path):
    try:
        with open(img_path, "rb") as image_file:
            image_b64 = base64.b64encode(image_file.read()).decode("utf-8")
        
        vision_prompt = (
           "你是一位具備頂尖 OCR 辨識與邏輯分析能力的專家。請仔細觀看這張圖片，並「嚴格遵守」以下步驟進行回覆：\n\n"
            "第一步【強制 OCR 擷取】：\n"
            "請先把你雙眼在圖片中看到的「所有文字」一字不漏地擷取出來。若為表格，請盡量還原欄位對應關係。如果圖片中完全沒有文字，請明確回答「本圖片無文字」。\n\n"
            "第二步【客觀畫面描述】：\n"
            "請以最客觀的角度描述圖片的外觀與排版（例如：這是一張白底黑字的表格截圖、這是一張折線圖、這是一張設備實體照片）。\n"
            "🚨【最高警告】：絕對禁止憑空捏造圖片中不存在的物件（如風景、山丘、人物等）。只要你沒看到的，就不准寫！\n\n"
            "第三步【專業深度解析】：\n"
            "請基於「第一步」擷取出來的真實文字與「第二步」的客觀畫面，進行邏輯分析：\n"
            " - 若內容涉及行動通訊或網管參數（如 IMMLB、Handover、PRB），請以資深電信維運專家的角度，解釋這些參數的用途與優化目的。\n"
            " - 若為其他領域圖表或報表，請總結其核心重點。\n\n"
            "第四步【排版規範】：\n"
            "請全程使用繁體中文 (zh-TW)，嚴格按照上述「第一步」至「第三步」的架構條列式輸出。不要使用 Markdown 表格。若圖片過於模糊無法辨識，請直接回答「圖片解析度不足，無法準確辨識」。"
        )
        
        message = HumanMessage(content=[
            {"type": "text", "text": vision_prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
        ])
        response = vision_llm.invoke([message])
        return response.content.strip()
    except Exception as e:
        return f"🚨 圖片解析過程發生錯誤：{str(e)}"

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
    if "行南維運小幫手" not in user_message:
        return
        
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
        try:
            message_content = line_bot_blob_api.get_message_content(event.message.id)
            with tempfile.NamedTemporaryFile(dir='/tmp', prefix='line_img_', suffix='.jpg', delete=False) as tf:
                tf.write(message_content)
                img_path = tf.name

            ai_reply = get_vision_ai_response(img_path)
            ai_reply = ai_reply.replace("###", "").replace("**", "").strip()
            os.remove(img_path)

        except Exception as e:
            ai_reply = f"🚨 圖片接收或解析失敗：{str(e)}"

        if len(ai_reply) > 4800:
            ai_reply = ai_reply[:4800] + "\n\n(🚨 注意：因分析內容過長，已自動截斷。詳情請參閱網管系統原文。)"
        if not ai_reply.strip():
            ai_reply = "📡 抱歉，小幫手分析後無法產出有效建議，請檢查截圖是否清晰。"

        MessagingApi(api_client).reply_message_with_http_info(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=ai_reply)]
            )
        )

@app.route("/", methods=['GET'])
def hello():
    return "Xingnan O&M Helper is running perfectly!"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
