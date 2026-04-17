import os
import tempfile
import base64
import requests
import threading
from apscheduler.schedulers.background import BackgroundScheduler
import pytz
from flask import Flask, request, abort
import logging

# 隱藏 NVIDIA SDK 的詳細載入資訊
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
# 【文字大腦】
llm = ChatNVIDIA(model="ai-llama-3_1-70b-instruct", nvidia_api_key=NVIDIA_API_KEY, temperature=0.2, top_p=0.7, client={"timeout": 120})

# 【知識庫向量模型】
embeddings = NVIDIAEmbeddings(model="ai-nv-embed-v1", nvidia_api_key=NVIDIA_API_KEY, truncate="END")

# 【全方位視覺大腦】：調升溫度至 0.3 以利描述一般圖片，並增加 timeout 確保複雜圖表處理完成
vision_llm = ChatNVIDIA(model="ai-llama-3_2-90b-vision-instruct", nvidia_api_key=NVIDIA_API_KEY, temperature=0.3, client={"timeout": 150})

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
        print(f"✅ 早安廣播已成功發送")
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
        return

    documents = []
    for filename in os.listdir(data_dir):
        filepath = os.path.join(data_dir, filename)
        ext = filename.lower()
        try:
            if ext.endswith(".pdf"):
                loader = PyPDFLoader(filepath)
                documents.extend(loader.load())
            elif ext.endswith(".docx"):
                loader = Docx2txtLoader(filepath)
                documents.extend(loader.load())
        except Exception as e:
            print(f"❌ 讀取檔案 {filename} 錯誤: {e}")

    if documents:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)
        docs = text_splitter.split_documents(documents)
        vector_store = FAISS.from_documents(docs, embeddings)
        print(f"✅ 知識庫載入完成！共切成 {len(docs)} 個區塊。")

rag_thread = threading.Thread(target=initialize_rag)
rag_thread.start()

# ================= 5. 核心邏輯：文字解析與【升級版】視覺解析 =================
system_prompt = (
   "你是一位資深的 5G/6G 與衛星通訊維運專家。你的任務是協助主管與工程師快速決策。\n\n"
    "【最高指導原則：嚴格依賴參考資料】\n"
    "1. 你的所有診斷與處置建議必須基於提供的【參考資料】。\n"
    "2. 若參考資料中未涵蓋，請回答：「目前提供的技術知識庫中，查無對應的處置規範可供參考」。\n\n"
    "【核心規範】：\n"
    "1. 必須使用「繁體中文 (zh-TW)」。\n"
    "2. 嚴禁使用 Markdown 表格。請使用條列式。\n"
    "3. 字數嚴禁超過 800 字。\n\n"
    "【參考資料】：\n{context}\n\n"
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
    return document_chain.invoke({"input": user_input, "context": []})

def get_vision_ai_response(img_path):
    """
    修改後的視覺辨識功能：
    1. 支援一般圖片辨識與描述。
    2. 強度圖表趨勢與邏輯分析。
    3. 保留電信維運專業建議。
    """
    try:
        with open(img_path, "rb") as image_file:
            image_b64 = base64.b64encode(image_file.read()).decode("utf-8")
        
        vision_prompt = (
            "你是一位具備頂尖視覺辨識與多領域邏輯分析能力的 AI 助手。請觀察此圖片並執行以下分析：\n\n"
            "一、 【圖像分類與定性】：\n"
            "判斷圖片類型（如：一般生活照、機房/工程現場照、網管數據截圖、或效能走勢圖表）。\n\n"
            "二、 【核心內容提取】：\n"
            "1. 若有文字或數據，請精準擷取關鍵參數（如 Cell ID, PRB, 手寫筆記內容等）。\n"
            "2. 若為圖表（如折線圖、柱狀圖），請描述視覺上的趨勢（如：10點後數據陡升、曲線呈現週期性波動）。\n"
            "3. 若為一般照片，請描述主體物件、環境背景及其狀態。\n\n"
            "三、 【專業邏輯推論】：\n"
            "1. 針對通訊維運圖表：分析數據間的因果關係（例如：干擾上升與流量下降的同步性）。\n"
            "2. 針對一般/現場照片：指出觀察到的重點或潛在風險（如設備燈號、施工環境）。\n\n"
            "四、 【輸出規範】：\n"
            "請使用繁體中文 (zh-TW)，嚴禁 Markdown 表格，請條列式呈現。若圖片過於模糊請告知。"
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
         clean_message = "請用資深行動通訊維運前輩的人設簡短打個招呼，並問我有什麼需要幫忙分析？"

    try:
        ai_reply = get_text_ai_response(clean_message)
        ai_reply = ai_reply.replace("###", "").replace("**", "").strip()
    except Exception as e:
        ai_reply = f"🚨 抱歉，大腦暫時連不上線 (錯誤: {str(e)})"

    with ApiClient(configuration) as api_client:
        MessagingApi(api_client).reply_message(
            ReplyMessageRequest(reply_token=event.reply_token, messages=[TextMessage(text=ai_reply)])
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
            ai_reply = f"🚨 圖片解析失敗：{str(e)}"

        MessagingApi(api_client).reply_message(
            ReplyMessageRequest(reply_token=event.reply_token, messages=[TextMessage(text=ai_reply)])
        )

@app.route("/", methods=['GET'])
def hello():
    return "Xingnan O&M Helper is running perfectly!"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
