import base64
import mimetypes
import time
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

# 1) 把本地图像转成 data:URL
def to_data_url(path: str) -> str:
    mime = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def llm_explain(image_path,
                image_type,
                model,
                base_url=None,
                api_key=None):
    client = ChatOpenAI(api_key=api_key,
                        base_url=base_url,
                        model=model)
    # model = "qwen-vl-plus",
    # base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1",
    # api_key = 'sk-659ce5172d2d4739a498d32167e2a1dc'
    data_url = to_data_url(image_path)
    start_time = time.time()
    try:
        messages = [
            SystemMessage(content="You are an expert in explainable artificial intelligence (XAI) and good at explaining any XAI plots"),
            HumanMessage(content=[
                {"type": "text", "text": f"This is a {image_type}，what does this plot indicate?"},
                {"type": "image_url", "image_url": {"url": data_url}}
            ])
        ]

        response = client.invoke(messages)

        end_time = time.time()
        processing_time = end_time - start_time

        '''需要的结果'''
        content = response.content.strip()
        success_info = (f"📋 XAI plot explanation results:"
                        f"{content}"
                        f"🕒 processing time: {processing_time:.2f} seconds")

        return success_info

    except Exception as e:
        end_time = time.time()
        processing_time = end_time - start_time
        wrong_info = (f"❌ Errors occurs during processing: {e}"
                      f"⏱️ Time used before failure: {processing_time:.2f} seconds")
        return wrong_info
