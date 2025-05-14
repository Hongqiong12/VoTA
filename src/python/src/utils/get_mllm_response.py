# -*- coding: utf-8 -*-            
# @Time : 2025/5/14 14:44
# @Autor: joanzhong
# @FileName: get_mllm_response.py
# @Software: IntelliJ IDEA
import base64
import time
import openai

from mimetypes import guess_type


def local_image_to_data_url(image_path):
    """
    Read local images and convert them into base64 encoding
    :param image_path: 本地的图像地址
    :return:
    """
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'

    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')
    return f"data:{mime_type};base64,{base64_encoded_data}"


def query_base(model, prompt, api_key, image_path=None):
    # Initialize the OpenAI client based on the model specified
    if model in ['gpt', 'gemini-1.5-pro']:
        client = openai.OpenAI(api_key=api_key)
    elif 'qwen' in model:
        client = openai.OpenAI(
            api_key=api_key,
            ase_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    else:
        raise ValueError(f"Unknown model: {model}")

    # Prepare the message content depending on whether an image is included
    message_content = [
        {
            "role": "user",
            "content": {
                "type": "text",
                "text": prompt
            }
        }
    ]

    # If an image path is provided, add it to the message
    if image_path is not None:
        message_content.append(
            {
                "role": "user",
                "content": {
                    "type": "image_url",
                    "image_url": {
                        "url": local_image_to_data_url(image_path)
                    }
                }
            }
        )

    count = 0
    while True:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=message_content,
                max_tokens=2000,
                temperature=0.7
            )
            # Check and handle any refusal message in the response
            if response.choices[0].message.refusal:
                return response.choices[0].message.refusal

            return response.choices[0].message.content
        except Exception as e:
            if "Error code: 400" in str(e):
                return "I'm sorry."
            else:
                count += 1
                if count == 3:
                    raise e
                time.sleep(5)
                continue

