import base64
from openai import OpenAI
import time
from openai import RateLimitError
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

client = OpenAI(
    base_url='https://qianfan.baidubce.com/v2',
    api_key=''  # Please replace it with the API Key you obtained from Baidu Smart Cloud.
)
def analyze_video_with_qianfan(video_path, original_prompt,maxtokens,save_path):
    with open(video_path, "rb") as f:
        video_data = base64.b64encode(f.read()).decode('utf-8')



    retry_count = 0
    max_wait_time = 600  

    while True:
        try:
            response = client.chat.completions.create(
                model="ernie-4.5-turbo-vl-latest", 
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video_url",
                                "video_url": {
                                    "url": f"data:video/mp4;base64,{video_data}"
                                }
                            },
                            {
                                "type": "text",
                                "text": "Please describe the main actions, scenes, and key elements of the current video in one paragraph. At the same time, do not pay attention to the video duration and the face, but you need to pay attention to the gender. Only output this one paragraph and do not add additional supplementary descriptions."  
                            }
                        ]
                    }
                ],
                temperature=0.2, 
                top_p=0.8,
                extra_body={ 
                    "penalty_score": 1, 
                    "stop": []
                }
            )

            para2=response.choices[0].message.content
            break
        except RateLimitError as e:
            retry_count += 1
            wait_time = min(60 * (2 ** (retry_count - 1)), max_wait_time)
            print(f"Encountered TPM restriction, waiting for {wait_time} seconds before retrying... (Retry {retry_count})")
            time.sleep(wait_time)
        except Exception as e:
            print(f"Other errors occurred: {e}")
            headers = e.response.headers
            request_id = headers.get('x-request-id')
            print(f"ID: {request_id}")
            break


    retry_count = 0
    max_wait_time = 600 

    while True:
        try:
            response = client.chat.completions.create(
                model="ernie-4.5-turbo-vl-latest", 
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video_url",
                                "video_url": {
                                    "url": f"data:video/mp4;base64,{video_data}"
                                }
                            },
                            {
                                "type": "text",
                                "text": "first paragraph: " + original_prompt
                            },
                            {
                                "type": "text",
                                "text": "second paragraph: " + para2
                            },
                            {
                                "type": "text",
                                "text": """Please process my input in accordance with the following instructions.Rules:
    1. Read the two paragraphs of text and a video provided.
    2. Perform four comparative analyses in sequence:
        a. Compare the first paragraph of text with the video and identify contradictions between them.
        b. Compare the first paragraph of text with the second paragraph of text and identify contradictions between them.
        c. Analyze the second paragraph of text and identify parts that are not in line with common sense.
        d. Analyze the video and identify parts that are not in line with common sense.
    3. Integrate the four analyses and remove duplicate parts. If there are overlapping parts between contradictions and elements not in line with common sense, give priority to retaining the contradiction parts.
    Attention: Do not add additional titles, content, descriptions, or explanations to the output. And I need you to uniformly describe that in the output, replace the first paragraph with the prompt, and replace the second paragraph with the video. 
    if there is no \"Not in line with common sense\" just skip this part in the output.
    Don't mention the parts where there are no contradictions.
    4. Output the results in the format given below.
    Format:
    Contradictions:
    1.
    2.
    3.
    ...
    Not in line with common sense:
    1.
    2.
    3.
    ...
    """
                            }
                        ]
                    }
                ],
                max_tokens=maxtokens,
                temperature=0.2, 
                top_p=0.8,
                extra_body={ 
                    "penalty_score": 1, 
                    
                    "stop": []
                }
            )
            resp=response.choices[0].message.content
            print(resp)
            print("\n")
            #print(response)
            break
        except RateLimitError as e:
            retry_count += 1
            wait_time = min(60 * (2 ** (retry_count - 1)), max_wait_time)
            print(f"Encountered TPM restriction, waiting for {wait_time} seconds before retrying... (Retry {retry_count})")
            time.sleep(wait_time)
        except Exception as e:
            print(f"Other errors occurred: {e}")
            exit(1)

    lines = [line.rstrip('\n') for line in resp.splitlines()]

    contradictions_title = "Contradictions:"
    common_sense_title = "Not in line with common sense:"

    contradictions_list = []
    contradictions_start = None
    for i, line in enumerate(lines):
        if line == contradictions_title:
            contradictions_start = i
            break

    if contradictions_start is not None:
        for line in lines[contradictions_start + 1 :]:
            if line == common_sense_title or not line.strip():
                break
            if line.strip():
                contradictions_list.append(line[3:])

    common_sense_list = []
    common_sense_start = None
    for i, line in enumerate(lines):
        if line == common_sense_title:
            common_sense_start = i
            break

    if common_sense_start is not None:

        for line in lines[common_sense_start + 1 :]:
            if not line.strip():  
                break
            common_sense_list.append(line[3:])
    i=1
    mypro=""
    for item in contradictions_list:
        mypro+=f"{i}. {item} Please strengthen the description of the prompt at this point based on this contradiction, making the prompt more specific and detailed to avoid contradictions between the generated video and the prompt.\n"
        i+=1
    if common_sense_start is not None:
        for item in common_sense_list:
            mypro+=f"{i}. {item} Add, Modify or strengthen the descriptions in the prompt according to the unreasonable aspects to prevent such issues from occurring in the video.\n"
            i+=1

    #print(mypro)
    #print("\n")

    retry_count = 0
    max_wait_time = 600 
    my_prompt = "original prompt:\"" + original_prompt + "\"\n"
    my_prompt +="""Please first read the prompt I send.I'm optimizing the prompt for video generation. 
    Please analyze each point mentioned below and modify the prompt passage, strengthening its description to make it more specific and detailed.
    Please note that the modifications should maintain the original meaning of the prompt as much as possible, unless it is inherently unreasonable. Our goal is to strengthen its narrative so that the video has as few contradictions and unreasonable elements as possible.
    """
    my_prompt+=mypro+"\n"
    my_prompt+="Attention: Output the revised prompt without any titles, like \"Revised prompt: \" or summaries or reasons.\n"

    #print(my_prompt)
    new_prompt=""

    while True:
        try:
            response = client.chat.completions.create(
                model="ernie-4.5-turbo-vl-latest", 
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": my_prompt
                            }
                        ]
                    }
                ],
                temperature=0.2, 
                top_p=0.8,
                extra_body={ 
                    "penalty_score": 1, 
                    "stop": []
                }
            )
            new_prompt=response.choices[0].message.content
            print(new_prompt)
            #print(response)
            break
        except RateLimitError as e:
            retry_count += 1
            wait_time = min(60 * (2 ** (retry_count - 1)), max_wait_time)
            print(f"Encountered TPM restriction, waiting for {wait_time} seconds before retrying... (Retry {retry_count})")
            time.sleep(wait_time)
        except Exception as e:
            print(f"Other errors occurred: {e}")
            break
    try:
        import os
        # Ensure the save directory exists (create if it doesn't)
        os.makedirs(save_path, exist_ok=True)
        
        video_filename = os.path.basename(video_path) 
        txt_filename = f"analysis_{os.path.splitext(video_filename)[0]}.txt"  
        txt_save_path = os.path.join(save_path, txt_filename)  #
        content=resp+"\n"+response.choices[0].message.content
        with open(txt_save_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"\nThe analysis results have been saved to:{txt_save_path}")
    except Exception as e:
        print(f"\nFailed to save TXT file: {e}")
    return new_prompt
if __name__ == "__main__":
    # Default parameters when running standalone (for testing only)
    video_path = "result/42_0.1_0009.mp4"
    prompt = "A woman stands in a sunlit flower shop..."
    analyze_video_with_qianfan(video_path, prompt,maxtokens=8000,save_path="./result3")