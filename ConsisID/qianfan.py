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
                model="ernie-4.5-turbo-vl", 
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
                model="ernie-4.5-turbo-vl", 
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
                                "text": "Original prompt: " + original_prompt
                            },
                            {
                                "type": "text",
                                "text": "Video description text: " + para2
                            },
                            {
                                "type": "text",
                                "text": """Please process my input in accordance with the following instructions.
    1. Read the two paragraphs of text and a video provided.
    2. Perform four comparative analyses in sequence:
        a. Compare the original prompt I uploaded with the video description text I uploaded, and find the parts in the video description text that do not conform to the description in the original prompt. That is, based on the content of the original prompt, find the parts in the video description text that are semantically inconsistent with it. Note: If the description in the video description text is a supplement to the relevant part of the original prompt and does not change the core meaning of this part, it is ok and no need to be outputed.
        b. Compare the original prompt I uploaded with the video I uploaded, and identify parts in the video that do not match the original prompt. This includes any omissions, deviations, semantic inconsistencies, or reversed priorities in the video's main subject, actions, scenes, quantities, attributes, styles, spatial relationships, and timelines. Note: When the core theme content is not affected, additional content is allowed and is regarded as a correct supplement and does not need to be outputed.
        c. Analyze the video description text I uploaded and identify parts where the narrative is ambiguous, contrary to common sense, or has errors or confusion in terms of space and time.
        d. Analyze the video I uploaded. Check if there are parts where the form, features, attributes, or quantity of the core subject throughout the video are inconsistent. Also, check if there are parts with screen flickering, intense and abnormal camera jitter, abnormal inter - frame picture quality in the video frames. And check if there are parts in the video content that do not conform to the physical laws of the real world.
    3. First, integrate Analysis a and Analysis b to remove duplicate parts. Note that when integrating a and b, do not regard supplementary content without conflicts as one of the problems. Then, integrate Analysis c and Analysis d to remove duplicate parts. Based on the results of the first two integrations, conduct a final integration and deduplication, and then output.
    4. 
    5.If the analysis results indicate an alignment issue, output in the following format:
    Alignment issue:
    1.
    2.
    3.
    ...
    If the analysis results indicate a consistency problem, output in the following format:
    Consistency issue:
    1.
    2.
    3.
    ...
    If the analysis results indicate a stability problem, output in the following format:
    Stability issue:
    1.
    2.
    3.
    ...
    If the analysis results indicate a physical rationality problem, output in the following format:
    Physical issue:
    1.
    2.
    3.
    ...
    Attention: If there are problems in multiple aspects, output them in sequence. Note that no additional titles, content, descriptions, or explanations should be added to the output. Comply with the format completely.
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
            #print(resp)
            #print("\n")
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
    resp="""You are a strict text review assistant. The user will provide a piece of text containing multiple sections, each starting with a heading like "XXX issue:" followed by specific descriptive content. Your task is as follows:

Read each section and determine whether the content **explicitly identifies a real, actual problem** or whether the content indicates that **there is no actual issue in this section** (i.e., the problem is not present or does not apply).

Criteria (based on semantic meaning):
- If the section describes a **defect, error, inconsistency, contradiction, or implausibility**, treat it as **"has an actual problem"** and retain the entire original text of that section (including the heading and description).
- If the section indicates that **there is no problem here** (e.g., containing phrases like "No issues found", "Nothing to report", "None"), or is empty, or states that the check passed without substantive criticism, treat it as **"has no actual problem"** and delete that entire section completely (do not keep any text from it).

Output requirements:
- Output **only** the full sections that were retained.
- Preserve the original blank lines or spacing between the retained sections.
- If all sections are deleted, output an empty string (or the phrase "No issues").
- Do **not** add any extra commentary, explanations, or modifications to the original wording.

Example 1 (Input):
Alignment issue:
1. The original prompt specifies 'no ground lights or artificial illumination visible', but the video description text mentions a 'female figure with blonde hair' superimposed onto the landscape, which introduces an artificial element not present in the original prompt.

Consistency issue:
No consistency issues found. The text adheres to the prompt.

Stability issue:
No stability issues found. The text adheres to the prompt.

Physical issue:
1. The superimposition of a 'female figure with blonde hair' onto the landscape is not a natural occurrence and does not conform to the physical laws of the real world.

Example 1 (Output):
Alignment issue:
1. The original prompt specifies 'no ground lights or artificial illumination visible', but the video description text mentions a 'female figure with blonde hair' superimposed onto the landscape, which introduces an artificial element not present in the original prompt.

Physical issue:
1. The superimposition of a 'female figure with blonde hair' onto the landscape is not a natural occurrence and does not conform to the physical laws of the real world.

(Note: The "Consistency issue" section was deleted because it contained no actual problem.)

Now, perform the above operation on the following user input:
"""+resp
    while True:
        try:
            response = client.chat.completions.create(
                model="ernie-4.5-turbo-vl", 
                messages=[
                    {
                        "role": "user",
                        "content": [
                            
                            {
                                "type": "text",
                                "text": resp
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

    
    align_title1 = "Alignment issue:"
    align_title2 = "alignment issue:"
    
    consist_title1 = "Consistency issue:"
    consist_title2 = "consistency issue:"
    
    stable_title1 = "Stability issue:"
    stable_title2 = "stability issue:"
   
    physical_title1 = "Physical issue:"
    physical_title2 = "physical issue:"

    
    alignment_list = []
    consistency_list = []
    stability_list = []
    physical_list = []

    
    all_titles = [align_title1, align_title2, consist_title1, consist_title2,
                stable_title1, stable_title2, physical_title1, physical_title2]

    # ---------------------- Alignment  ----------------------
    alignment_start = None
    for i, line in enumerate(lines):
        if line == align_title1 or line == align_title2:
            alignment_start = i
            break

    if alignment_start is not None:
        for line in lines[alignment_start + 1 :]:
            if line in all_titles or not line.strip():
                break
            if line.strip():
                alignment_list.append(line[3:])

    # ---------------------- Consistency ----------------------
    consistency_start = None
    for i, line in enumerate(lines):
        if line == consist_title1 or line == consist_title2:
            consistency_start = i
            break

    if consistency_start is not None:
        for line in lines[consistency_start + 1 :]:
            if line in all_titles or not line.strip():
                break
            if line.strip():
                consistency_list.append(line[3:])

    # ---------------------- Stability  ----------------------
    stability_start = None
    for i, line in enumerate(lines):
        if line == stable_title1 or line == stable_title2:
            stability_start = i
            break

    if stability_start is not None:
        for line in lines[stability_start + 1 :]:
            if line in all_titles or not line.strip():
                break
            if line.strip():
                stability_list.append(line[3:])

    # ---------------------- Physical  ----------------------
    physical_start = None
    for i, line in enumerate(lines):
        if line == physical_title1 or line == physical_title2:
            physical_start = i
            break

    if physical_start is not None:
        for line in lines[physical_start + 1 :]:
            if line in all_titles or not line.strip():
                break
            if line.strip():
                physical_list.append(line[3:])

    # ============================================
    mypro = ""
    p = 1
  
    if alignment_list:
        mypro += f"{p}. Alignment issue occurred in the video:"
        p += 1
        i = 1
        for item in alignment_list:
            mypro += f"{i}. {item} "
            i += 1
        mypro+="Modify the original prompt words for the parts where problems occur, and change this part to a more detailed description. Absolutely do not modify content unrelated to the problems, do not add irrelevant content such as styles, picture quality, and special effects that are not mentioned in the original prompt words, and ensure that resolving the alignment issue does not trigger new problems with consistency and stability.\n"

   
    if consistency_list:
        mypro += f"{p}. Consistency issue occurred in the video:"
        p += 1
        i = 1
        for item in consistency_list:
            mypro += f"{i}. {item} "
            i += 1
        mypro+="Based on these issues, add to the end of the prompt words. Ensure that the discovered XXX subject remains unchanged, with no addition or reduction. Ensure that the XXX subject always appears in the form of YYYY. (Fill in X and Y according to the description of the original prompt words and the issues.)\n"

    
    if stability_list:
        mypro += f"{p}. Stability issue occurred in the video:"
        p += 1
        i = 1
        for item in stability_list:
            mypro += f"{i}. {item} "
            i += 1
        mypro+="Based on these issues, provide a stable and mature camera movement plan that requires specific description. And add the camera movement method to the end of the prompt words.\n"

    
    if physical_list:
        mypro += f"{p}. Physical issue occurred in the video:"
        p += 1
        i = 1
        for item in physical_list:
            mypro += f"{i}. {item} "
            i += 1
        mypro+="Modify the description related to the problem in the prompt words or add constraints to avoid generating the same problem again. Please note that absolutely do not modify content unrelated to the problem to prevent triggering unknown issues.\n"

    #print(mypro)
    #print("\n")

    retry_count = 0
    max_wait_time = 600 
    my_prompt = "Original prompt:\"" + original_prompt + "\"\n"
    my_prompt +="""Please first read the original prompt I uploaded. You are a professional video prompt words optimizer. Please modify the prompt words based on each of the following points. Do not skip or overlook any. Note that the modifications should fully preserve the user's intent, neither adding to nor subtracting from the core theme part.
    """
    my_prompt+=mypro+"\n"
    my_prompt+="""After completing the above modifications, please appropriately polish the modified prompt from the following aspects:
    A. If the original prompt already contain relevant restrictions regarding stability, picture style, format, etc., it is prohibited to overlook or delete these contents. The description of this part can be further refined.
    B. If the scenery or object is the main subject, describe it accurately and in detail: Describe in detail the specific objects contained in the scenery, their compositional relationships, and their relative spatial layouts.
    C. The prompt should not contain evaluation or assessment statements (i.e., statements like 'it would be better if...' should not appear). If such statements exist, delete this part.
    D. The prompt do not need to describe counterexamples. If there are any, delete this part.
    E. The prompt should not add excessive restrictions on secondary content, as the prompt words need to highlight the core content.
    Please only output the final prompt words. Note: Do not add any titles, content, descriptions, or explanations to the output. Only output the modified prompt words.
    """

    #print(my_prompt)
    new_prompt=""

    while True:
        try:
            response = client.chat.completions.create(
                model="ernie-4.5-turbo-vl", 
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": my_prompt
                            },
                            {
                                "type": "text",
                                "text": """"""
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
            break
    try:
        import os
        # Ensure the save directory exists (create if it doesn't)
        os.makedirs(save_path, exist_ok=True)
        
        video_filename = os.path.basename(video_path) 
        txt_filename = f"analysis_{os.path.splitext(video_filename)[0]}.txt"  
        txt_save_path = os.path.join(save_path, txt_filename)  #
        content=resp+"\n"+response.choices[0].message.content
        with open(txt_save_path, 'a', encoding='utf-8') as f:
            f.write("\n"+content)
        
        print(f"\nThe analysis results have been saved to:{txt_save_path}")
    except Exception as e:
        print(f"\nFailed to save TXT file: {e}")
    return new_prompt
if __name__ == "__main__":
    # Default parameters when running standalone (for testing only)
    video_path = "resultvbench/0009.mp4"
    prompt = "A person using smartphone in home, while sitting on sofa."
    analyze_video_with_qianfan(video_path, prompt,maxtokens=8000,save_path="./resultvbench")