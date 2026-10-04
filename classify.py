import json
import time
import tinker

SYSTEM_PROMPT = """You are an email classification assistant.
You classify emails into one of 4 labels:
- must_act: The user must take an action (e.g. pay a bill, reply to a question, click a confirmation link) or there will be a negative consequence.
- worth_a_look: A specific opportunity or personally relevant item (e.g. call for speakers, scholarship, hackathon, job opening fitting the user, tool release the user uses). Generic tech digests are noise.
- fyi: Informational but no action needed (e.g. receipts, shipping updates, bank statements).
- noise: Promotional, newsletters, generic marketing, social pings, etc.

Return strict JSON: {"label": "<label>", "why": "<under 10 words>", "deadline": "<YYYY-MM-DD or null>", "summary": "<brief summary>"}
"""

def format_chatml(msgs, base_model=False):
    s = ""
    for m in msgs:
        s += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
    s += "<|im_start|>assistant\n"
    if base_model:
        s += "<think>\n</think>\n"
    return s

async def classify_email(sampling_client, tokenizer, user_message, few_shot_msgs=None, base_model=False):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if few_shot_msgs:
        messages.extend(few_shot_msgs)
    messages.append({"role": "user", "content": user_message})
    
    prompt_str = format_chatml(messages, base_model=base_model)
    input_tokens = tokenizer.encode(prompt_str)
    prompt_token_count = len(input_tokens)
    model_input = tinker.types.ModelInput.from_ints(input_tokens)
    
    stop_strings = ["<|im_end|>"]
    params = tinker.types.SamplingParams(max_tokens=1024, temperature=0.0, stop=stop_strings)
    
    start_t = time.perf_counter()
    try:
        result = await sampling_client.sample_async(
            prompt=model_input,
            num_samples=1,
            sampling_params=params
        )
        completion_tokens = result.sequences[0].tokens
        completion_token_count = len(completion_tokens)
        content = tokenizer.decode(completion_tokens)
    except Exception as e:
        return None, 0, False, f"ERROR: {e}", prompt_token_count, 0
        
    latency = time.perf_counter() - start_t
    
    # parse json
    try:
        s = content.strip()
        if "<think>" in s and "</think>" in s:
            s = s[s.find("</think>")+8:].strip()
            
        if s.startswith("```json"): s = s[7:]
        if s.startswith("```"): s = s[3:]
        if s.endswith("```"): s = s[:-3]
        
        if "}" in s:
            s = s[:s.rfind("}")+1]
            
        parsed = json.loads(s.strip())
        valid_json = True
    except Exception as e:
        parsed = {
            "label": "invalid",
            "deadline": None,
            "why": "",
            "summary": ""
        }
        valid_json = False
        
    return parsed, latency, valid_json, content, prompt_token_count, completion_token_count
