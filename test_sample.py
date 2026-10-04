import asyncio
import tinker
from dotenv import load_dotenv

load_dotenv()

async def main():
    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(base_model="Qwen/Qwen3.6-35B-A3B")
    
    tokenizer = sampling_client.get_tokenizer()
    prompt = "Write one short email subject line about a hackathon deadline."
    print(f"Prompt: {prompt}")
    
    prompt_tokens = tokenizer.encode(prompt)
    model_input = tinker.types.ModelInput.from_ints(prompt_tokens)
    
    params = tinker.types.SamplingParams(max_tokens=50, temperature=0.7)
    
    result = await sampling_client.sample_async(
        prompt=model_input, 
        num_samples=1, 
        sampling_params=params
    )
    
    response_text = tokenizer.decode(result.sequences[0].tokens)
    print("Response:")
    print(response_text)

if __name__ == "__main__":
    asyncio.run(main())
