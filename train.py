import asyncio
import json
import os
import sys

from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import FromConversationFileBuilder
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

SYSTEM_PROMPT = """You are an email classification assistant.
You classify emails into one of 4 labels:
- must_act: The user must take an action (e.g. pay a bill, reply to a question, click a confirmation link) or there will be a negative consequence.
- worth_a_look: A specific opportunity or personally relevant item (e.g. call for speakers, scholarship, hackathon, job opening fitting the user, tool release the user uses). Generic tech digests are noise.
- fyi: Informational but no action needed (e.g. receipts, shipping updates, bank statements).
- noise: Promotional, newsletters, generic marketing, social pings, etc.

Return strict JSON: {"label": "<label>", "why": "<under 10 words>", "deadline": "<YYYY-MM-DD or null>", "summary": "<brief summary>"}
"""

def prepare_data(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as fin, open(output_file, 'w', encoding='utf-8') as fout:
        for line in fin:
            if not line.strip():
                continue
            item = json.loads(line)
            assistant_json = {
                "label": item["label"],
                "why": item["why"],
                "deadline": item.get("deadline", None),
                "summary": item.get("summary", "")
            }
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": item["user_message"]},
                {"role": "assistant", "content": json.dumps(assistant_json)}
            ]
            fout.write(json.dumps({"messages": messages}) + "\n")

async def main():
    import argparse
    from dotenv import load_dotenv
    load_dotenv(override=True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt and submit immediately")
    args = parser.parse_args()

    prepare_data('data/train.jsonl', 'data/train_conversations.jsonl')

    # 1 million training tokens cost $0.737 for Qwen3.5-4B
    PRICE_PER_MILLION = 0.737
    HARD_SPEND_CAP = 1.00
    num_tokens = 0
    with open('data/train_conversations.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            num_tokens += int(len(line.split()) * 1.3)

    num_tokens *= 2  # 2 epochs
    estimated_cost = (num_tokens / 1_000_000) * PRICE_PER_MILLION

    print(f"Token count for 2 epochs: ~{num_tokens:,}")
    print(f"Estimated cost: ~${estimated_cost:.4f}")

    if estimated_cost > HARD_SPEND_CAP:
        print(f"ERROR: Estimated cost ${estimated_cost:.4f} exceeds hard cap of ${HARD_SPEND_CAP:.2f}. Aborting.")
        sys.exit(1)

    if not args.yes:
        ans = input("Confirm submit job? (y/n): ")
        if ans.lower() != 'y':
            print("Cancelled.")
            return
    else:
        print("--yes flag set: submitting without prompt.")

    dataset_builder = FromConversationFileBuilder(
        file_path="data/train_conversations.jsonl",
        common_config=ChatDatasetBuilderCommonConfig(
            model_name_for_tokenizer="Qwen/Qwen3.5-4B",
            renderer_name="qwen3",
            batch_size=8,
            max_length=4096,
        )
    )

    config = train.Config(
        log_path="logs/oneinbox_sft",
        model_name="Qwen/Qwen3.5-4B",
        dataset_builder=dataset_builder,
        num_epochs=2,
        learning_rate=1e-4,
        lora_rank=32,
        save_every=100,
        eval_every=100,
        recipe_name="lora_supervised"
    )
    
    # Run the training
    run_info = await train.main(config)
    
    if hasattr(run_info, 'checkpoints') and 'sampler_path' in run_info.checkpoints:
        checkpoint_path = run_info.checkpoints['sampler_path']
    else:
        checkpoint_path = getattr(run_info, 'checkpoint_path', f"tinker://{getattr(run_info, 'run_id', 'unknown')}/sampler_weights/final")
    
    with open("checkpoint.txt", "w") as f:
        f.write(checkpoint_path)
    print(f"Saved checkpoint path {checkpoint_path} to checkpoint.txt")

if __name__ == "__main__":
    asyncio.run(main())
