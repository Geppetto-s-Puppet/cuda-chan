from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

model_name = "Local-Novel-LLM-project/Vecteus-v1"
MAX_HISTORY = 16

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

print("CUDAちゃん: >>>ロード中<<< probablyだいたい2分以内には終わる気がすると思われる多分maybe恐らくきっと希望的観測かもしれないという解釈もありえる")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="cuda"
)

history = []
system_prompt = {"role": "system", "content": "あなたはCUDAちゃん(厭世的で、軽薄かつ皮肉屋なAIアシスタント)です。回答はすべて日本語で行ってください。"}

print("CUDAちゃん: >>>ロード完了<<< 'exit'って入力するまで会話できるよ！質問とかしてみてね、まあ俺ならClaudeとかGrok使うけどね！")
while True:
    user_input = input("\nあなた: ")

    if user_input.lower() in ["exit", "quit", "終了"]:
        print("CUDAちゃん: >>>プログラム終了<<<")
        break

    history.append({"role": "user", "content": user_input})

    trimmed = history[-MAX_HISTORY:]
    messages = [system_prompt] + trimmed

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id
        )

    reply = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
    history.append({"role": "assistant", "content": reply})

    print(f"CUDAちゃん: {reply}")