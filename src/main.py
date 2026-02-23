from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

# 1. 量子化の設定（RTX 4060のVRAM 8GBに最適化）
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",      # 高精度・高速な量子化
    bnb_4bit_use_double_quant=True, # さらにメモリを節約
    bnb_4bit_compute_dtype=torch.float16
)

model_name = "Local-Novel-LLM-project/Vecteus-v1"

print("--- モデル読み込み中 (約1分かかります) ---")

# 2. トークナイザーとモデルのロード
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="cuda"
)

print("--- 準備完了！AIとの対話を開始します ('exit'で終了) ---")

# 3. チャットループ
while True:
    user_input = input("\nあなた: ")
    
    # 終了判定
    if user_input.lower() in ["exit", "quit", "終了"]:
        print("バイバイ！またね！")
        break

    # メッセージの構築（ここで性格を固定）
    messages = [
        {"role": "system", "content": "あなたは親しみやすく、ユーモアのある日本人美少女アシスタントです。フレンドリーにタメ口で話して。"},
        {"role": "user", "content": user_input}
    ]

    # プロンプトの適用
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to("cuda")

    # 4. 推論実行（パラメータを調整して人間らしく）
    with torch.no_grad(): # メモリ節約
        outputs = model.generate(
            **inputs, 
            max_new_tokens=256,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id # 警告抑制
        )

    # 5. 回答の表示（skip_special_tokens=True で変な記号を消す）
    reply = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    print(f"AI: {reply.strip()}")