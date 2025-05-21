from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import torch


model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-one-to-many-mmt")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-one-to-many-mmt", src_lang="en_XX")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("Type 'exit' to quit.\n")
while True:
    input_text = input("Enter a sentence in English: ")
    if input_text.lower() == "exit":
        break

    model_inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

    generated_tokens = model.generate(
        **model_inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id["hi_IN"]
    )

    translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    print(f"Translation (Hindi): {translated_text}\n")
