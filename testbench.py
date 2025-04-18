from transformers import AutoTokenizer, AutoModelForCausalLM

# file_path = "/mnt/e/repos/Labs/Labs.ML/LLMs/Models/LLMs/Noromaid"
# filename = "Lumimaid-v0.2-12B-Q6_K_L.gguf"
file_path = "E:\\repos\\Labs\\Labs.ML\\LLMs\\Models\\LLMs\\Llama-3.2-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(file_path)
model = AutoModelForCausalLM.from_pretrained(file_path).to("cuda")

model_inputs = tokenizer(["The secret to baking a good cake is "], return_tensors="pt").to("cuda")
generated_ids = model.generate(**model_inputs, max_length=300)
tokenizer.batch_decode(generated_ids)[0]