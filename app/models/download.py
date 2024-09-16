from transformers import AutoModelForCausalLM, AutoTokenizer

# Specify the model ID (replace 'meta-llama/Llama-2-7b-hf' with your desired model)
model_id = "meta-llama/Llama-2-7b-hf"

# Define the local path where you want to store the model
save_directory = "/Users/bruce/Desktop/SCHOOL CHATBOT/models"

# Replace this with your Hugging Face token
hf_token = "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"

# Download the model
tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=hf_token)
model = AutoModelForCausalLM.from_pretrained(model_id, use_auth_token=hf_token)

# Save the tokenizer and model locally
tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)

