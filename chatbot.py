print("Starting the chatbot script...")
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import torch
print("CUDA Available:", torch.cuda.is_available())
print("GPU Count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))
import os
app = Flask(__name__)
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
chat_history_ids = None
@app.route('/chat', methods=['POST'])
def chat():
    global chat_history_ids

    user_input = request.json.get('message', '')

    # Encode user input and add it to the chat history
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

    # Append the new user input tokens to the chat history
    bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1) if chat_history_ids is not None else new_input_ids

    # Generate a response from the model
    chat_history_ids = model.generate(bot_input_ids, max_length=500, pad_token_id=tokenizer.eos_token_id)

    # Decode the response
    bot_response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

    return jsonify({'response': bot_response})

def fine_tune_model():
    # Load dataset (assuming you have a text file 'train.txt' with conversational data)
    dataset = load_dataset('text', data_files={'train': 'train.txt'})

    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        save_steps=10_000,
        save_total_limit=2,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
    )

    # Train the model
    trainer.train()

    # Save the fine-tuned model
    model.save_pretrained("./fine_tuned_model")
    tokenizer.save_pretrained("./fine_tuned_model")

    if __name__ == '__main__':
        # Check if fine-tuned model exists, otherwise use the pre-trained model
        if os.path.exists("./fine_tuned_model"):
            print("Loading fine-tuned model...")
            model = AutoModelForCausalLM.from_pretrained("./fine_tuned_model")
            tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_model")
        else:
            print("No fine-tuned model found. Using pre-trained model.")
    
        # Uncomment the line below to fine-tune the model (optional)
        # fine_tune_model()
    
        # Run the Flask server
        app.run(host='0.0.0.0', port=5000, debug=True)