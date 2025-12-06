import tinker

try:
    print("🔌 Connecting to Tinker...")
    client = tinker.ServiceClient()
    
    # Use Llama 3.1 8B Instruct (Supported)
    model = "meta-llama/Llama-3.1-8B-Instruct"
    
    print(f"🎯 Targeted Model: {model}")

    # 1. We need a Training Client to get the tokenizer
    print("📖 Fetching Tokenizer...")
    training_client = client.create_lora_training_client(base_model=model)
    tokenizer = training_client.get_tokenizer()

    # 2. We need a Sampling Client to run inference
    print("🧪 Creating Sampler...")
    sampling_client = client.create_sampling_client(base_model=model)
    
    # 3. Run the sample
    print("🚀 Sending Request...")
    # Using specific tokenizer encoding to ensure compatibility
    prompt_tokens = tokenizer.encode("Hello!")
    prompt = tinker.ModelInput.from_ints(prompt_tokens)
    
    future = sampling_client.sample(
        prompt=prompt, 
        num_samples=1, 
        sampling_params=tinker.SamplingParams(max_tokens=10)
    )
    result = future.result()
    
    # Decode result
    text = tokenizer.decode(result.sequences[0].tokens)
    print(f"✅ Success! Model said: '{text}'")

except Exception as e:
    print(f"❌ Error: {e}")
