# The official vLLM image plus bitsandbytes, which serving the 4B student in 4 bits needs (serve.py).
FROM vllm/vllm-openai:v0.22.1
RUN pip install --no-cache-dir bitsandbytes==0.50.2
