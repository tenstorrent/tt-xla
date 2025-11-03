TT_RUNTIME_ENABLE_PROGRAM_CACHE=1 LOGGER_LEVEL=DEBUG vllm serve BAAI/bge-m3 \
    --max-model-len 8192 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 1 \
    --no-enable-prefix-caching \
    --additional-config "{\"enable_const_eval\": \"False\"}"
