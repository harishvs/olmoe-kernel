from vllm import LLM,SamplingParams
import time

prompt="write python funtion to calculate first n numbers of fibonachi sequence"
llm = LLM(model="allenai/OLMoE-1B-7B-0924-Instruct")
t0 = time.perf_counter()
outputs = llm.generate([prompt], SamplingParams(max_tokens=1))
ttft_ms = (time.perf_counter() - t0) * 1000



print("outpuuuuuuut",outputs[0].outputs[0].text)
print("ttft_ms",ttft_ms)
