# Adapted from: https://github.com/GeeeekExplorer/nano-vllm/blob/main/bench.py

import time
from random import randint, seed

from minisgl.core import SamplingParams
from minisgl.llm import LLM


def _run_experiment(
    llm: LLM,
    prompt_token_ids: list[list[int]],
    sampling_params: list[SamplingParams],
) -> tuple[int, float, float]:
    t_start = time.time()
    llm.generate(prompt_token_ids, sampling_params)
    t_elapsed = time.time() - t_start
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / t_elapsed
    return total_tokens, t_elapsed, throughput


def main():
    seed(0)
    num_seqs = 256
    max_input_len = 1024
    max_output_len = 1024
    num_runs = 5

    # align the hyperparameters
    llm = LLM(
        "/home/wujl022/models/Huggingface/Meta-Llama-3-8B-Instruct-GPTQ-4bit-gs128",
        max_seq_len_override=4096,
        max_extend_tokens=16384,
        cuda_graph_max_bs=256,
    )

    prompt_token_ids = [
        [randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)
    ]
    sampling_params = [
        SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_output_len))
        for _ in range(num_seqs)
    ]

    llm.generate(["Benchmark: "], SamplingParams(temperature=0.1))

    times: list[float] = []
    throughputs: list[float] = []
    total_tokens_list: list[int] = []

    for i in range(1, num_runs + 1):
        total_tokens, t, throughput = _run_experiment(llm, prompt_token_ids, sampling_params)
        total_tokens_list.append(total_tokens)
        times.append(t)
        throughputs.append(throughput)
        print(
            f"Run {i}: Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s"
        )

    avg_time = sum(times) / num_runs
    avg_throughput = sum(throughputs) / num_runs
    print(
        f"\nAverage over {num_runs} runs: Time: {avg_time:.2f}s, Throughput: {avg_throughput:.2f}tok/s"
    )


if __name__ == "__main__":
    main()
