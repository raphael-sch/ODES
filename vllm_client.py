import argparse
import re
import os
import time

import torch
from vllm import LLM, SamplingParams

from client import base_client

os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

parser = argparse.ArgumentParser()
parser.add_argument("--project_id", type=int, required=True)
parser.add_argument('--url', default="http://127.0.0.1:5000", type=str)
parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
parser.add_argument("--hf_token", type=str)
args = parser.parse_args()

print('args vllm', args)


def execute_job(job, llm, state):
    job_id = job['job_id']
    instances = job['instances']
    prompt = job['prompt']
    seed = job['seed']
    new_state = job['weight_state']
    hyperparameters = job['hyperparameters']
    sigma = hyperparameters['sigma'] # noise strength
    alpha = hyperparameters['alpha'] # learning rate
    is_dev = seed == -1 # dev batch

    print('old_state', state)
    print('new_state', new_state)
    assert state == new_state[:len(state)]
    state_delta = new_state[len(state):]
    print('state_delta', state_delta)

    # bring weights into desired state
    for group_members in state_delta:
        response = llm.collective_rpc(
            method=perturb_weights,
            kwargs=dict(group_weight=alpha, group_members=group_members, sign=1.0))
    state = new_state

    kl = None
    if is_dev:
        kl = get_kl(llm)

    if not is_dev:
        # apply exploration with seed
        response = llm.collective_rpc(
            method=perturb_weights,
            kwargs=dict(group_weight=1.0, group_members=[[sigma, seed]], sign=1.0))

    sampling_params = SamplingParams(temperature=hyperparameters['temperature'],
                                     max_tokens=hyperparameters['max_tokens'])
    prompts = [prompt + instance['text'] for instance in instances]
    t = time.time()
    outputs = llm.generate(prompts, sampling_params) # maybe .chat() ??
    s = time.time() - t

    if not is_dev:
        # remove seed exploration
        response = llm.collective_rpc(
            method=perturb_weights,
            kwargs=dict(group_weight=1.0, group_members=[[sigma, seed]], sign=-1.0))

    results = list()
    tokens = 0
    for output in outputs:
        generated_text = output.outputs[0].text
        tokens += len(output.outputs[0].token_ids)
        #print('generated_text', generated_text)
        answer = get_answer(generated_text)
        results.append(answer)
    print(f"job {job_id} executed")
    tokens_per_second = int(tokens / s)
    results = dict(results=results, tokens_per_second=tokens_per_second, kl=kl)
    return llm, state, results


def perturb_weights(worker, group_weight, group_members, sign):
    device = worker.device
    model = worker.model_runner.model

    for group_member in group_members:
        noise_weight, seed = group_member
        group_member[1] = torch.Generator(device=device).manual_seed(int(seed))

    for name, parameter in sorted(model.named_parameters()):
        if 'norm' in name.lower():
            continue
        if group_weight == 1.0 and len(group_members) == 1: # we perturb with one random seed
            noise_weight, gen = group_members[0]
            noise = torch.randn(parameter.shape, dtype=parameter.dtype, device=parameter.device, generator=gen)
            parameter.data.add_(noise, alpha=noise_weight * sign)
        else: # we update according to previous population rewards
            update_tensor = get_update_tensor(group_members, parameter)
            parameter.data.add_(update_tensor, alpha=sign * (group_weight / len(group_members)))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return None

def get_update_tensor(update_group, parameter):
    update = torch.zeros_like(parameter)
    for noise_weight, gen in update_group:
        noise = torch.randn(parameter.shape, dtype=parameter.dtype, device=parameter.device, generator=gen)
        update.data.add_(noise * noise_weight)
    return update

def get_kl(llm):
    all_logprobs = get_logprobs(llm)
    assert len(all_logprobs) == len(args.init_logprobs)
    kls = list()
    for logprobs, init_logprobs in zip(all_logprobs, args.init_logprobs):
        kl = torch.mean(torch.exp(init_logprobs) * (init_logprobs - logprobs.clamp(min=-1e9)), dim=-1)
        kls.append(kl.item())
    return sum(kls) / len(kls)



def get_logprobs(llm):
    sampling_params = SamplingParams(temperature=0.0,
                                     max_tokens=1,
                                     prompt_logprobs=1)
    outputs = llm.generate(args.init_logprob_texts, sampling_params)
    all_logprobs = list()
    for output in outputs:
        token_ids = output.prompt_token_ids[1:-1]
        token_dicts = output.prompt_logprobs[1:-1]
        assert len(token_ids) == len(token_dicts)
        logprob_values = list()
        for token_id, token_dict in zip(token_ids, token_dicts):
            logprob = token_dict[token_id].logprob
            logprob_values.append(logprob)
        logprob_values = torch.as_tensor(logprob_values, device='cpu')
        all_logprobs.append(logprob_values)
    return all_logprobs

def init_vllm():
    llm = LLM(model=args.model_name,
              hf_token=args.hf_token,
              enable_prefix_caching=False,
              gpu_memory_utilization=args.gpu_memory_utilization)
    state = []

    test_jon = args.test_job
    instances = test_jon['instances']
    prompt = test_jon['prompt']
    hyperparameters = test_jon['hyperparameters']
    sampling_params = SamplingParams(temperature=hyperparameters['temperature'],
                                     max_tokens=hyperparameters['max_tokens'])
    prompts = [prompt + instance['text'] for instance in instances]
    outputs = llm.generate(prompts, sampling_params)

    logprob_texts = list()
    for prompt, output in zip(prompts, outputs):
        generated_text = output.outputs[0].text
        logprob_texts.append(prompt + generated_text)
    args.init_logprob_texts = logprob_texts
    args.init_logprobs = get_logprobs(llm)
    return llm, state


def main():
    llm, state = init_vllm()
    while True:
        job = base_client.fetch_job(args.url, args.project_id, args.worker_id)
        llm, state, results = execute_job(job, llm, state)
        base_client.deliver_job_results(args.url, args.worker_id, job, results)

answer_pattern = re.compile(
    r"</think> ?<answer>(.*?)</answer>",
    flags=re.DOTALL
)

def get_answer(completion):
    answer_match = re.search(answer_pattern, completion)
    answer = answer_match.group(1) if answer_match else None
    return answer

def get_worker_name():
    name = 'CPU'
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
    if name.startswith('NVIDIA'):
        name = name[len('NVIDIA'):]
    return name.strip()

if __name__ == "__main__":
    worker_name = get_worker_name()
    worker_id, model_name, test_job = base_client.startup(args.url, args.project_id, worker_name)
    args.worker_id = worker_id
    args.model_name = model_name
    args.test_job = test_job
    main()
