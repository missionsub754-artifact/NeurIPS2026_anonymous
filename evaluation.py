# import argparse
import json
import logging
import time
from typing import Literal, Tuple
import gc
import os 
# import numpy as np
import math
import torch
from inference.generate import BaseGenerator, SpeculativeGenerator
from inference.options import args_parser
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)


class JsonData:
    def __init__(self, path) -> None:
        with open(path) as fin:
            self.data = json.load(fin)

    def __getitem__(self, index) -> Tuple[str, str]:
        return self.data[index]

    def __len__(self):
        return len(self.data)


def run_eval(
        draft_model,
        target_model,
        draft_model_str,
        target_model_str,
        tokenizer,
        tokenizer_draft,
        fp16,
        n_config: int,
        datapath: str,
        max_new_tokens: int = 128,
        speculative_sampling=True,
        sampling_type="sampling",
        disable_tqdm: bool = False,
        num_iter: int = 1,
        verification_method: Literal["hr", "ul", "dl", "gr","tr", "srdv", "uhlm"] = "hr",
        K: int = 20,
        M: int = 10,
        u_s: int = 20,
        u_max: float = 2.0,
        u_th: float = 0.8,
):
    if verification_method not in ["hr", "ul", "dl", "gr","tr", "srdv", "uhlm"]:
        raise ValueError(
            f'`verification_method` can be "hr"/"ul"/"dl"/"gr"/"tr", but received "{verification_method}"'
        )

    target_model_temp = 0.3 # Softmax sampling
    draft_model_temp = 0.3 # Softmax sampling
    
    dataloader = JsonData(datapath)
    generator = SpeculativeGenerator(
        draft_model,
        target_model,
        tokenizer=tokenizer,
        tokenizer_draft=tokenizer_draft,
        verification_method=verification_method,
        fp16=fp16,
        eos_token_id=tokenizer.eos_token_id,
        n_config=n_config,
        max_new_tokens=max_new_tokens,
        draft_model_temp=draft_model_temp,
        target_model_temp=target_model_temp,
        speculative_sampling=speculative_sampling,
        K=K,
        M=M,
        u_s=u_s,
        u_max=u_max,
        u_th=u_th,
    )

    draft_model.eval()
    target_model.eval()

    logger.info("evaluation start.")
    start_time = time.time()

    acceptance_count = 0
    draft_token_count = 0
    invocation_count = 0
    latency_wireless = 0
    latency_draft = 0
    latency_verification = 0
    latency_comm = 0
    ul_load_fin = []
    dl_load_fin = []
    output_seq = []
    output_length = []

    iterator = range(len(dataloader))
    with torch.no_grad():
        for sample_idx in iterator if disable_tqdm else tqdm(iterator):
            prompt_text = dataloader[sample_idx]
            prompt_size = len(prompt_text.encode('utf-8')) * 8 # 1 Byte/char
            inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")
            inputs_draft = tokenizer_draft(prompt_text, return_tensors="pt").to("cuda")
            input_ids = inputs.input_ids
            input_ids_draft = inputs_draft.input_ids
            init_length = input_ids.size(1)
            output = generator.generate(input_ids, input_ids_draft)
            generated_text = tokenizer.decode(output.sequences[0, init_length:], skip_special_tokens=True)

            acceptance_count += output.acceptance_count
            draft_token_count += output.draft_token_count
            invocation_count += output.invocation_count
            latency_wireless += output.sample_time
            latency_draft += output.draft_time
            latency_verification += output.verification_time
            output.ul_load[0] += prompt_size
            ul_load_fin.append(output.ul_load)
            dl_load_fin.append(output.dl_load)
            output_length.append(output.sequences.size(1) - init_length)

            output_seq.append({"prompt": prompt_text, "output": generated_text})
    end_time = time.time()

    logger.info("evaluation complete.")

    run_time = end_time - start_time

    latency = run_time / (acceptance_count + invocation_count)
    acceptance_rate = acceptance_count / draft_token_count
    block_efficiency = 1 + acceptance_count / invocation_count
    latency_wo_comm = (latency_draft + latency_verification) / (acceptance_count + invocation_count)
    tput = (acceptance_count + invocation_count) / (latency_draft + latency_verification)
    tput_wireless = (acceptance_count + invocation_count) / latency_wireless
    latency_wireless = latency_wireless / (acceptance_count + invocation_count)

    logger.info("Running time: {:.2f} s".format(run_time))
    logger.info("Latency portion (draft): {:.2f} s".format(latency_draft / (latency_draft + latency_verification)))
    logger.info("Latency: {:.2f} s".format(latency_wo_comm * 1000))
    logger.info("Token latency: {:.2f} ms".format(latency * 1000))
    logger.info("Acceptance rate: {:.2f}".format(acceptance_rate))
    logger.info("Block efficiency: {:.2f}".format(block_efficiency))
    logger.info("Token tput: {:.2f} tokens/sec".format(tput))

    data = {
        "ul_load": ul_load_fin,
        "dl_load": dl_load_fin,
        "out_length": output_length,
        "latency_draft": latency_draft,
        "latency_verification": latency_verification,
        "latency": latency_wo_comm * 1000,
        "tput": tput,
        "acc. rate": acceptance_rate,
        "invocation cnt": invocation_count,
        "acceptance cnt": acceptance_count,
    }

    if target_model_temp < 1:
        target_model_temp *= 10

    save_path = '_'.join([verification_method, 'temp', str(int(target_model_temp)), 'len', str(n_config), 'iter', str(num_iter), 'loads.json'])
    save_out_path = '_'.join([verification_method, 'temp', str(int(target_model_temp)), 'len', str(n_config), 'iter', str(num_iter), 'output.json'])
    
    if verification_method == 'uhlm':
        save_path = '_'.join([verification_method, 'temp', str(int(target_model_temp)), 'len', str(n_config), 'th', str(u_th), 'iter', str(num_iter), 'loads.json'])
        save_out_path = '_'.join([verification_method, 'temp', str(int(target_model_temp)), 'len', str(n_config), 'th', str(u_th), 'iter', str(num_iter), 'output.json'])
    dataset_name = os.path.splitext(os.path.basename(datapath))[0]
    
    if draft_model_str == 'JackFram/llama-68m':
        logs_dir = f'logs/{target_model_str}/{dataset_name}/load'
        logs_dir_out = f'logs/{target_model_str}/{dataset_name}/output'
    else:
        logs_dir = f'logs/{draft_model_str}/{target_model_str}/{dataset_name}/load'
        logs_dir_out = f'logs/{draft_model_str}/{target_model_str}/{dataset_name}/output'
        
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(logs_dir_out, exist_ok=True)

    loads_file = f'{logs_dir}/{save_path}'
    output_file = f'{logs_dir_out}/{save_out_path}'

    with open(loads_file, "w") as f:
        json.dump(data, f, indent=4)

    with open(output_file, "w") as f:
        json.dump(output_seq, f, indent=4)


def run_baseline_eval(
        target_model,
        target_model_str,
        tokenizer,
        datapath: str,
        max_new_tokens: int = 128,
        sampling_type="sampling",
        disable_tqdm: bool = False,
        num_iter: int = 1,
):
    target_model_temp = 0.3

    dataloader = JsonData(datapath)
    generator = BaseGenerator(
        target_model,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
        temp=target_model_temp,
    )

    target_model.eval()

    logger.info("evaluation start.")
    start_time = time.time()

    invocation_count = 0
    iterator = range(len(dataloader))
    init_length_mat = []
    output_length_mat = []
    output_seq = []

    vocab_size = len(tokenizer.get_vocab())
    token_size = math.ceil(math.log2(vocab_size))

    with torch.no_grad():
        for sample_idx in iterator if disable_tqdm else tqdm(iterator):
            prompt_text = dataloader[sample_idx]
            prompt_size = len(prompt_text.encode('utf-8')) * 8 # 1 Byte/char
            inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")
            input_ids = inputs.input_ids # (bs,sequence length)
            init_length = input_ids.size(1)

            uplink_load = prompt_size

            output = generator.generate(input_ids)
            generated_text = tokenizer.decode(output.sequences[0], skip_special_tokens=True)

            downlink_load = token_size * output.sequences.size(1)

            invocation_count += output.invocation_count

            init_length_mat.append(init_length)
            output_length_mat.append(output.sequences.size(1) - init_length)
            output_seq.append({"prompt": prompt_text, "output": generated_text})

    end_time = time.time()

    logger.info("evaluation complete.")

    if target_model_temp < 1:
        target_model_temp *= 10

    save_model_name = 'draft' if target_model_str[-1] == 'm' else 'target'

    if save_model_name == 'target' and target_model_str[0] == 'Q': # Qwen
        save_model_name = 'draft'

    save_path = '_'.join([save_model_name, 'temp', str(int(target_model_temp)), 'iter', str(num_iter), 'loads.json'])
    save_out_path = '_'.join([save_model_name, 'temp', str(int(target_model_temp)), 'iter', str(num_iter), 'output.json'])

    dataset_name = os.path.splitext(os.path.basename(datapath))[0]
    logs_dir = f'logs/{target_model_str}/{dataset_name}/load'
    logs_dir_out = f'logs/{target_model_str}/{dataset_name}/output'
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(logs_dir_out, exist_ok=True)

    loads_file = f'{logs_dir}/{save_path}'
    output_file = f'{logs_dir_out}/{save_out_path}'

    run_time = end_time - start_time

    latency = run_time / invocation_count

    tput = invocation_count / run_time

    logger.info("Running time: {:.2f} s".format(run_time))
    logger.info("Token latency: {:.2f} ms".format(latency * 1000))
    logger.info("Token tput: {:.2f} tokens/sec".format(tput))

    data = {
        "init_length": init_length_mat,
        "ul_load": uplink_load,
        "dl_load": downlink_load,
        "output_length": output_length_mat,
        "latency": latency * 1000,
        "tput": tput,
    }

    with open(loads_file, "w") as f:
        json.dump(data, f, indent=4)

    with open(output_file, "w") as f:
        json.dump(output_seq, f, indent=4)


def main(args):
    torch_dtype = torch.float16 if args.fp16 else torch.float32

    logger.info("The full evaluation configuration:\n" + repr(args))

    ModelLoader = AutoModelForCausalLM
    TokenizerLoader = AutoTokenizer

    logger.info("Loading draft model: {}".format(args.draft_model))
    if not args.run_baseline:
        draft_model = ModelLoader.from_pretrained(
            args.draft_model,
            torch_dtype=torch_dtype,
            device_map=0,
        )

    logger.info("Loading target model: {}".format(args.target_model))
    target_model = ModelLoader.from_pretrained(
        args.target_model,
        torch_dtype=torch_dtype,
        device_map="auto",
    )
    tokenizer = TokenizerLoader.from_pretrained(args.tokenizer)
    tokenizer_draft = TokenizerLoader.from_pretrained(args.draft_model)

    if args.run_baseline:
        run_baseline_eval(
            target_model,
            target_model_str=args.target_model,
            tokenizer=tokenizer,
            datapath=args.datapath,
            max_new_tokens=args.max_new_tokens,
            sampling_type="sampling",
            disable_tqdm=args.disable_tqdm,
            num_iter=args.num_iter,
        )
    else:
        run_eval(
            draft_model,
            target_model,
            args.draft_model,
            args.target_model,
            fp16=args.fp16,
            tokenizer=tokenizer,
            tokenizer_draft=tokenizer_draft,
            n_config=args.n_config,
            datapath=args.datapath,
            max_new_tokens=args.max_new_tokens,
            speculative_sampling=True,
            sampling_type="sampling",
            disable_tqdm=args.disable_tqdm,
            num_iter=args.num_iter,
            verification_method=args.verification_method,
            K=args.K,
            M=args.M,
            u_s=args.u_s,
            u_max=args.u_max,
            u_th=args.u_th,
        )


if __name__ == "__main__":
    args = args_parser()

    if args.tokenizer is None:
        args.tokenizer = args.target_model

    '''For iterations!'''
    for ii in range(args.tot_iter):
        args.num_iter = ii
        main(args)
        gc.collect()
        torch.cuda.empty_cache()
