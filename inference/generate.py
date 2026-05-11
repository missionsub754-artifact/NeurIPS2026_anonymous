from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import time
import torch
import math
from torch.onnx import verification
from transformers.modeling_outputs import ModelOutput, CausalLMOutputWithPast

from . import strategies


@dataclass
class DecoderOnlyOutput(ModelOutput):
    sequences: torch.LongTensor
    acceptance_count: int = None
    draft_token_count: int = None
    invocation_count: int = None
    draft_time: float = None
    verification_time: float = None
    sample_time: float = None
    ul_load: list = None
    dl_load: list = None


class BaseGenerator:
    def __init__(
        self,
        model,
        eos_token_id: int,
        max_new_tokens: int = 128,
        temp: float = 1,
    ) -> None:
        self.model = model
        self.eos_token_id = eos_token_id
        self.max_new_tokens = max_new_tokens
        self.temp = temp

    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
    ) -> DecoderOnlyOutput:
        past_key_values = None
        invocation_count = 0

        init_input_len = input_ids.size(-1) # shape of input IDs -> 1,length 

        while True:
            if past_key_values is not None:
                # KV -> Tuple of length l(# of layers), i.e., KV[l] = (Key, Value)
                # Key, Value -> Tuple of (bs) X (# of KV heads) X (Past Sequence length) X (Head dimension)
                pruned_input_ids = input_ids[:, past_key_values[0][0].size(2) :] 
            else:
                pruned_input_ids = input_ids

            outputs: CausalLMOutputWithPast = self.model(
                input_ids=pruned_input_ids,
                use_cache=True,
                past_key_values=past_key_values,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False,
            )

            logits = outputs.logits
            past_key_values = outputs.past_key_values

            batch_num, seq_len, _ = logits.size() # Logits: (bs) X (length) X (Vocabulary size)

            ground_probs = torch.softmax(
                logits / self.temp, dim=-1
            )  # (bs) X (Length) X (Vocabulary size)

            ground_tokens = torch.multinomial(
                ground_probs.view(batch_num * seq_len, -1), num_samples=1
            )  # ((bs)*(Length)) X 1

            ground_tokens = ground_tokens.view(batch_num, seq_len) # (bs) X (Length)

            input_ids = torch.cat(
                (input_ids, ground_tokens[:, -1:].to(input_ids)), dim=1
            )

            invocation_count += 1

            if (
                self.eos_token_id in input_ids[0, -1:]
                or input_ids.size(-1) - init_input_len >= self.max_new_tokens
            ):
                break
        return DecoderOnlyOutput(sequences=input_ids, invocation_count=invocation_count)


class SpeculativeGenerator:
    def __init__(
        self,
        draft_model,
        target_model,
        tokenizer,
        tokenizer_draft,
        verification_method,
        fp16,
        eos_token_id: int,
        n_config: int,
        max_new_tokens: int = 128,
        draft_model_temp: float = 1,
        target_model_temp: float = 1,
        speculative_sampling: bool = True,
        K: int = 20,
        M: int = 10,
        u_s: int = 20,
        u_max: float = 2.0,
        u_th: float = 0.8,
    ) -> None:
        self.eos_token_id = eos_token_id
        self.max_new_tokens = max_new_tokens
        self.strategy: strategies.Strategy = None
        self.verification_method = verification_method
        self.draft_model = draft_model
        self.target_model = target_model
        self.tokenizer = tokenizer
        self.tokenizer_draft = tokenizer_draft
        self.n_config = n_config

        vocab_size_draft = len(self.tokenizer_draft.get_vocab())
        vocab_size = len(self.tokenizer.get_vocab())

        self.token_size_draft = math.ceil(math.log2(vocab_size_draft))
        self.token_size = math.ceil(math.log2(vocab_size))
        self.prob_size = 16 if fp16 else 32

        self.strategy = strategies.BaseStrategy(
            draft_model=draft_model,
            target_model=target_model,
            tokenizer=tokenizer,
            tokenizer_draft=tokenizer_draft,
            verification_method=verification_method,
            fp16=fp16,
            n_config=n_config,
            draft_model_temp=draft_model_temp,
            target_model_temp=target_model_temp,
            speculative_sampling=speculative_sampling,
            K=K,
            M=M,
            u_s=u_s,
            u_max=u_max,
            u_th=u_th,
        )

    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        input_ids_draft: Optional[torch.Tensor] = None,
    ) -> DecoderOnlyOutput:
        target_model_past_key_values = None
        draft_model_past_key_values = None

        invocation_count = 0
        acceptance_count = 0

        init_input_len = input_ids.size(-1) # Input IDs: (BS) X (Length), Target prompt
        init_input_len_draft = input_ids_draft.size(-1)

        draft_time = 0
        verification_time = 0

        ul_load = []
        dl_load = []
        init_round = True # If not init_round -> Text (not noken) processing needed in the draft
        
        pass_count = 0

        while True:
            if not init_round:
                if not (self.verification_method == 'uhlm' and verification_output.pass_indicator):
                    last_target_token_id = verification_output.sequences[0, -1].item()
                    
                    target2draft = self.strategy.target2draft 
                    draft_token_id = target2draft.get(last_target_token_id, None)
                    if draft_token_id is not None:
                        new_draft_token = torch.tensor(
                            [[draft_token_id]], dtype=torch.long, device="cuda"
                        )
                        input_ids_draft = torch.cat((input_ids_draft, new_draft_token), dim=1)
                    else:
                        tail_ground_token_str = self.tokenizer.decode(verification_output.sequences[0, -1], clean_up_tokenization_spaces=False)
                        ground_token_draft = self.tokenizer_draft(tail_ground_token_str, add_special_tokens=False, return_tensors="pt").to("cuda")
                        input_ids_draft = torch.cat((input_ids_draft, ground_token_draft.input_ids), dim=1)

            start_draft = time.time()
            draft_output = self.strategy.generate_draft(
                input_ids_draft,
                past_key_values=draft_model_past_key_values,
            )
            end_draft = time.time()
            draft_time += end_draft - start_draft
            
            draft_model_past_key_values = draft_output.past_key_values # Tuple: (Layer) X (Key, Value)
            
            start_verification = time.time()

            verification_output = self.strategy.verify(
                input_ids=input_ids,
                input_ids_draft=draft_output.sequences,
                target_model_past_key_values=target_model_past_key_values,
                draft_model_past_key_values=draft_output.past_key_values,
                cand_probs=draft_output.cand_probs,
                u_values=draft_output.u_values,   # None for non-uhlm methods
            )

            end_verification = time.time()
            verification_time += end_verification - start_verification
            
            ul_load_temp = draft_output.ul_load

            if self.verification_method in ('hr', 'srdv') and verification_output.acceptance_count != self.n_config: # Rejection occurs
                if verification_output.srdv_ok:
                    ul_load_temp += self.token_size
                if verification_output.common_indicator:
                    ul_load_temp += self.token_size_draft

            if self.verification_method == 'uhlm':
                if verification_output.pass_indicator:      # u < u_th token (pass)
                    ul_load_temp = 0
                    pass_count += 1
                else:       
                    common_vocab_len = len(self.strategy.common_target_idx)
                    ul_load_temp = (pass_count + 1) * self.token_size_draft + common_vocab_len * self.prob_size
                    pass_count = 0

            ul_load.append(ul_load_temp)
            dl_load.append(verification_output.dl_load)

            input_ids = verification_output.sequences
            input_ids_draft = verification_output.sequences_draft

            draft_model_past_key_values = (
                verification_output.draft_model_past_key_values
            )
            target_model_past_key_values = (
                verification_output.target_model_past_key_values
            )

            invocation_count += 1
            acceptance_count += verification_output.acceptance_count

            token_stop_idx = self.n_config

            init_round = False

            if (
                self.eos_token_id in input_ids[0, -token_stop_idx:]
                or input_ids.size(-1) - init_input_len >= self.max_new_tokens
            ):
                break

        sample_time = draft_time + verification_time  

        return DecoderOnlyOutput(
            sequences=input_ids,
            acceptance_count=acceptance_count,
            draft_token_count=invocation_count * token_stop_idx,
            invocation_count=invocation_count,
            draft_time=draft_time,
            verification_time=verification_time,
            sample_time=sample_time,
            ul_load=ul_load,
            dl_load=dl_load,
        )