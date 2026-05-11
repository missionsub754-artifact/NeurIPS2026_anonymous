import warnings
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, List, Literal, Optional, Tuple, Union, Dict, List
import copy
import math
import torch
import time
from transformers import PreTrainedTokenizerBase
from transformers.modeling_outputs import ModelOutput
from transformers.cache_utils import Cache
from tqdm import tqdm

@dataclass
class DecoderOnlyDraftOutput(ModelOutput):
    sequences: torch.LongTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    cand_probs: Optional[Tuple[torch.FloatTensor]] = None
    ul_load: float = None
    u_values: Optional[Tuple[float]] = None


@dataclass
class DecoderOnlyVerificationOutput(ModelOutput):
    sequences: torch.LongTensor = None
    sequences_draft: torch.LongTensor = None
    target_model_past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    draft_model_past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    acceptance_count: Optional[int] = None
    dl_load: float = None
    common_indicator: bool = None
    srdv_ok: bool = None
    pass_indicator: bool = None

def _detect_family(tok: PreTrainedTokenizerBase) -> str:
    try:
        sample_size = min(2000, tok.vocab_size)
        pieces = tok.convert_ids_to_tokens(list(range(sample_size)))
        if sum(1 for p in pieces if p and p.startswith("▁")) > 10:
            return "sentencepiece"
    except Exception:
        pass
    if hasattr(tok, "sp_model"):
        return "sentencepiece"
    cls = type(tok).__name__
    if any(x in cls for x in ("Llama", "SentencePiece", "Albert", "T5", "XLNet")):
        return "sentencepiece"
    return "bpe"
 
def _is_special_like(piece: str) -> bool:
    if piece.startswith("<|") and piece.endswith("|>"):
        return True
    if piece.startswith("[") and piece.endswith("]"):
        return True
    if (
        piece.startswith("<")
        and piece.endswith(">")
        and not re.fullmatch(r"<0x[0-9A-Fa-f]{2}>", piece)
    ):
        return True
    return False
 
 
def _normalize(piece: str, family: str) -> Optional[str]:
    if _is_special_like(piece):
        return None
    if family == "sentencepiece":
        if re.fullmatch(r"<0x[0-9A-Fa-f]{2}>", piece):
            return None
        if piece.startswith("▁"):
            piece = " " + piece[1:]
    else:  # bpe
        if piece.startswith("Ġ"):
            piece = " " + piece[1:]
        elif piece == "Ċ":
            piece = "\n"
    if any(ord(c) < 32 and c not in ("\n", "\t", "\r") for c in piece):
        return None
    return piece
 
def _build_surface_map(
    tok: PreTrainedTokenizerBase,
    family: str,
    special_strings: set,
) -> Dict[str, List[int]]:
    vocab_size = len(tok.get_vocab())
    special_ids = set(tok.all_special_ids)
    pieces = tok.convert_ids_to_tokens(list(range(vocab_size)))
 
    surface_map: Dict[str, List[int]] = defaultdict(list)
    for tid, piece in enumerate(pieces):
        if piece is None:
            continue
        if tid in special_ids:
            continue
        if piece in special_strings:
            continue
 
        surface = _normalize(piece, family)
        if surface is None:
            continue
 
        # Round-trip: vocab lookup (context-free, safe for all families)
        recovered = tok.convert_tokens_to_ids(piece)
        if not (isinstance(recovered, int) and recovered == tid):
            continue
 
        surface_map[surface].append(tid)
 
    return dict(surface_map)
 
def _get_special_strings(tok: PreTrainedTokenizerBase) -> set:
    special = set()
    for s in getattr(tok, "all_special_tokens", []):
        if s is not None:
            special.add(s)
    for s in getattr(tok, "all_special_tokens_extended", []):
        content = getattr(s, "content", None)
        if content is not None:
            special.add(content)
        elif isinstance(s, str):
            special.add(s)
    return special
 
def find_common_vocab(
    tokenizer_draft: PreTrainedTokenizerBase,
    tokenizer: PreTrainedTokenizerBase,
    include_special_tokens: bool = True,
) -> Optional[Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]]:
    """Find 1:1 token mappings between draft and target tokenizers."""
    family_draft = _detect_family(tokenizer_draft)
    family_target = _detect_family(tokenizer)
 
    if include_special_tokens:
        special_draft = set()
        special_target = set()
    else:
        special_draft = _get_special_strings(tokenizer_draft)
        special_target = _get_special_strings(tokenizer)
 
    draft_s2ids = _build_surface_map(tokenizer_draft, family_draft, special_draft)
    target_s2ids = _build_surface_map(tokenizer, family_target, special_target)
 
    common_surfaces = set(draft_s2ids) & set(target_s2ids)
 
    pairs: List[Tuple[int, int]] = []
    for surface in common_surfaces:
        d_ids = draft_s2ids[surface]
        t_ids = target_s2ids[surface]
        # strict 1:1 only
        if len(d_ids) == 1 and len(t_ids) == 1:
            pairs.append((d_ids[0], t_ids[0]))
 
    pairs.sort(key=lambda x: x[1])
 
    common_draft_idx = torch.tensor([d for d, _ in pairs], dtype=torch.long)
    common_target_idx = torch.tensor([t for _, t in pairs], dtype=torch.long)
 
    draft_vocab_size = max(tokenizer_draft.get_vocab().values()) + 1
    draft2target = torch.full((draft_vocab_size,), -1, dtype=torch.long)
    if len(common_draft_idx) > 0:
        draft2target[common_draft_idx] = common_target_idx
 
    return common_draft_idx, common_target_idx, draft2target

def _crop_past_kv(past_kv, max_length):
    if past_kv is None:
        return None
    
    # Cache object
    if isinstance(past_kv, Cache):
        past_kv.crop(max_length)
        return past_kv
    
    # Legacy (Tuple type KV)
    if isinstance(past_kv, (tuple,list)):
        new_past = []
        for layer in past_kv:
            k, v = layer
            new_past.append((k[:,:, :max_length,:], v[:,:, :max_length, :]))
        return type(past_kv)(new_past)
    
    raise TypeError(f"Unsupported KV type: {type(past_kv)}")

# HR resampling
def _SDX(
        ground_probs: torch.FloatTensor,
        cand_probs: torch.FloatTensor,
        cand_token: torch.LongTensor,
        common_token_idx: torch.LongTensor,
        oov_token_idx: torch.LongTensor,
):
    accept_threshold = ground_probs[cand_token] / cand_probs[cand_token]
    if torch.rand(1, device=accept_threshold.device) <= accept_threshold:
        # Accept
        accept = True
        common_indicator = False
        return accept, common_indicator
    else:
        # Reject
        theta = torch.nn.functional.relu(ground_probs - cand_probs)
        theta_1 = torch.sum(theta[common_token_idx])
        theta_2 = torch.sum(ground_probs[oov_token_idx])
        
        decision_threshold = theta_1 / (theta_1 + theta_2)
        if torch.rand(1, device=decision_threshold.device) <= decision_threshold:
            # Device-side resampling
            ground_probs[oov_token_idx] = 0
            ground_probs -= cand_probs
            ground_probs = torch.nn.functional.relu(ground_probs, inplace=True)
            ground_probs /= theta_1
            common_indicator = True # Indicator for resampling in common set
        else:
            # Server-side resampling
            ground_probs[common_token_idx] = 0
            ground_probs /= theta_2
            common_indicator = False
        accept = False
        return accept, common_indicator

# Naive resampling
def _SDHet(
        ground_probs: torch.FloatTensor,
        cand_probs: torch.FloatTensor,
        cand_token: torch.LongTensor,
        common_token_idx: torch.LongTensor,
        oov_token_idx: Optional[torch.LongTensor] = None,
) -> bool:
    cand_probs = cand_probs.to(ground_probs.device)
    
    accept_threshold = ground_probs[cand_token] / cand_probs[cand_token]
    if torch.rand(1, device=accept_threshold.device) <= accept_threshold:
        # Accept
        return True
    else:
        # Reject
        ground_probs -= cand_probs
        ground_probs = torch.nn.functional.relu(ground_probs, inplace=True)
        ground_probs /= ground_probs.sum()
        return False

### Strategy classes
class Strategy:
    def __init__(
            self,
            draft_model,
            target_model,
            tokenizer,
            tokenizer_draft,
            verification_method,
            n_config: int,
            draft_model_temp: float = 1,
            target_model_temp: float = 1,
            speculative_sampling: bool = True,
    ) -> None:

        self.n_config = n_config
        self.draft_model = draft_model
        self.target_model = target_model
        self.draft_model_device = (
            draft_model.model.get_input_embeddings().weight.device
        )
        self.target_model_device = (
            target_model.model.get_input_embeddings().weight.device
        )
        self.draft_model_temp = draft_model_temp
        self.target_model_temp = target_model_temp
        self.speculative_sampling = speculative_sampling
        self.verification_method = verification_method

        self.acceptance_check: Callable[
            [torch.FloatTensor, torch.FloatTensor, torch.LongTensor, 
             torch.LongTensor, torch.LongTensor, Optional[torch.LongTensor]],
            bool,
        ] = None
        
        vocab_size_draft = len(tokenizer_draft.get_vocab())
        vocab_size = len(tokenizer.get_vocab())

        self.token_size_draft = math.ceil(math.log2(vocab_size_draft))
        self.token_size = math.ceil(math.log2(vocab_size))

        self.common_draft_idx, self.common_target_idx, self.draft2target = find_common_vocab(
        tokenizer_draft=tokenizer_draft,
        tokenizer=tokenizer
        )
        
        self.common_draft_idx = self.common_draft_idx.to(draft_model.device)
        self.common_target_idx = self.common_target_idx.to(target_model.device)
        self.draft2target = self.draft2target.to(target_model.device)
        
        self.target2draft: dict = {
            int(t): int(d)
            for d, t in zip(self.common_draft_idx.tolist(), self.common_target_idx.tolist())
        }

        print(f"draft vocab length:{vocab_size_draft}")
        print(f"target vocab length:{vocab_size}")
        print(f"common vocab length:{self.common_target_idx.size()}")

        common_token_set = set(self.common_target_idx.tolist())
        self.oov_target_idx = torch.tensor([i for i in range(len(tokenizer.get_vocab())) if i not in common_token_set])
        self.oov_target_idx = self.oov_target_idx.to(target_model.device)

        common_token_set = set(self.common_draft_idx.tolist())
        self.oov_draft_idx = torch.tensor([i for i in range(len(tokenizer_draft.get_vocab())) if i not in common_token_set])
        self.oov_draft_idx = self.oov_draft_idx.to(draft_model.device)

        print("common:", len(self.common_target_idx), "unique:", len(set(self.common_target_idx.tolist())))

        if speculative_sampling:
            print(f'Verification mode: {self.verification_method}')
            if self.verification_method in ('ul', 'dl', 'uhlm'):
                self.acceptance_check = _SDHet
            else: # 'hr', 'rs', 'gr', 'tr' -> _SDX
                self.acceptance_check = _SDX

    def generate_draft(
            self,
            input_ids: torch.LongTensor,
            past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]],
    ) -> DecoderOnlyDraftOutput:
        raise NotImplementedError

    def acceptance_check(self, 
                        ground_probs: torch.FloatTensor,
                        cand_probs: torch.FloatTensor,
                        cand_token: torch.LongTensor,
                        common_token_idx: torch.LongTensor,
                        common_token_idx_draft: torch.LongTensor,
                        oov_token_idx: Optional[torch.LongTensor]) -> Optional[int]:
        raise NotImplementedError

    def verify(
            self,
            input_ids: torch.LongTensor,
            input_ids_draft: torch.LongTensor,
            target_model_past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]],
            draft_model_past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]],
            cand_probs: Optional[Tuple[torch.FloatTensor]],
            u_values=None,
    ) -> DecoderOnlyVerificationOutput:
        raise NotImplementedError

class BaseStrategy(Strategy):
    def __init__(
            self,
            draft_model,
            target_model,
            tokenizer,
            tokenizer_draft,
            verification_method,
            fp16,
            n_config: int,
            draft_model_temp: float = 1,
            target_model_temp: float = 1,
            speculative_sampling: bool = True,
            K: int = 20,
            M: int = 10,
            u_s: int = 20,
            u_max: float = 2.0,
            u_th: float = 0.8,
    ) -> None:
        super().__init__(
            draft_model,
            target_model,
            tokenizer,
            tokenizer_draft,
            verification_method,
            n_config,
            draft_model_temp,
            target_model_temp,
            speculative_sampling,
        )
        self.fp16 = fp16
        self.verification_method = verification_method
        
        # SRDV
        self.K = K
        self.M = M

        # UHLM
        self.u_s = u_s
        self.u_max = u_max
        self.u_th = u_th
    
    # SRDV resampler
    def _srdv_resample(self, orig_ground_prob, mapped_cand_probs):
        for iter in range(self.M):
            candidates = torch.multinomial(orig_ground_prob, num_samples=self.K,
                                           replacement=True)
            for cand in candidates:
                cand_item = cand.item()
                q = orig_ground_prob[cand_item]
                p = mapped_cand_probs[cand_item]
                
                acc_prob = torch.clamp((q - p) / q, min=0.0, max=1.0)
                if torch.rand(1, device=acc_prob.device) <= acc_prob:
                    return True, cand_item, iter
        return False, None, iter

    # uhlm: uncertainty measure
    def _compute_uhlm_uncertainty(self, step_cand_probs, cand_token_item):
        if self.draft_model_temp == 0:
            return 0.0

        probs_1d = step_cand_probs[0]                     
        log_p = torch.log(probs_1d.clamp(min=1e-40))

        # Sample s temperatures uniformly from (0, theta_max]
        temps = torch.rand(self.u_s, device=probs_1d.device) * self.u_max
        temps = temps.clamp(min=1e-6)  # avoid division by zero

        mismatch = 0
        for i in range(self.u_s):
            tau = temps[i].item()
            ratio = self.draft_model_temp / tau         
            perturbed_logits = log_p * ratio             
            perturbed_probs = torch.softmax(perturbed_logits, dim=-1)

            perturbed_probs[self.oov_draft_idx] = 0.0
            p_sum = perturbed_probs.sum()
            perturbed_probs = perturbed_probs / p_sum

            perturbed_tok = torch.multinomial(perturbed_probs, num_samples=1).item()
            if perturbed_tok != cand_token_item:
                mismatch += 1

        return mismatch / self.u_s
        
    def generate_draft(
            self,
            input_ids: torch.LongTensor,
            past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]],
    ) -> DecoderOnlyDraftOutput:
        input_ids = input_ids.to(self.draft_model_device)
        cand_probs = []
        init_input_length = input_ids.size(1)
        u_values = [] # For UHLM

        past_key_values = past_key_values
        if past_key_values is not None:
            pruned_input_ids = input_ids[:, past_key_values.get_seq_length():]  # Non computed input tokens (1,T_new)
        else:  
            pruned_input_ids = input_ids
        for step in range(self.n_config):  # max_draft_len: n_config
            outputs: BaseModelOutputWithPast = self.draft_model.model(
                input_ids=pruned_input_ids,
                use_cache=True,
                past_key_values=past_key_values,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False,
            )  # Forward pass of the draft model

            hidden_states = outputs.last_hidden_state
            logits = self.draft_model.lm_head(hidden_states[0, -1:])

            past_key_values = outputs.past_key_values

            step_cand_probs = torch.softmax(logits.float() / self.draft_model_temp, dim=-1)
            step_cand_probs[:, self.oov_draft_idx] = 0
            step_cand_probs /= torch.sum(step_cand_probs, dim=1)
            if torch.isnan(step_cand_probs.sum(dim=1)):
                print(f"sum:{step_cand_probs.sum(dim=1)}")
                print(f"sum of common:{step_cand_probs[:, self.common_draft_idx].sum(dim=1)}")
                print(f"oov idx:{self.oov_draft_idx}")
                print(f"oov idx (len): {self.oov_draft_idx.size(0)}")
                print(f"common idx (len): {self.common_draft_idx.size(0)}")

            cand_tokens = torch.multinomial(
                step_cand_probs, num_samples=1
            ).view(1, -1) 

            cand_probs.append(step_cand_probs) # Distributions

            # uhlm: uncertainty calculation
            if self.verification_method == 'uhlm':
                if self.n_config > 1:
                    raise ValueError("uhlm verification only supports n_config = 1")
                
                u = self._compute_uhlm_uncertainty(step_cand_probs, cand_tokens[0, 0].item())
                u_values.append(u)

            pruned_input_ids = cand_tokens 

            input_ids = torch.cat((input_ids, pruned_input_ids), dim=1)

        prob_size = 16 if self.fp16 else 32
        uplink_token_size = pruned_input_ids.size(1) * self.token_size_draft  # Token loads (All token's indices)
        uplink_prob_size = pruned_input_ids.size(1) * prob_size

        uplink_load = uplink_token_size + uplink_prob_size

        if self.verification_method == 'ul':
            uplink_load = pruned_input_ids.size(1) * (self.common_draft_idx.size(0) * prob_size + self.token_size_draft)
        
        if self.verification_method == 'uhlm':
            uplink_load = 0

        return DecoderOnlyDraftOutput(
            sequences=input_ids,
            past_key_values=past_key_values,
            cand_probs=tuple(cand_probs),
            ul_load=uplink_load,
            u_values=u_values,
        )

    def _forward_target_model(
            self,
            input_ids: torch.LongTensor,
            past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]],
    ):
        input_ids = input_ids.to(self.target_model_device)

        past_key_values = past_key_values
        
        if past_key_values is not None:
            pruned_input_ids = input_ids[:, past_key_values.get_seq_length():]
        else:
            pruned_input_ids = input_ids

        outputs: BaseModelOutputWithPast = self.target_model.model(
            input_ids=pruned_input_ids,
            use_cache=True,
            past_key_values=past_key_values,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )

        hidden_states = outputs.last_hidden_state
        past_key_values = outputs.past_key_values
        logits = self.target_model.lm_head(hidden_states[:,-self.n_config-1:])
        return logits, past_key_values

    def _forward_target_model_uhlm(
            self,
            input_ids_without_cand: torch.LongTensor,
            past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]],
    ):
        input_ids_without_cand = input_ids_without_cand.to(self.target_model_device)
 
        if past_key_values is not None:
            pruned_input_ids = input_ids_without_cand[:, past_key_values.get_seq_length():]
        else:
            pruned_input_ids = input_ids_without_cand
 
        outputs = self.target_model.model(
            input_ids=pruned_input_ids,
            use_cache=True,
            past_key_values=past_key_values,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )
 
        hidden_states = outputs.last_hidden_state  
        past_key_values = outputs.past_key_values
 
        verify_logit = self.target_model.lm_head(hidden_states[:, -1:])  
        verify_logit = verify_logit[0, 0]  
 
        ground_prob = torch.softmax(verify_logit.float() / self.target_model_temp, dim=-1)
 
        return ground_prob, past_key_values

    def verify(
            self,
            input_ids: torch.LongTensor,
            input_ids_draft: torch.LongTensor,
            target_model_past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]],
            draft_model_past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]],
            cand_probs: Optional[Tuple[torch.FloatTensor]],
            u_values=None,
    ) -> DecoderOnlyVerificationOutput:
        # UHLM
        if self.verification_method == 'uhlm':
            return self._verify_uhlm(
            input_ids=input_ids,
            input_ids_draft=input_ids_draft,
            target_model_past_key_values=target_model_past_key_values,
            draft_model_past_key_values=draft_model_past_key_values,
            cand_probs=cand_probs,
            u_values=u_values,
            )


        cand_draft = input_ids_draft[:, -self.n_config:].to(device=input_ids.device) # Candidate tokens (draft vocabulary ids)
        init_input_length_draft = input_ids_draft.size(1) - self.n_config

        input_ids_cand_tokens = self.draft2target[cand_draft]
        input_ids = torch.cat((input_ids, input_ids_cand_tokens), dim=1)

        input_ids = input_ids.to(self.target_model_device)
        torch.cuda.synchronize()

        logits, target_model_past_key_values = self._forward_target_model(
            input_ids, target_model_past_key_values
        )

        logits = logits[0] 
        unverified_tokens = input_ids[0, -self.n_config:]  
        init_input_length = input_ids.size(1) - self.n_config

        ground_probs = torch.softmax(logits.float() / self.target_model_temp, dim=-1) 
        current_ground_prob = ground_probs[0] # V,
        if self.verification_method == 'gr' or self.verification_method == 'tr':
            org_ground_prob = ground_probs.clone()

        ground_probs = ground_probs[1:] 

        keep_indices = list(range(init_input_length)) 
        keep_indices_draft = list(range(init_input_length_draft)) 
        to_drop_len = 0
        common_indicator = None
        
        srdv_accepted_token = None  # 'srdv' path
        srdv_ok = None              # 'srdv' path
        uhlm_tail_token = None      # 'uhlm' path
        pass_indicator = None       # 'uhlm' path

        for depth in range(self.n_config): # max_draft_len = len(n_config)
            mapped_cand_probs = torch.zeros_like(current_ground_prob)
            mapped_cand_probs[self.common_target_idx] = cand_probs[depth][0][self.common_draft_idx]

            # ==============================================================
            # SRDV
            # ==============================================================
            if self.verification_method == 'srdv':
                orig_ground_prob_srdv = current_ground_prob.clone()

                accept, common_indicator = self.acceptance_check(
                    current_ground_prob.to(torch.float32),
                    mapped_cand_probs.to(torch.float32),
                    unverified_tokens[depth],
                    self.common_target_idx,
                    self.oov_target_idx,
                )

                if accept:
                    current_ground_prob = ground_probs[depth]
                    keep_indices.append(init_input_length + depth)
                    keep_indices_draft.append(init_input_length_draft + depth)
                    if depth == self.n_config - 1:
                        to_drop_len += 1
                        depth = self.n_config
                else:
                    srdv_ok, srdv_tok, rounds = self._srdv_resample(
                        orig_ground_prob_srdv.to(torch.float32), mapped_cand_probs.to(torch.float32)
                        )
                    if srdv_ok:
                        srdv_accepted_token = torch.tensor(
                            [srdv_tok], dtype=torch.long, device=input_ids.device
                        )
                    break
            
            # hr, gr, tr, ul/dl
            else:
                if self.acceptance_check is _SDX:
                    accept, common_indicator = self.acceptance_check(
                        current_ground_prob.to(torch.float32), 
                        mapped_cand_probs.to(torch.float32),  
                        unverified_tokens[depth], 
                        self.common_target_idx, # Token index (Common)
                        self.oov_target_idx, # Token index (OOV)
                    )
                else:
                    accept = self.acceptance_check(
                        current_ground_prob.to(torch.float32), 
                        mapped_cand_probs.to(torch.float32),  
                        unverified_tokens[depth], 
                        self.common_target_idx, # Token index (Common)
                        self.oov_target_idx, # Token index (OOV)
                    )

                if accept is True: # There is an accepted candidate
                    current_ground_prob = ground_probs[depth] # Next token's ground distribution, (V,)
                    keep_indices.append(init_input_length + depth) # Keep indices: init input IDs + accepted token position index
                    keep_indices_draft.append(init_input_length_draft + depth) # Keep indices: init input IDs + accepted token position index
                    if depth == self.n_config - 1: # All candidates are accepted
                        to_drop_len += 1
                        depth = self.n_config
                else: # Rejection occurs
                    break
        
        keep_indices = torch.tensor(
            keep_indices, dtype=torch.long, device=self.target_model_device
        )
        keep_indices_draft = torch.tensor(
            keep_indices_draft, dtype=torch.long, device=self.draft_model_device
        )

        if to_drop_len != 0: # All candidates are accepted
            keep_indices_draft_kv = keep_indices_draft[: len(keep_indices_draft) - to_drop_len]
        else:
            keep_indices_draft_kv = keep_indices_draft

        if (current_ground_prob < 0).any():
            print(f"Sum: {current_ground_prob.sum().item()}")
            print("Error: current_ground_prob contains negative values!")

        # Tail token selection
        if srdv_accepted_token is not None:
            tail_ground_token = srdv_accepted_token

        elif self.verification_method in ('gr', 'tr'):
            if to_drop_len != 0: # All candidates are accepted
                tail_ground_token = torch.multinomial(current_ground_prob, num_samples=1).to(
                device=input_ids.device)
            else:
                current_ground_prob = org_ground_prob[depth]
                if self.verification_method == 'gr':  # Resampling from target - top-1 sampling (GR)
                    tail_ground_token = torch.topk(current_ground_prob,k=1)[1].to(device=input_ids.device)
                else:  # Resampling from the target distribution 
                    tail_ground_token = torch.multinomial(current_ground_prob,num_samples=1).to(device=input_ids.device) 
        else:
            tail_ground_token = torch.multinomial(current_ground_prob, num_samples=1).to(
                device=input_ids.device
            )

        input_ids = input_ids.index_select(dim=1, index=keep_indices)
        input_ids = torch.cat((input_ids, tail_ground_token[None]), dim=1)
        input_ids_draft = input_ids_draft.index_select(dim=1, index=keep_indices_draft)

        # Target KV cache update
        target_keep_len = int(keep_indices.numel())
        target_model_past_key_values = _crop_past_kv(target_model_past_key_values, target_keep_len)
        # Draft KV cache update
        draft_keep_len = int(keep_indices_draft_kv.numel())
        draft_model_past_key_values = _crop_past_kv(draft_model_past_key_values, draft_keep_len)

        prob_size = 16 if self.fp16 else 32

        if self.verification_method == 'hr':  #  Comm-load: Proposed (X-CoSD)
            downlink_load = self.token_size if depth == self.n_config else prob_size * len(self.common_target_idx)
            if depth != self.n_config and not common_indicator: # OOV resampling
                downlink_load +=  self.token_size
        elif self.verification_method in ('ul', 'dl'):  #  Comm-load: Baseline (UL, DL)
            if self.verification_method == 'dl' and depth != self.n_config:
                downlink_load = prob_size * (len(self.common_target_idx) + len(self.oov_target_idx))
            else:
                downlink_load = self.token_size
        elif self.verification_method in ('gr', 'tr'):
            downlink_load = self.token_size
        else: # 'srdv'
            if depth == self.n_config: # All accepted
                downlink_load = self.token_size
            else:
                if srdv_ok:
                    downlink_load = (rounds + 1) * self.K * (self.token_size + prob_size)
                else:
                    downlink_load = self.M * self.K * self.token_size + prob_size * len(self.common_target_idx)
                    if not common_indicator:
                        downlink_load += self.token_size

        return DecoderOnlyVerificationOutput(
            sequences=input_ids,
            sequences_draft=input_ids_draft,
            target_model_past_key_values=target_model_past_key_values,
            draft_model_past_key_values=draft_model_past_key_values,
            acceptance_count=depth,
            dl_load=downlink_load,
            common_indicator=common_indicator,
            srdv_ok=srdv_ok,
            pass_indicator=pass_indicator,
        )
    
    # UHLM Verification
    def _verify_uhlm(
            self,
            input_ids: torch.LongTensor,
            input_ids_draft: torch.LongTensor,
            target_model_past_key_values,
            draft_model_past_key_values,
            cand_probs,
            u_values,
    ) -> DecoderOnlyVerificationOutput:
        assert u_values is not None and len(u_values) == 1, \
            "UHLM requires n_config=1 and u_values of length 1"
 
        u = u_values[0]
 
        init_input_length_draft = input_ids_draft.size(1) - self.n_config  # length before candidate
        init_input_length = input_ids.size(1)                              # length before candidate (target)
 
        cand_draft = input_ids_draft[:, -self.n_config:].to(device=input_ids.device)  
        input_ids_cand_tokens = self.draft2target[cand_draft]              
        input_ids_with_cand = torch.cat((input_ids, input_ids_cand_tokens), dim=1)
 
        if u < self.u_th:
            input_ids_out       = input_ids_with_cand                    
            input_ids_draft_out = input_ids_draft                         
 
            return DecoderOnlyVerificationOutput(
                sequences=input_ids_out,
                sequences_draft=input_ids_draft_out,
                target_model_past_key_values=target_model_past_key_values,  
                draft_model_past_key_values=draft_model_past_key_values,   
                acceptance_count=0,
                dl_load=0,
                common_indicator=None,
                srdv_ok=None,
                pass_indicator=True,
            )
 
        current_ground_prob, target_model_past_key_values = self._forward_target_model_uhlm(
            input_ids, target_model_past_key_values   
        )
        mapped_cand_probs = torch.zeros_like(current_ground_prob)
        mapped_cand_probs[self.common_target_idx] = cand_probs[0][0][self.common_draft_idx]
 
        unverified_token = input_ids_with_cand[0, init_input_length]  # candidate id (target vocab)
 
        accept = self.acceptance_check(
            current_ground_prob.to(torch.float32),
            mapped_cand_probs.to(torch.float32),
            unverified_token,
            self.common_target_idx,
            self.oov_target_idx,
        )
 
        if accept:
            input_ids_out       = input_ids_with_cand
            input_ids_draft_out = input_ids_draft
 
            return DecoderOnlyVerificationOutput(
                sequences=input_ids_out,
                sequences_draft=input_ids_draft_out,
                target_model_past_key_values=target_model_past_key_values, 
                draft_model_past_key_values=draft_model_past_key_values,
                acceptance_count=0,
                dl_load=self.token_size,
                common_indicator=None,
                srdv_ok=None,
                pass_indicator=False,  
            )
 
        else:
            residual = current_ground_prob
            tail_ground_token = torch.multinomial(residual, num_samples=1).to(
                device=input_ids.device
            )  

            input_ids_out = torch.cat((input_ids, tail_ground_token[None]), dim=1)
 
            input_ids_draft_out = input_ids_draft[:, :init_input_length_draft]
 
            draft_keep_len = init_input_length_draft
            draft_model_past_key_values = _crop_past_kv(
                draft_model_past_key_values, draft_keep_len
            )
 
            return DecoderOnlyVerificationOutput(
                sequences=input_ids_out,
                sequences_draft=input_ids_draft_out,
                target_model_past_key_values=target_model_past_key_values, 
                draft_model_past_key_values=draft_model_past_key_values,
                acceptance_count=0,
                dl_load=self.token_size,
                common_indicator=None,
                srdv_ok=None,
                pass_indicator=False, 
            )