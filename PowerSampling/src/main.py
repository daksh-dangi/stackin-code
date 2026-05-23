import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import DynamicCache
from torch.utils._pytree import tree_map

local_path = "./Qwen3-4B"

tokenizer = AutoTokenizer.from_pretrained(local_path)
model = AutoModelForCausalLM.from_pretrained(
    local_path, 
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
THINK_END_ID = tokenizer.encode("</think>", add_special_tokens=False)[-1]
eos_token_id = tokenizer.eos_token_id

################################
# HYPERPARAMETER INITALIZATION #
################################
# all are taken directly from the paper (except L & H as no values were specified)
# will test with different H, as H is only used to estimate the likelihood scaling factors
BLOCK_SIZE, MC_BUDGET, TRAJECTORY_LENGTH = 192, 8, 2880
alpha, k  = 4, 8
L, HORIZON_LENGTH = 16, 50
temperature, top_k, top_p = 0.8, 50, 0.9
eps = 1e-10

messages = [
    [{"role": "user", "content": "How does a small change in initial conditions affect the behavior of the Lorenz attractor system, and how does this relate to chaos theory in physics? Specifically, analyze the system's behavior for a range of values of the parameters \u03c3 (Prandtl number), \u03c1 (Rayleigh number), and \u03b2 (aspect ratio)."}],
    [{"role": "user", "content": "What is the alpha decay mode of Uranium-238 (238U) into Thorium-234 (234Th)? What is the resulting nuclide and energy released in the process?"}]
]
input_size = len(messages)
batched_texts = [
    tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) 
    for msg in messages
]
tokenizer.padding_side = "left"
inputs = tokenizer(batched_texts, padding=True, truncation=True, return_tensors="pt").to(model.device)

attention_mask = inputs["attention_mask"]


def sample_token(log_probs, num_sample): # logits is [batch_size, vocab_size]
    logits_temp = log_probs / temperature
    probs = F.softmax(logits_temp, dim=-1)

    topk_probs, indices = torch.topk(probs, dim=-1, k=top_k) # this k is for top_k sampling
    cum_topk_probs = torch.cumsum(topk_probs, dim=-1)

    sorted_indices_to_remove = cum_topk_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False
    topk_probs.masked_fill_(sorted_indices_to_remove, 0.0)

    renormalized_probs = topk_probs / torch.sum(topk_probs, dim=-1, keepdim=True)
    sampled_topp_token = torch.multinomial(renormalized_probs, num_samples=num_sample, replacement=True)

    sampled_ids = torch.gather(indices, -1, sampled_topp_token)
    sampled_id_prob = torch.gather(log_probs, -1, sampled_ids)

    return sampled_id_prob.view(-1), sampled_ids.view(-1, 1)

kv_cache = None
full_trajectories = inputs["input_ids"].clone() # [batch_size, seq_len]
finished_sequences = []


with torch.no_grad():
    for _ in range(TRAJECTORY_LENGTH//BLOCK_SIZE):
        '''
        intiially, kv_cache is of the form tuple(tuple(tensor, tensor)...) where the outer tuple traverses over layers
        since we're taking multiple samples within each batch, we would need to repeat the kv_cache num_sample times
        so that in future generation steps, we don't encounter errors. to this effect, we use tree_map, which iterates through
        complicated structures and only operates the specified function to the innermost tensors,, perfect for this use case !
        repeat_interleave is so that instead of repeating the batches like (b0,b1,b0,b1), we interleave them to get (b0,b0,b1,b1)
        '''
        if not kv_cache:
            output = model(**inputs, use_cache=True)
        else:
            attention_mask = F.pad(attention_mask, (0, 1), value=1)
            output = model(last_token_id, past_key_values=kv_cache, attention_mask=attention_mask, use_cache=True)
        # [batch_size, vocab_size], slicing out at -1 squeezes out the dimension
        last_token_log_prob = F.log_softmax(output.logits[:, -1, :], dim=-1)

        kv_cache = output.past_key_values # inner tensor is of the form [batch_size, num_heads, seq_len, head_dim]
        if hasattr(kv_cache, "to_legacy_cache"):
            kv_cache = kv_cache.to_legacy_cache()
        kv_cache = tree_map(lambda x: torch.repeat_interleave(x, repeats=L, dim=0), kv_cache)
        attention_mask = torch.repeat_interleave(attention_mask, repeats=L, dim=0)

        sentence_prob, last_token_id = sample_token(last_token_log_prob, num_sample=L)
        gen_output = last_token_id.clone()
        # below is replaced by the sample_token function, as we don't want greedy decoding
        # sentence_prob, last_token_id = torch.max(last_token_log_prob, dim=-1)

        # below is replaced by the torch.max function which performs it in 1 step
        # last_token_id = torch.argmax(last_token_log_prob, dim=-1) # [batch]
        # sentence_prob = torch.gather(last_token_log_prob, -1, last_token_id.unsqueeze(-1))
        for _ in range(BLOCK_SIZE-1):
            # next_mask_bit = torch.ones((attention_mask.shape[0], 1), device=model.device, dtype=attention_mask.dtype)
            # attention_mask = torch.cat((attention_mask, next_mask_bit), dim=-1)
            attention_mask = F.pad(attention_mask, (0, 1), value=1) # this is an optimized way of performing the above

            # need to convert to Dynamic cache from the legacy tuple system
            if isinstance(kv_cache, tuple):
                kv_cache = DynamicCache.from_legacy_cache(kv_cache)

            output = model(last_token_id, past_key_values=kv_cache, attention_mask=attention_mask, use_cache=True)
            kv_cache = output.past_key_values
            if hasattr(kv_cache, "to_legacy_cache"):
                kv_cache = kv_cache.to_legacy_cache()

            last_token_log_prob = F.log_softmax(output.logits[:, -1, :], dim=-1)
            # only sampling once, as we've already created num_sample different generations for each batch above
            step_log_prob, last_token_id = sample_token(last_token_log_prob, num_sample=1)
            gen_output = torch.cat((gen_output, last_token_id), dim=-1)

            sentence_prob += step_log_prob
        
        sentence_prob = sentence_prob.view(-1, L) # [batch_size, num_sample]
        gen_output = gen_output.view(inputs["input_ids"].shape[0], L, -1)

        topk_sentence_probs, topk_sentence_indices_2d = torch.topk(sentence_prob, k, dim=-1) # [batch_size, k]
        topk_sentence_indices_3d = topk_sentence_indices_2d.unsqueeze(-1).expand(-1, -1, gen_output.shape[2]) # [batch_size, num_samples, seq_len]
        topk_sentences = torch.gather(gen_output, 1, topk_sentence_indices_3d) # [batch_size, num_samples, seq_len]

        batch_size, chunk_size = topk_sentences.shape[0], k // 2

        # reshape x to be [batch_size, num_samples, num_heads, seq_len, vocab_size] so that the indices work for gathering
        def select_topk_kv(x):
            _, num_heads, S, D = x.shape
            x = x.view(batch_size, L, num_heads, S, D)

            expanded_indices = topk_sentence_indices_2d.view(batch_size, k, 1, 1, 1).expand(-1, -1, num_heads, S, D)

            return torch.gather(x, 1, expanded_indices).view(batch_size * k, num_heads, S, D)

        kv_cache = tree_map(select_topk_kv, kv_cache)
        kv_cache_view = tree_map(lambda x: x.view(batch_size, k, x.shape[1], x.shape[2], x.shape[3]), kv_cache)
        # attention mask shape is [batch_size * L, seq_len]
        attention_mask_view = attention_mask.view(batch_size, L, attention_mask.shape[-1])
        attention_mask = torch.gather(attention_mask_view, 1, topk_sentence_indices_2d.view(batch_size, k, 1).expand(-1, -1, attention_mask.shape[-1]))
        attention_mask = attention_mask.view(batch_size, k, attention_mask.shape[-1])

        kv_cache = []
        new_attention_mask = []
        batched_last_token_id = []
        chosen_blocks = []

        # this step would require num_samples * topk or 8 * 8 = 64 generations per batch, so i will go batch by batch
        # and split each batch into 2, so 32 generations per input, to ensure no OOM
        for b_idx in range(batch_size):
            batch_tokens = []
            batch_probs = []
            for mini_batch in range(2):
                start_idx = mini_batch * chunk_size
                end_idx = (mini_batch + 1) * chunk_size

                curr_batch = topk_sentences[b_idx, start_idx:end_idx, :]
                curr_batch = torch.repeat_interleave(curr_batch, repeats=MC_BUDGET, dim=0)
                
                def process_kv(x):
                    chunked_kv = x[b_idx, start_idx:end_idx, ...]# [batch_size, K, H, S, D] -> [chunk_size, H, S, D]
                    return torch.repeat_interleave(chunked_kv, repeats=MC_BUDGET, dim=0) # [chunk_size * k, H, S, D]
                
                curr_kv_cache = tree_map(process_kv, kv_cache_view)
                curr_attention_mask = attention_mask[b_idx, start_idx:end_idx, :]
                curr_attention_mask = torch.repeat_interleave(curr_attention_mask, repeats=MC_BUDGET, dim=0)
                curr_attention_mask = F.pad(curr_attention_mask, (0, 1), value=1)

                if isinstance(curr_kv_cache, tuple):
                    curr_kv_cache = DynamicCache.from_legacy_cache(curr_kv_cache)

                output = model(curr_batch[:, -1:], past_key_values=curr_kv_cache, attention_mask=curr_attention_mask, use_cache=True)
                curr_kv_cache = output.past_key_values

                last_token_log_prob = F.log_softmax(output.logits[:, -1, :], dim=-1)
                chunk_sentence_prob, last_token_id = sample_token(last_token_log_prob, num_sample=1)
                chunk_gen_output = last_token_id.clone()

                for _ in range(HORIZON_LENGTH-1):
                    curr_attention_mask = F.pad(curr_attention_mask, (0, 1), value=1)
                    output = model(last_token_id, past_key_values=curr_kv_cache, attention_mask=curr_attention_mask, use_cache=True)
                    curr_kv_cache = output.past_key_values

                    last_token_log_prob = F.log_softmax(output.logits[:, -1, :], dim=-1)
                    step_log_prob, last_token_id = sample_token(last_token_log_prob, num_sample=1)
                    chunk_gen_output = torch.cat((chunk_gen_output, last_token_id), dim=-1) # shape of [num_samples/2 * k, 1] or [32, 1]

                    chunk_sentence_prob += step_log_prob # shape of [num_samples/2 * k] or [32]

                batch_tokens.append(chunk_gen_output)
                batch_probs.append(chunk_sentence_prob)
            
            all_final_tokens = torch.cat(batch_tokens, dim=0)
            all_final_probs = torch.cat(batch_probs, dim=0) # [num_samples, M]

            power_prob = torch.exp(all_final_probs * (alpha-1)) #using exp here, as we're calculating log softmax
            power_prob = power_prob.view(-1, MC_BUDGET) # [num_samples, M]
            total_sum = torch.sum(power_prob, dim=1, keepdim=True)

            zeta = (1/MC_BUDGET) * total_sum.squeeze(1)
            zeta_loo = 1/(MC_BUDGET-1) * (total_sum - power_prob) # [num_samples, M]

            p_hat_denominator = torch.sum(torch.exp(topk_sentence_probs[b_idx, ...] * alpha) * zeta, dim=-1) + eps # scalar value
            p_hat = (torch.exp(topk_sentence_probs[b_idx, ...] * alpha) * zeta)/ p_hat_denominator # [k]

            p_hat_loo_denominator = torch.sum(torch.exp(topk_sentence_probs[b_idx, ...] * alpha).unsqueeze(-1) * zeta_loo, dim=0) + eps # [M]
            p_hat_loo = (torch.exp(topk_sentence_probs[b_idx, ...] * alpha).unsqueeze(-1) * zeta_loo)/ p_hat_loo_denominator # [k, M]

            jk_probs = MC_BUDGET*p_hat - (((MC_BUDGET - 1)/MC_BUDGET) * torch.sum(p_hat_loo, dim=-1)) # [k]
            jk_probs = torch.nan_to_num(jk_probs, nan=-100.0, posinf=100.0, neginf=-100.0) #adjusting for any nan values
            
            # _, rollout_id = torch.max(jk_probs, dim=-1)
            jk_probs_dist = torch.clamp(jk_probs, min=0.0)
            jk_probs_dist = jk_probs_dist / torch.sum(jk_probs_dist, dim=-1, keepdim=True)
            rollout_id = torch.multinomial(jk_probs_dist, num_samples=1).squeeze(-1)

            kv_cache.append(tree_map(lambda x: x[b_idx:b_idx+1, rollout_id, ...], kv_cache_view))
            new_attention_mask.append(attention_mask[b_idx:b_idx+1, rollout_id, :])

            chosen_blocks.append(topk_sentences[b_idx:b_idx+1, rollout_id, :])
            batched_last_token_id.append(topk_sentences[b_idx:b_idx+1, rollout_id, -1])
        
        full_trajectories = torch.cat((full_trajectories, torch.cat(chosen_blocks, dim=0)), dim=-1)
        kv_cache = tree_map(lambda *args: torch.cat(args, dim=0), *kv_cache)
        attention_mask = torch.cat(new_attention_mask, dim=0)
        last_token_id = torch.stack(batched_last_token_id, dim=0)

        # check for batches with </think>
        finished_mask = (torch.cat(chosen_blocks, dim=0) == THINK_END_ID).any(dim=-1)
        if finished_mask.any():
            finished_kv = tree_map(lambda x: x[finished_mask], kv_cache)
            finished_atn_mask = torch.unbind(attention_mask[finished_mask], dim=0)
            finished_batch = torch.unbind(full_trajectories[finished_mask], dim=0)

            for i in range(len(finished_batch)):
                single_kv = tree_map(lambda x: x[i:i+1, ...], finished_kv)
                if isinstance(single_kv, tuple):
                    single_kv = DynamicCache.from_legacy_cache(single_kv)
                finished_sequences.append((finished_batch[i].unsqueeze(0), single_kv, finished_atn_mask[i].unsqueeze(0)))

            # ensuring only unfinished sequence tidbits remain
            kv_cache = tree_map(lambda x: x[~finished_mask], kv_cache)
            attention_mask = attention_mask[~finished_mask]
            last_token_id = last_token_id[~finished_mask]
            full_trajectories = full_trajectories[~finished_mask]

        if isinstance(kv_cache, tuple):
            kv_cache = DynamicCache.from_legacy_cache(kv_cache)
        
        if full_trajectories.size(0) == 0:
            break


    frequency_penalty = 0.5
    final_completed_sequences = []
    for seq_ids, kv_cache, attn_mask in finished_sequences:
        current_input_id = seq_ids[:, -1:] 
        for _ in range(2000): # setting an upper limit for generation
            attn_mask = F.pad(attn_mask, (0, 1), value=1)
            outputs = model(input_ids=current_input_id, past_key_values=kv_cache, attention_mask=attn_mask, use_cache=True)
            
            logits = outputs.logits[:, -1, :]
            counts = torch.bincount(seq_ids[0], minlength=logits.shape[-1])
            logits = logits - (counts * frequency_penalty)
            
            next_token_id = torch.argmax(logits, dim=-1, keepdim=True)
            
            seq_ids = torch.cat([seq_ids, next_token_id], dim=-1)
            current_input_id = next_token_id
            kv_cache = outputs.past_key_values
            
            if next_token_id.item() == eos_token_id:
                break
                
        final_completed_sequences.append(seq_ids.squeeze(0))


final_text = tokenizer.batch_decode(final_completed_sequences, skip_special_tokens=True)
for i, text in enumerate(final_text):
    print(f"Batch {i} Output:\n{text}\n")