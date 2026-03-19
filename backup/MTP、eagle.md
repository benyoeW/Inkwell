# 全局整体调用逻辑：
```
run_busy_loop()
  ├─ _process_input_queue()   ← 从 input_queue 读取新请求/abort     # 将请求传输到引擎侧，然后执行下面的函数
  └─ _process_engine_step() 
       ├─ step_fn()            ← 即 step() 或 step_with_batch_queue()
       │    ├─ scheduler.schedule()                    ← ① 调度，决定本轮送哪些请求
       │    ├─ executor.execute_model(sched_output)   ← ② 触发 GPU 前向
       │    ├─ scheduler.get_grammar_bitmask()         ← ③ 结构化输出的语法掩码
       │    ├─ executor.sample_tokens(grammar_output)  ← ④ 采样    这里会进行拒绝采样 & draft model的推测解码前向
       │    └─ scheduler.update_from_output()          ← ⑤ 处理结果、判断截止
       └─ post_step()
            └─ executor.take_draft_token_ids()         ← ⑥ 取出 draft tokens（同步模式）
                 └─ scheduler.update_draft_token_ids() ← ⑦ 写入 request.spec_token_ids

每个函数的循环与终止条件：
_process_input_queue：
       获取所有的请求，并处理完所有获取的请求。这里的处理主要指的是将请求送入到引擎侧
_process_engine_step：
       单次执行，不循环
整体调用逻辑中，只有run_busy_loop是一直循环的，每个step结束之后会从run_busy_loop头部继续执行
```

验证和生成在同一次 sample_tokens 里完成
投机解码的一个完整推理步骤是这样流动的：

```
第 N 轮 execute_model
  ├─ execute_model()（带 draft tokens 作为输入,得到target model的验证tokens）
  ├─ sample_tokens():
  │    ├─ self.sample()      ← ①（拒绝采样的验证） 验证上一轮的 draft tokens
  │    ├─ postprocess()      ← ② 更新状态（已被接受的 token + KV cache 进度）
  │    └─ propose_draft()    ← ③ 生成下一轮的新 draft tokens
  └─ 返回结果

***************************************************************************


第 N 轮（验证第 N-1 轮的 draft）:
  目标模型输入: [正常 token..., d0, d1, d2, d3]
                                 ↑上一轮 draft
  rejection_sample → 接受 [d0, d1, bonus]，拒绝 [d2, d3]
  postprocess     → last_sampled = bonus，num_computed += 3
  propose_draft   → 新 draft [d0', d1', d2', d3']（基于 bonus 的 hidden state）
  写回 draft_tokens

第 N+1 轮（验证第 N 轮的 draft）:
  目标模型输入: [正常 token..., bonus, d0', d1', d2', d3']
  ...

```

# draft token 验证与生成阶段
```
调用链路总揽
EngineCore.step()                         # 引擎核心
  └─> Executor.execute_model()            # 执行器 (UniProc/MultiProc/Ray)    ⭐️
      └─> WorkerBase.execute_model()     # Worker 基类
           └─> GPUWorker.execute_model() # GPU Worker
                └─> GPUModelRunner.execute_model()   ← 目标模型前向
                      (返回 None，表示需要继续调用 sample_tokens)
   
  └─> Executor.sample_tokens()  ⭐️
       └─> GPUWorker.sample_tokens()
            └─> GPUModelRunner.sample_tokens()        ← 采样 + 拒绝采样 + 草稿提案



####################### execute_model #######################

model_runner.execute_model()
    │
    │  主模型前向
    ├──→ DeepseekV3ForCausalLM.forward()
    │         └──→ 返回 hidden_states (最后一层输出) 
                             # 这里将信息到GPUModelRunner类中的execute_model_state参数
                             # 这里也更新了kv_connector_output，获取已完成的请求传输状态


####################### sample_tokens #######################

 sample_tokens(hidden_states)
    │         └──→ 返回 sampled_token_ids
    │
    │ MTP 投机解码提议
    └──→ propose_draft(hidden_states)    # 在sample_tokens函数的最后部分执行
              │
              └──→ 调用EagleSpeculator.propose() # 上一个函数只调用本函数
                        │
                        │  首步: 用主模型的 hidden_states + sampled_tokens
                        │  后续: 用前面draft model生成的的 hidden_states + sampled_tokens
                        ├──→ run_model() ──→ DeepSeekMTP.forward()
                        │         │              │
                        │         │              └──→ DeepSeekMultiTokenPredictor.forward()
                        │         │                       │
                        │         │                       ├── embed_tokens(input_ids)
                        │         │                       └── MTPLayer.forward(embeds, hidden_states)
                        │         │                               ├── enorm(embeds) + hnorm(hidden)
                        │         │                               ├── eh_proj(concat)
                        │         │                               └── mtp_block(transformer层)
                        │         │
                        │         └──→ 返回 new_hidden_states
                        │
                        ├──→ compute_logits(new_hidden_states)     # 这里会加载对应的mtp layer层以及共享的lm_head
                        │         └──→ SharedHead.norm + head ──→ logits
                        │                         # 这里做了两步操作：RMSNorm + Linear （hidden --> volcab_size)
                        │
                        ├──→ sample draft_token  
                        │
                        │  后续步: 用 MTP 自己的 hidden_states + draft_token (循环)  
                        └──→ 重复 run_model → compute_logits → sample
                              (共重复了 num_speculative_tokens-1 步,加上开头的1步一共num_speculative_tokens步)


一些详细调用链路和注意点：

```

# 关于KVcache
draft token 的 KV 不写入 prefix cache
new tokens 包含 verified 和 unverified draft tokens，
但只缓存 verified tokens（通过 request.num_tokens 来 cap）
**注意num_computed_token是target model生成的KV**
**draft model负责产生token，在多步前向的时候也会保存和服用KV**

在propose_draft过程中，除了第0步输入 的token长度是变长的，后续进行前向的输入token长度都是1+上一步的hidden_state
在后续步数中，，之前draft otken的KVcache是已经被保存的，因为在scheduler中显存已经申请了；
在拒绝采样之后会更新computed tokens，没有被采纳的token其KVcache可以被覆盖；

```
Scheduler._schedule_running_requests()
    │
    ├── num_new_tokens = num_tokens_with_spec(含上轮draft) - num_computed_tokens
    │
    └── kv_cache_manager.allocate_slots(
            num_new_tokens,
            num_lookahead_tokens = num_spec_tokens  ← 为下一轮 draft 预留
        )
            │
            ├── num_tokens_need_slot = computed + new + lookahead
            ├── 按 num_tokens_need_slot 分配物理 block
            └── prefix cache 只写入到 request.num_tokens（不含 draft）
                   ↑
                   确保被拒绝的 draft token 不污染 prefix cache


```

# deepseek mtp
在propose_draft过程中，除了第0步输入 的token长度是变长的，后续进行前向的输入token长度都是1+上一步的hidden_state
在后续步数中，，之前draft otken的KVcache是已经被保存的，因为在scheduler中显存已经申请了；
在拒绝采样之后会更新computed tokens，没有被采纳的token其KVcache可以被覆盖；

总结：
输入：token + 上一轮的hidden_state
处理：
        [token]-->embedding-->rmsNorm
                                                                     --> cat --> linear --> mtp_block --> residual --> shared_head
        [hidden_state]-->rmsNorm

具体细节在下方查看

```
Step 1: embed_tokens(input_ids)
  input_ids         [8]
        ↓  VocabParallelEmbedding
  inputs_embeds     [8, 7168]

Step 2: 掩码 position==0 的 embedding 置零
  inputs_embeds     [8, 7168]   → 第0和第4行（position=0）被清零

Step 3: enorm(inputs_embeds)
  inputs_embeds     [8, 7168]   → [8, 7168]   (RMSNorm，形状不变)

Step 4: hnorm(previous_hidden_states)
  previous_hidden_states  [8, 7168]  → [8, 7168]  (RMSNorm，形状不变)

Step 5: concat + eh_proj
  torch.cat([inputs_embeds, previous_hidden_states], dim=-1)
                    [8, 7168] + [8, 7168] → [8, 14336]
  eh_proj (Linear: 14336 → 7168)
  hidden_states     [8, 14336] → [8, 7168]

Step 6: mtp_block (DeepseekV2DecoderLayer)
  ┌── input_layernorm(hidden_states)
  │     residual         [8, 7168]   (clone)
  │     hidden_states    [8, 7168]   (RMSNorm)
  │
  ├── self_attn (MLA Attention)
  │     positions        [8]
  │     hidden_states    [8, 7168] → [8, 7168]
  │     (内部: Q/K/V投影、RoPE、PagedAttention、输出投影)
  │
  ├── post_attention_layernorm(hidden_states, residual)
  │     residual        += hidden_states  (fused add)  → [8, 7168]
  │     hidden_states    [8, 7168]   (RMSNorm)
  │
  └── mlp (DeepseekV2MoE)
        hidden_states    [8, 7168]
        gate(hidden_states) → router_logits  [8, n_routed_experts]
        experts(...)    → [8, 7168]
        + shared_experts → [8, 7168]
        hidden_states    [8, 7168]

  return hidden_states [8, 7168], residual [8, 7168]

Step 7: 残差连接
  hidden_states = residual + hidden_states  → [8, 7168]

Step 8: compute_logits
  shared_head.norm(hidden_states)   [8, 7168] → [8, 7168]  (RMSNorm)
  logits_processor(head, ...)
    lm_head (Linear: 7168 → 129280)
  logits           [8, 7168] → [8, 129280]

```

**mtp_layers**
对于deepseek v3来说，其mtp_layers只有一个（mtp_start_layer_idx：61，num_mtp_layers：1）
```
DeepSeek-V3 checkpoint 结构
├── model.embed_tokens.weight           ← 主模型 embedding
├── model.layers.0 ~ 60.*              ← 主模型 61 层 Transformer
└── model.layers.61.*                  ← MTP 层权重
    ├── embed_tokens.weight            ← 与主模型 embed_tokens 共享（路径重写）
    ├── enorm.weight
    ├── hnorm.weight
    ├── eh_proj.weight
    ├── shared_head.norm.weight
    ├── shared_head.head.weight        ← lm_head，与主模型 lm_head 共享（DeepSeek-V3 报告）
    └── self_attn.*, mlp.*             ← 独立的 MTP Transformer block 权重
```

假设有多层的话，根据其实现逻辑，其embedding和shared_head还是只会共享一层；

# 其他
代码中的具体逻辑参考文件内注释，主要是：
vllm/v1/worker/gpu/model_runner.py
vllm/model_executor/models/deepseek_mtp.py
vllm/v1/core/sched/scheduler.py
vllm/v1/engine/core.py
vllm/v1/worker/gpu/spec_decode/eagle.py
vllm/v1/worker/gpu/spec_decode/rejection_sample.py

