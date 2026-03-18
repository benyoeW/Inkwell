问题记录；
外部是如何调用DeepSeekMTP
拒绝采样的具体函数是怎样
验证的流程
在草稿流程scheduler是如何分配KV的？调度机制有什么区别？
model_runner中的postprocess调用时机。
draftmodel 的时候加载mtp模型权重的

全局整体调用逻辑：
```
run_busy_loop()
  ├─ _process_input_queue()   ← 从 input_queue 读取新请求/abort
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
_process_engine_step：单次执行，不循环

```


验证和生成在同一次 sample_tokens 里完成
投机解码的一个完整推理步骤是这样流动的：

```
第 N 轮 execute_model
  ├─ 目标模型前向（带 draft tokens 作为输入）
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

# draft token 生成阶段
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



主模型执行前向
model_runner.execute_model()
    │
    │  主模型前向
    ├──→ DeepseekV3ForCausalLM.forward()
    │         └──→ 返回 hidden_states (最后一层输出) 
                             # 这里将信息到GPUModelRunner类中的execute_model_state参数
                             # 这里也更新了kv_connector_output，获取已完成的请求传输状态


 draft验证与生成阶段
 sample_tokens(hidden_states)
    │         └──→ 返回 sampled_token_ids
    │
    │ MTP 投机解码提议
    └──→ propose_draft(hidden_states)    # 在sample_tokens函数的最后部分执行
              │
              └──→ 调用EagleSpeculator.propose() # 上一个函数只调用本函数
                        │
                        │  首步: 用主模型的 hidden_states + sampled_tokens
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
**注意num_computed_token是target model生成的KV，draft model只负责产生token即可**

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