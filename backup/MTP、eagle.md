问题记录；
外部是如何调用DeepSeekMTP
原始eagle及其代码
验证的流程
在草稿流程scheduler是如何分配KV的？调度机制有什么区别？
model_runner中的postprocess调用时机。
draftmodel 的时候加载mtp模型权重的


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


 draft模型生成
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

# 验证阶段
