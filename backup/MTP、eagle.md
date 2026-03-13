问题记录；
外部是如何调用DeepSeekMTP
原始eagle及其代码
eagle是如何多步进行draft的？目前只看到从model_runner.sample_tokens()发起调用之后进行一步draft；
eagle mtp vllm流程实现
草稿的流程
验证的流程

```
model_runner.execute_model()
    │
    │ ① 主模型前向
    ├──→ DeepseekV3ForCausalLM.forward()
    │         └──→ 返回 hidden_states (最后一层输出)
    │
    │ ② 主模型采样
    ├──→ sample_tokens(hidden_states)
    │         └──→ 返回 sampled_token_ids
    │
    │ ③ MTP 投机解码提议
    └──→ propose_draft(hidden_states)
              │
              └──→ EagleSpeculator.propose()
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
                        ├──→ compute_logits(new_hidden_states)
                        │         └──→ SharedHead.norm + head ──→ logits
                        │
                        ├──→ sample draft_token
                        │
                        │  后续步: 用 MTP 自己的 hidden_states + draft_token (循环)
                        └──→ 重复 run_model → compute_logits → sample
                              (共 num_speculative_tokens 步)
```

