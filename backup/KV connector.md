1. RequestA进入等待队列
2. 调度器发现RequestA有远程KV → 设置为WAITING_FOR_REMOTE_KVS
3. 启动异步传输，RequestA放入skipped队列
4. 调度器处理RequestB、RequestC（不阻塞！）
5. Worker异步传输KV缓存到本地
6. 下一轮调度：检查RequestA传输状态
7. 传输完成 → RequestA状态改为WAITING
8. RequestA可以进行正常调度分配


## 整体流程：
可以先看一下waiting的代码；

一个请求从sheduler侧开始：
  **某schedule轮次**：先获取远端可传输的token --> 申请显存空间 --> 构造远端传输的元数据 --> 异步的话更改请求状态(end)
  **某schedule轮次**：若请求状态为等待KV传输-->是否传输就绪-->是则为申请的显存缓存KV，并更改请求状态--〉调度（end）
                                                                                                      -->否则让请求进入临时队列（end）

**某update_from_output轮次**：（......）--》worker侧完成传输之后，将 KVconnector 中完成的请求放到 scheduler类中的 finished_recving_kv_req_ids 中
**某schedule轮次**：_update_waiting_for_remote_kv 检查finished_recving_kv_req_ids远程传输KV请求是否就绪，若就绪更新分配block的元数据（**传输完成后则说明该KV已经分配到对应block当中了**）

shceduler.py代码
```
        # Next, schedule the WAITING requests. ==========================================================
        if not preempted_reqs:
            while self.waiting and token_budget > 0:
                if len(self.running) == self.max_num_running_reqs:
                    break

                request = self.waiting.peek_request()

                # KVTransfer: skip request if still waiting for remote kvs.
                if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                    # 检查远程KV是否就绪，并存入分配的显存
                    is_ready = self._update_waiting_for_remote_kv(request)
                    if is_ready:
                        if request.num_preemptions:
                            # We must be loading for a resumed preemption
                            # rather than a new request.
                            request.status = RequestStatus.PREEMPTED
                        else:
                            request.status = RequestStatus.WAITING
                    else:
                        logger.debug(
                            "%s is still in WAITING_FOR_REMOTE_KVS state.",
                            request.request_id,
                        )
                        self.waiting.pop_request()
                        # 这里放入了临时队列
                        skipped_waiting_requests.prepend_request(request)
                        continue
                        # 然后在调度其他请求结束的时候，检查临时队列，将临时队列的req放到waitting队列

                # Skip request if the structured output request is still waiting
                # for FSM compilation.
                if request.status == RequestStatus.WAITING_FOR_FSM:
                    structured_output_req = request.structured_output_request
                    if structured_output_req and structured_output_req.grammar:
                        request.status = RequestStatus.WAITING
                    else:
                        self.waiting.pop_request()
                        skipped_waiting_requests.prepend_request(request)
                        continue

                # Streaming: skip request if still waiting for next streaming req.
                if request.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
                    assert not request.streaming_queue
                    self.waiting.pop_request()
                    skipped_waiting_requests.prepend_request(request)
                    continue

                # Check that adding the request still respects the max_loras
                # constraint.
                if (
                    self.lora_config
                    and request.lora_request
                    and (
                        len(scheduled_loras) == self.lora_config.max_loras
                        and request.lora_request.lora_int_id not in scheduled_loras
                    )
                ):
                    # Scheduling would exceed max_loras, skip.
                    self.waiting.pop_request()
                    skipped_waiting_requests.prepend_request(request)
                    continue

                num_external_computed_tokens = 0
                load_kv_async = False

                # Get already-cached tokens.

                # num_computed_tokens ==0 表示还没开始KV传输的相关计算
                # 抢占KVcache会抢占整个请求的，不会只抢占单个请求的部分KV，
                # 所以如果num_computed_tokens！=0，不需要connector加载远程KV
                if request.num_computed_tokens == 0:
                    # Get locally-cached tokens.
                    new_computed_blocks, num_new_local_computed_tokens = (
                        self.kv_cache_manager.get_computed_blocks(request)
                    )

                    # Get externally-cached tokens if using a KVConnector.
                    # 通过connector获取远程可以加载多少个token的KVcache
                    if self.connector is not None:
                        # 通过connector获取远程是否有已经计算的token的KVcache
                        ext_tokens, load_kv_async = (
                            self.connector.get_num_new_matched_tokens(
                                request, num_new_local_computed_tokens
                            )
                        )
                        # 表示无法确定查询情况（不是为0），暂时不调度本请求，将本请求放入临时队列
                        if ext_tokens is None:
                            # The request cannot be scheduled because
                            # the KVConnector couldn't determine
                            # the number of matched tokens.
                            self.waiting.pop_request()
                            skipped_waiting_requests.prepend_request(request)
                            continue

                        request.num_external_computed_tokens = ext_tokens
                        num_external_computed_tokens = ext_tokens

                    # Total computed tokens (local + external).
                    # 加载远程之后重新赋值num_computed_tokens
                    num_computed_tokens = (
                        num_new_local_computed_tokens + num_external_computed_tokens
                    )
                else:
                    # KVTransfer: WAITING reqs have num_computed_tokens > 0
                    # after async KV recvs are completed.
                    # 在传输好KV之前就已经标记好了该请求的num_computed_tokens
                    new_computed_blocks = self.kv_cache_manager.empty_kv_cache_blocks
                    num_new_local_computed_tokens = 0
                    num_computed_tokens = request.num_computed_tokens

                encoder_inputs_to_schedule = None
                external_load_encoder_input = []
                new_encoder_compute_budget = encoder_compute_budget

                # 对于offloading connector，如果有命中远端KV，load_kv_async默认为异步，否则为同步

                # 异步加载，加载本请求的KVcache同时处理其他请求
                if load_kv_async:
                    # KVTransfer: loading remote KV, do not allocate for new work.
                    assert num_external_computed_tokens > 0
                    # 这里不进行调度，只分配显存
                    num_new_tokens = 0
                    # 后面设置请求的状态为WAITING_FOR_REMOTE_KVS

                # 同步加载，获取调度的token个数
                else:
                    # Number of tokens to be scheduled.
                    # We use `request.num_tokens` instead of
                    # `request.num_prompt_tokens` to consider the resumed
                    # requests, which have output tokens.
                    num_new_tokens = request.num_tokens - num_computed_tokens
                    threshold = self.scheduler_config.long_prefill_token_threshold
                    if 0 < threshold < num_new_tokens:
                        num_new_tokens = threshold

                    # chunked prefill has to be enabled explicitly to allow
                    # pooling requests to be chunked
                    if (
                        not self.scheduler_config.enable_chunked_prefill
                        and num_new_tokens > token_budget
                    ):
                        # If chunked_prefill is disabled,
                        # we can stop the scheduling here.
                        break

                    num_new_tokens = min(num_new_tokens, token_budget)
                    assert num_new_tokens > 0

                    # Schedule encoder inputs.
                    if request.has_encoder_inputs:
                        (
                            encoder_inputs_to_schedule,
                            num_new_tokens,
                            new_encoder_compute_budget,
                            external_load_encoder_input,
                        ) = self._try_schedule_encoder_inputs(
                            request,
                            num_computed_tokens,
                            num_new_tokens,
                            encoder_compute_budget,
                            shift_computed_tokens=1 if self.use_eagle else 0,
                        )
                        if num_new_tokens == 0:
                            # The request cannot be scheduled.
                            break

                if self.need_mamba_block_aligned_split:
                    num_new_tokens = self._mamba_block_aligned_split(
                        request,
                        num_new_tokens,
                        num_new_local_computed_tokens,
                        num_external_computed_tokens,
                    )
                    if num_new_tokens == 0:
                        break

                # Handles an edge case when P/D Disaggregation
                # is used with Spec Decoding where an
                # extra block gets allocated which
                # creates a mismatch between the number
                # of local and remote blocks.
                effective_lookahead_tokens = (
                    0 if request.num_computed_tokens == 0 else self.num_lookahead_tokens
                )

                num_encoder_tokens = (
                    self._num_encoder_max_input_tokens
                    if self.is_encoder_decoder and request.has_encoder_inputs
                    else 0
                )
                # 在本地申请新的KV cache的空间
                new_blocks = self.kv_cache_manager.allocate_slots(
                    request,
                    num_new_tokens, # 总的token数量 - 已经计算KV的token数量
                    num_new_computed_tokens=num_new_local_computed_tokens, # 本地的KVcache
                    new_computed_blocks=new_computed_blocks,
                    num_lookahead_tokens=effective_lookahead_tokens,
                    num_external_computed_tokens=num_external_computed_tokens, # 远程已经计算好的KVcache
                    delay_cache_blocks=load_kv_async, # 这里应该是是否先延迟远端KVcache的加载
                    num_encoder_tokens=num_encoder_tokens,
                )

                if new_blocks is None:
                    # The request cannot be scheduled.

                    # NOTE: we need to untouch the request from the encode cache
                    # manager
                    if request.has_encoder_inputs:
                        self.encoder_cache_manager.free(request)
                    break

                # KVTransfer: the connector uses this info to determine
                # if a load is needed. Note that
                # This information is used to determine if a load is
                # needed for this request.

                # ========================================================================

                # KVcache传输触发链路：
                # scheduler端 在申请完显存之后，通过 update_state_after_alloc 将元数据传输到worker侧
                # Worker端    持续监听传输队列，执行传输，并更新scheduler端的请求状态
                # scheduler端 _update_waiting_for_remote_kv 检查远程KV是否就绪，若就绪则存入分配好的显存
                
                # ========================================================================

                if self.connector is not None:
                    # connector更新request的block，远端的KVcache加载的 Metadata
                    self.connector.update_state_after_alloc(   # 如果传输完成，update_state_after_alloc函数会直接返回
                        request,
                        self.kv_cache_manager.get_blocks(request.request_id),
                        num_external_computed_tokens,
                    )

                # Request was already popped from self.waiting
                # unless it was re-added above due to new_blocks being None.
                request = self.waiting.pop_request()

                # 如果是异步的话，更改请求的状态为WAITING_FOR_REMOTE_KVS，不进行调度，添加到临时队列中
                if load_kv_async:
                    # 如果请求远程KV传输完成，
                    # If loading async, allocate memory and put request
                    # into the WAITING_FOR_REMOTE_KV state.
                    skipped_waiting_requests.prepend_request(request)
                    # 将该请求标记为异步加载远端KV状态
                    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
                    continue   # 异步到此end

                # 如果是非异步的话，直接可以进行推理了，在执行模型前向的时候，在进行每个layer的attn计算之前等待加载完成

                self._update_connector_prefix_cache_stats(request)

                # 进行调度，添加到running队列中
                self.running.append(request)
                if self.log_stats:
                    request.record_event(
                        EngineCoreEventType.SCHEDULED, scheduled_timestamp
                    )
                if request.status == RequestStatus.WAITING:
                    scheduled_new_reqs.append(request)
                elif request.status == RequestStatus.PREEMPTED:
                    scheduled_resumed_reqs.append(request)
                else:
                    raise RuntimeError(f"Invalid request status: {request.status}")

                if self.lora_config and request.lora_request:
                    scheduled_loras.add(request.lora_request.lora_int_id)
                # 获取该请求的kv block信息
                req_to_new_blocks[request.request_id] = (
                    self.kv_cache_manager.get_blocks(request.request_id)
                )
                # 记录调度的token数量，缩减token_budget
                num_scheduled_tokens[request.request_id] = num_new_tokens
                token_budget -= num_new_tokens
                request.status = RequestStatus.RUNNING
                request.num_computed_tokens = num_computed_tokens
                # Count the number of prefix cached tokens.
                if request.num_cached_tokens < 0:
                    request.num_cached_tokens = num_computed_tokens

```


**scheduler**端调用build_connector_meta()
构建需要传输的请求元数据（包括块ID、请求ID、token数量）
返回KVConnectorMetadata对象

worker侧的gpu.model_runner在前向计算之前执行：
```
        # Run model.
        if use_cudagraph:
            # Run CUDA graph.
            # NOTE(woosuk): Here, we don't need to pass the input tensors,
            # because they are already copied to the CUDA graph input buffers.

            # 绑定metadata，开始KV传输
            self.kv_connector.pre_forward(scheduler_output)

            # 请求分为KV传输与非KV传输的
            # 对于KV传输的请求，根据connector类型，决定是否异步加载
            #   若为异步传输：scheduler侧会将该请求的 num_new_tokens = 0,该请求本次不会进行前向计算
            #   若为同步传输：num_new_tokens正常计算，通过layer by layer进行前向计算
            # 对于非KV传输的请求：直接进行前向计算

            # 用cuda graph方式进行前向计算
            hidden_states = self.cudagraph_manager.run(
                input_batch.num_tokens_after_padding
            )
        else:
            .......

        # 前向执行完毕之后，
        kv_connector_output = self.kv_connector.post_forward(scheduler_output)   # 
        self.execute_model_state = hidden_states, input_batch, kv_connector_output
        return None
```

**worker**端接收到scheduler_output后：
提取kv_connector_metadata
调用bind_connector_metadata()绑定元数据
调用start_load_kv()开始异步加载
模型前向执行完毕之后调用self.kv_connector.post_forward(scheduler_output)
post_forward调用链路：
│    └── post_forward()      # 获取传输完成的请求 ID                                                    
│          │                                                                    
│         ├ ── get_finished()                                                   
│          │     └── OffloadingConnectorWorker.get_finished()   # 检查所有传输任务是否完成
│          │           └── OffloadingWorker.get_finished() 【offloading_connector.py】
│          │                 └── SingleDirectionOffloadingHandler.get_finished()【cpu_gpu.py】     # 返回完成的 job_id     
│          │                       └── event.query() 检查传输完成              
                  

## 举例：
Scheduler端：
1. 请求A分配blocks [4,5,6], 需要传输35个token
2. 构建ReqMeta(request_id="A", block_ids=[4,5,6], num_tokens=35)
3. 放入P2pNcclConnectorMetadata.requests列表

Worker端：
1. 解析metadata获取blocks [4,5,6]
2. 启动异步DMA传输：
   - block4: tokens 0-15
   - block5: tokens 16-31 
   - block6: tokens 32-34 (最后3个token)
3. 传输完成后标记请求A为可调度状态

注意点：
如果已经通过connector计算好外部token数量等信息，就直接给num_computed_token赋值了，虽说还没加载完成；

对比：
不使用OffloadingConnector：
```
# 调用链
用户 -> vllm.LLM.genreate()           # 入口
      -> vllm/v1/engine.py           # 引擎
      -> vllm/v1/core/sched/scheduler.py  # 基础调度
      -> vllm/v1/worker/model_runner.py   # 模型执行
      -> vllm/attention/layers/*.py  # 注意力计算
```
使用OffloadingConnector：
```
# 调用链（扩展）
用户 -> vllm.LLM.generate()          # 入口
      -> vllm/v1/engine.py           # 引擎（检测KVTransferConfig）
      -> vllm/v1/core/sched/scheduler.py  # 调度+Connector决策
           ↳ vllm/distributed/kv_transfer/kv_connector/factory.py
           ↳ vllm/v1/kv_offload/factory.py
           ↳ vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py
           ↳ vllm/v1/kv_offload/arc_manager.py
      
      -> vllm/v1/worker/model_runner.py   # 执行+传输
           ↳ vllm/v1/worker/kv_connector_model_runner_mixin.py
           
      -> vllm/attention/layers/*.py  # 注意力计算+传输同步
           ↳ vllm/attention/utils/kv_transfer_utils.py
```


connector维护了两个role：scheduler和worker

这两个role对应的方法可以被vllm中对应的结构进行调用

## scheduler
场景设定
用户请求：prompt = "北京是中国的首都，它的人口是"  (10个token)
远端 Prefill 机器 已经算过这个 prompt 的前8个token的 KV Cache
本地 已经缓存了前3个token的 KV Cache
**get_num_new_matched_tokens()**
**update_state_after_alloc()**
**build_connector_meta()**

```
Scheduler Step:

  [新请求 req_A 进来]
       ↓
  get_num_new_matched_tokens(req_A, num_computed=3)
  → 返回 (5, True)  "远端有5个token可以给你"
       ↓
  KVCacheManager 分配 block_0, block_1 给这5个token
       ↓
  update_state_after_alloc(req_A, blocks, num_external=5)
  → Connector 内部记录: "block_0,1 要从远端拉数据"
       ↓
  ... 处理其他请求 ...
       ↓
  build_connector_meta(scheduler_output)
  → 把所有请求的传输任务打包成 metadata
  → 清空内部状态
  → 返回 meta
       ↓
  meta 被发送到 Worker 进程
       ↓
  Worker: start_load_kv() 根据 meta 真正开始传数据
```

## worker
```
═══════════════════════════════════════════════════════
                    WORKER 侧
═══════════════════════════════════════════════════════

⑤ bind_connector_metadata(meta)
  └─ self._connector_metadata = meta
  └─ Worker 现在知道要干什么了

⑥ start_load_kv(forward_context)
  └─ 读取 meta: req_A 需要从 10.0.0.1 拉 block_0, block_1
  └─ 后台线程异步开始拉数据
  └─ 立刻返回，开始前向传播

  [前向传播开始，逐层执行]

  --- Layer 0 ---
⑦  wait_for_layer_load("layer_0")
    └─ 等待 layer_0 的KV数据到达（可能短暂阻塞）
    └─ 数据到了 → 返回

    [执行 layer_0 的 Attention 计算]
    output_0 = attention_0(input, kv_cache[layer_0])

⑧  save_kv_layer("layer_0", kv_cache["layer_0"], attn_meta)
    └─ 提取 req_B 对应的KV数据
    └─ 放入发送队列，异步发送给 10.0.0.2
    └─ 立刻返回

  --- Layer 1 ---
⑦  wait_for_layer_load("layer_1")
    └─ 等 layer_1 数据... → 返回

    [执行 layer_1 的 Attention 计算]

⑧  save_kv_layer("layer_1", ...)
    └─ 异步发送 layer_1 的KV

  --- Layer 2, 3 ... 同上 ---

  [前向传播结束]

⑨ wait_for_save()
  └─ 等所有异步发送完成
  └─ 确保远端收到完整的KV数据
  └─ 返回

⑩ clear_connector_metadata()
  └─ self._connector_metadata = None
  └─ 清理完毕，准备下一次前向

═══════════════════════════════════════════════════════
                  请求结束时 (SCHEDULER 侧)
═══════════════════════════════════════════════════════

[req_A 生成完毕]

⑪ request_finished(req_A, block_ids=[block_0, block_1, block_2])
  └─ 判断是否需要把这个请求的KV异步发给别的缓存节点
  └─ 情况A: 不需要 → 返回 (False, None) → 立刻释放块
  └─ 情况B: 需要  → 返回 (True, params) → 块暂不释放
               → 后台异步发送KV
               → 发完后 get_finished() 返回 req_A
               → Scheduler 收到后才真正释放块
```

### 函数举例解析：
#### get_num_new_matched_tokens
```
举例子：
num_computed_tokens = 40
offloaded_block_size = 32
hits = 2
→ 计算: 32*(1+2)-40 = 56
→ 返回 (56, True)

request.num_tokens = 96 (总token)
offloaded_block_size = 32 (卸载块大小)
num_computed_tokens = 40 (已计算token)
hits = 2 (缓存命中2个连续块)

GPU内存状态 (已计算token: 40):
[■■■■■■■■■■■■■■■■] 卸载块0: token 0-31 (32t, 100%计算)
[■■■■□□□□□□□□□□□□] 卸载块1: token 32-63 (8/32t计算，24t待计算)

卸载存储中的命中块:（由于第0个卸载块完全计算，所以从第1个卸载块开始检查命中了多少个块，这里举例命中了2个块）
[□□□□□□□□□□□□□□□□] 卸载块1剩余: token 40-63 (缓存命中)
[□□□□□□□□□□□□□□□□] 卸载块2: token 64-95 (缓存命中)

→ 可加载56个token (24+32)
加载的时候是以token的每个layer粒度进行加载的，这里返回了token个数


def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int | None, bool]:

        # 卸载块也是有单位的，这里的一个卸载块对应多个GPU块
        # 计算请求总共需要多少个卸载块（不是GPU块）
        num_blocks = request.num_tokens // self.offloaded_block_size # num_tokens ： 请求的总token数量

        # block_size_factor：如果blocksize=16，offloaded_block_size=32，则block_size_factor为2
        assert len(request.block_hashes) // self.block_size_factor == num_blocks
        block_hashes = self._get_block_hashes(request)
        # 预热
        self.manager.touch(block_hashes)

        full_block_tokens = self.offloaded_block_size * num_blocks

        # 如果需要的token还不足一个卸载块的大小就不用搬运了
        if full_block_tokens - num_computed_tokens < self.offloaded_block_size:
            # we can load less than a block, skip
            return 0, False
        # 从已经计算的token的下一个token开始寻找可加载的块
        start_block_idx = num_computed_tokens // self.offloaded_block_size

        # 返回命中的连续块数
        hits = self.manager.lookup(
            self._get_block_hashes(request, start_idx=start_block_idx)
        )
        if hits is None:
            # indicates a lookup that should be tried later
            return None, False
        if hits == 0:
            return 0, False

        num_hit_tokens = (
            self.offloaded_block_size * (start_block_idx + hits) - num_computed_tokens
        )
        logger.debug(
            "Request %s hit %s offloaded tokens after %s GPU hit tokens",
            request.request_id,
            num_hit_tokens,
            num_computed_tokens,
        )
        if num_hit_tokens < self.offloaded_block_size:
            return 0, False

        if self._blocks_being_loaded:
            block_hashes = self._get_block_hashes(
                request, start_idx=start_block_idx, end_idx=start_block_idx + hits
            )

            if any(
                block_hash in self._blocks_being_loaded for block_hash in block_hashes
            ):
                # hit blocks are being loaded, delay request
                logger.debug(
                    "Delaying request %s since some of its blocks are already"
                    " being loaded",
                    request.request_id,
                )
                return None, False

        return num_hit_tokens, True
```



## 卸载

关于KV的卸载每个step会发生什么：

### 阶段 A：Scheduler 调度（schedule()）
A1. allocate_slots 申请 GPU 显存（发生在模型前向之前）
    为新要计算的token在GPU KV cache pool 申请GPU显存，
A2. build_connector_meta 决定 Store 任务
    每一个step都为上一个step的到的新的KV cache检查一下是否有新的KVblock需要store
    TransferSpec 只是记录了"将来要从哪些 GPU block 传到哪些 CPU block"
A3. _update_after_schedule 更新 num_computed_tokens

### 阶段 B：Worker 执行（execute_model）
B1. pre_forward（model_runner.py:949 → kv_connector.py:62）
    提交上一轮积压的 store job！
    创建卸载的CUDA stream，用于KV的offload
    关键1：等待当前推理 CUDA stream 的所有计算完成，再开始 offload（利用wait）
    关键2：stream是为了确保提交的 transfer job串行
B2. 模型前向计算
    将得到的KV写入GPU显存
B3. post_forward
```
# wait_for_save → prepare_store_kv：本步 reqs_to_store 为空，无新任务
# get_finished()：
for job_id, success in self.worker.get_finished():
    # SingleDirectionOffloadingHandler.get_finished() 查询 CUDA event
    # DMA传输 完成了吗？

这里一个请求的卸载传输是否完成取决于两个条件：
1. 数据的传输是否完毕
2. 请求是否结束
只有这样都满足条件之后结束后才会触发 finished_sending → complete_store → is_ready=True

```

### 举例

#### 背景参数确认

gpu_block_size = 16 token（1 个 GPU block 存 16 个 token 的 K/V）
offloaded_block_size = 32 token（2 个 GPU block = 1 个 offloaded block）
block_size_factor = 32 / 16 = 2

prompt = 200 token
需要 GPU block 数 = ⌈200/16⌉ = 13 个（前 12 个满，第 13 个存 8 个 token）
可以凑出的完整 offloaded block = 200 // 32 = 6 个（覆盖前 192 token = 12 个 GPU block）
剩余 8 个 token（第 13 个 GPU block）不满足 offloaded_block_size，本轮不 offload

#### Step 1 结束时的状态总结
```
GPU 显存：
  Block 0~11：完整存储 token 1~192 的 KV（每层），数据已写入
  Block 12：存储 token 193~200 的 KV（每层 8/16 slot 有效），数据已写入

CPU 内存：
  CPU block 0~5：已分配（LRU manager 知道这些位置），但数据尚未写入
  （manager 中 is_ready=False，状态为 pending）

Connector 状态：
  _unsubmitted_store_jobs = [(job_id=0, TransferSpec(GPU[0..11]→CPU[0..5]))]
  _next_stored_block_idx["A"] = 6
```

#### Step 2 结束时的状态总结
```
GPU 显存：
  Block 0~11：仍保留（decode 需要读这些 KV 做 attention），数据未变
  Block 12：第 9 个 slot 已写入新 token 的 KV

CPU 内存（如果 DMA 完成）：
  CPU block 0~5：已存入 token 1~192 的全部层 KV
  （is_ready 还是 False，因为 complete_store 未被调用）
```

  → 等到请求 A 结束后触发 finished_sending → complete_store → is_ready=True


**卸载的时候会检查block的唯一性，只保存不存在的KVcache；**






# 模块设计
采用 抽象-实现-工厂 的模式
层1: 抽象接口层 distributed/kv_transfer/kv_connector/v1/base.py
层2: 工厂层 kv_connector/factory.py
层3: 具体实现层 kv_connector/v1/offloading_connector.py
层4: Offloading 专属子系统 v1/kv_offload/
        Connector 实现的内部引擎，仅被 offloading_connector.py 使用

①  import factory.py
    → register_connector("OffloadingConnector", ...) 写入注册表
    → offloading_connector.py 此时未被加载
在vllm启动时会加载scheduler，scheduler内自动 import factory.py；

②  Scheduler.__init__()
    → create_connector(role=SCHEDULER)
      → importlib 首次加载 offloading_connector.py
      → OffloadingConnector.__init__ → CPUOffloadingSpec（计算块数）
      → OffloadingConnectorScheduler → CPUBackend + LRUOffloadingManager

③  Worker.initialize_from_config()  （KV cache 内存分配完成后）
    → ensure_kv_transfer_initialized()
    → create_connector(role=WORKER)
      → OffloadingConnector.__init__ → CPUOffloadingSpec（计算块数）
      → OffloadingConnectorWorker → OffloadingWorker（空路由表）

④  gpu_model_runner.initialize_kv_cache()
    → get_kv_connector() → ActiveKVConnector
    → register_kv_caches()
      → CpuGpuOffloadingHandlers 创建
      → 分配 pin memory、初始化 CUDA stream 池  ← 真正的 GPU/CPU 资源在此分配


