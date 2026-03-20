以 DeepSeek-V3/V2 的具体参数为基准（N=256 个专家，Top-K=8，每个 expert 的 intermediate 维度为 2048，shared experts 另计）

**第一步：Gate 打分（对每个token进行路由决策）**
每个 token 的 hidden state 先经过一个轻量线性层，得到对所有专家的亲和度分数。
Gate 层做的事情（只有一个下采样）：
        [T, 7168] × [7168, 256] → [T, 256]，得到每个 token 对 256 个专家的亲和度分数，然后 Softmax + Top-K 只挑最高的 8 个，并把这 8 个分数重新归一化作为后续加权用的 g_i。
这里注意是把8个expert的权重进行softmax；

**第二步：每个选中的 Expert 内部的 FFN 计算**
每个 expert 结构上就是一个标准 SwiGLU FFN，但参数完全独立、互不共享。

每个 expert 内部的三个矩阵：

W1（gate proj）：[7168, 2048]，做门控分支
W3（up proj）：[7168, 2048]，做值分支
W2（down proj）：[2048, 7168]，投回原始维度

SwiGLU 的计算是 SiLU(x·W1) ⊙ (x·W3)，然后再乘 W2，单个 expert 的输出维度回到 [1, 7168]。关键在于：256 个 expert 参数完全独立，但只有被路由到的 8 个会真正执行计算。

这里需要注意的是SiLU(x·W1) 和x·W3两个计算完毕之后，是做点乘；

**第三步：加权汇聚回 hidden states**
选中的 8 个 expert 各自输出一个 [T, 7168]，最后按门控权重加权求和。

<img width="686" height="364" alt="Image" src="https://github.com/user-attachments/assets/3d503832-37c8-4f24-b3ac-c5be648e9eb7" />

**注意**
用每个expert的权重乘每个专家的输出值，然后每个expert相加==》最后的输出
这里的权重参数来自第一步的softmax的输出
DeepSeek-V2/V3 中有 2 个 shared expert，加上topk=8的expert，一共是10个expert