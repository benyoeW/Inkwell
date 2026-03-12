<img width="2472" height="1054" alt="Image" src="https://github.com/user-attachments/assets/f11149e1-c60e-4d1f-aa8b-b3ce8e5aad9f" />
图片链接：

[https://www.bilibili.com/video/BV1fAPceeE7n/?spm_id_from=333.337.search-card.all.click&vd_source=d4d8e34229b29465d437ca4e527e32c5](url)

# 理论部分：

图中的输入是一个token的embedding向量
对于Q：
将输入经过一个下采样矩阵的到latent ctq，该维度小于正常的Q向量的维度；
然后将Q经过两次上采样，得到Q合并之前的两部分（一部分有RoPE一部分没有），拼接之后得到完整的Q

对于KV：
K也是由两部分拼接而成，一部分带有RoPE，一部分没有；
带有RoPE的：token的embedding向量直接经过矩阵变换和RoPE得到
不带有RoPE的：通过下采样得到latent ctkv，latent ctkv经过一个上采样的到K的第二部分

V直接由latent ctkv经过上采样的到V

# 代码实现：




