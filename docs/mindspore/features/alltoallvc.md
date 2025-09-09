# AllToAllVC 通信算子

## 背景与挑战

在大规模分布式训练中，模型参数、激活和专家路由数据需要在多个设备之间高效传输。
常见的通信模式包括 **AllToAll** 和 **AllToAllV**：

* **AllToAll**：要求每个 rank 向其他所有 rank 发送等量数据，无法应对实际训练中 **不均匀的 token 分布**。
* **AllToAllV**：支持不均匀的数据分散与聚集，但在 Ascend 硬件上性能有优化空间。

因此，在 MOE（Mixture-of-Experts）等需要 **动态不均匀 token 路由**的场景下，AllToAllV 的通信开销容易成为瓶颈。

## 解决方案

MindSpore **AllToAllVC** 通信算子，支持不均匀通信，并在性能上优于 AllToAllV。

### 解决思路

* **功能特性**
  AllToAllVC 与 AllToAllV 功能一致，均支持 **不均匀分散与聚集**。

* **性能优化**

  * AllToAllV 的实现中，通信过程需要进行 **两次 D2H（Device to Host）拷贝**。
  * AllToAllVC 算子优化为 **只进行一次拷贝**，显著降低了通信开销。
  * 在实际测试中，AllToAllVC 算子的性能优于基于 `torch.distributed.all_to_all_single` 的实现。

---

## 使用方法

（1）开启 AllToAllVC 特性，需要在运行时确保通信组已初始化，并加载 MindSpeed 中的通信优化模块。

（2）在 msrun 启动bash 脚本中，启用参数 **--enable-a2avc** 即可开启使用 AllToAllVC 特性。

（3）需在 CANN 8.3.RC1 及以上版本使用

（4）目前该特性具有以下约束：
  * 仅支持fix-router场景启用。
  * 使用该特性时，不能启用 overlap（即 **--overlap-alltoall** 或 **--moe-fb-overlap** 参数）。

---

## 使用效果

* **性能提升**：相对 AllToAllV，通信性能更优。

---

