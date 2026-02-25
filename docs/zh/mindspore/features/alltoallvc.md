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

  * AllToAllV 的算子实现中，通信过程需要进行 **一次 D2H（Device to Host）拷贝**。
  * AllToAllVC 算子内优化为 **不进行 D2H 拷贝**，显著降低了通信开销。
  * 在实际测试中，AllToAllVC 算子的性能优于基于 `torch.distributed.all_to_all_single` 的实现。

---

## 使用方法

（1）开启 AllToAllVC 特性，需要在运行时确保通信组已初始化，并加载 MindSpeed 中的通信优化模块。

（2）在 msrun 启动的 bash 脚本中，使用参数 **--enable-a2avc [int]** 即可控制是否使用 AllToAllVC 特性以及具体的模式。参数说明：
  * **[int]** 为整数，代表AllToAllVC的模式，可选值为 0，1，或 2。
  * **[int]** = 0：默认值。不启用 AllToAllVC 功能。
  * **[int]** = 1：启用 AllToAllVC，并在运行过程中对传输参数和计算结果执行多层次一致性校验，以便及时发现并中止潜在的非法或不匹配传值。该模式在健壮性上更为严格，但性能开销较高，性能低于模式 2。
  * **[int]** = 2：启用 AllToAllVC，但运行时不进行多层次校验，仅执行必要的通信与计算，性能优于模式 1。

（3）需在 CANN 8.3.RC1 及以上版本使用

（4）目前该特性具有以下约束：
  * 仅支持fix-router场景启用。
  * 使用该特性时，不能启用 overlap（即 **--overlap-alltoall** 或 **--moe-fb-overlap** 参数）。

---

## 使用效果

* **性能提升**： 在使用模式 2 （**--enable-a2avc 2**）时,相对 AllToAllV，通信性能更优。在剪裁 deepseekv3 用例上性能比 AllToAllV整体优化 0.4%。

---

