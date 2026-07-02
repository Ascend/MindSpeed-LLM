# Appendix

## FAQ

FAQ, please refer to [FAQ](./FAQ.md).

## Join the Ascend Developer Ecosystem

- 🌐 **Community resources**: Visit the [Ascend Open Source Community](https://gitcode.com/ascend) to get the latest model support.
- 📈 **Performance optimization**: Refer to [Profiling Data Collection](./pytorch/tools/profiling.md) to analyze bottlenecks.
- 💡 **Customization needs**: Extend custom models through `model_cfg.json`.

## Linearity

Based on the dense `GPT3-175B` LLM, we scaled the MFU and linearity experiments from 128 NPUs to 7,968 NPUs. The experimental data is shown below.

<p align="center"> <img src="./pytorch/figures/readme/linearity&mfu.png" height="490px" width="715px"> </p>

The figure shows the `MFU` values and the overall `linearity` of the cluster at the corresponding scale. You can click the following links to refer to the calculation formulas:

- [MFU calculation formula](https://gitcode.com/Ascend/MindSpeed-LLM/wiki/%E6%9C%AF%E8%AF%AD%E5%AE%9A%E4%B9%89%2F%E5%A4%A7%E6%A8%A1%E5%9E%8B%20MFU%20%E8%AE%A1%E7%AE%97%E5%85%AC%E5%BC%8F.md)
- [Linearity calculation formula](https://gitcode.com/Ascend/MindSpeed-LLM/wiki/%E6%9C%AF%E8%AF%AD%E5%AE%9A%E4%B9%89%2F%E7%BA%BF%E6%80%A7%E5%BA%A6%E5%85%AC%E5%BC%8F.md)
