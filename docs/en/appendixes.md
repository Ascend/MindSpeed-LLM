# Appendix

## FAQ

- **Question 1**

  Q: The training log shows "Checkpoint path not found"?
  A: Check whether `CKPT_LOAD_DIR` points to the correct path after weight conversion, and confirm that the folder contains `.ckpt` or `.bin` files. Otherwise, correct the weight path setting.

- **Question 2**

  Q: The dataset loading shows "out of range"?
  A: The fine-tuning script could not read the dataset. Check whether `DATA_PATH` in the script follows the sample format.

  ![img_3.png](./pytorch/figures/quick_start/img_3.png)

- **Question 3**

  Q: No runtime log file is generated?
  A: You need to create the `logs` folder yourself.

  ![img_1.png](./pytorch/figures/quick_start/img_1.png)

## Join the Ascend Developer Ecosystem

- 🌐 **Community resources**: Visit the [Ascend Open Source Community](https://gitcode.com/ascend) to get the latest model support.
- 📈 **Performance optimization**: Refer to [Profiling Data Collection](./pytorch/tools/profiling.md) to analyze bottlenecks.
- 💡 **Customization needs**: Extend custom models through `model_cfg.json`.

## Linearity

Based on the dense `GPT3-175B` LLM, we scaled the MFU and linearity experiments from 128 NPUs to 7,968 NPUs. The experimental data is shown below.

<p align="center"> <img src="https://raw.gitcode.com/Ascend/MindSpeed-LLM/raw/26.0.0/docs/en/pytorch/figures/readme/linearity&mfu.png" height="490px" width="715px"> </p>

The figure shows the `MFU` values and the overall `linearity` of the cluster at the corresponding scale.
