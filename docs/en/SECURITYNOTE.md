# Security Statement

## System Security Hardening

1. You are advised to enable ASLR at level 2 on the system. It is also known as full randomization of address space layout. You can configure it as follows.

    ```shell
    echo 2 > /proc/sys/kernel/randomize_va_space
    ```

## Recommended Runtime User

For security and least-privilege reasons, you are advised not to use `root` or other administrator accounts to run MindSpeed LLM.

## File Permission Control

1. You are advised to set the runtime system `umask` value to `0027` or higher on the host and in containers. This ensures that newly created directories have a maximum default permission of `750` and newly created files have a maximum default permission of `640`.
2. You are advised to apply permission controls to sensitive content such as personal data, business assets, source files, and various files saved during training. Relevant scenarios include permission control for the MindSpeed LLM installation directory and permission control for shared datasets used by multiple users. You can refer to Table 1 for the recommended permission settings.
3. MindSpeed LLM generates training data during data preprocessing and generates weight files during training. The default file permission is `640`. You can tighten the permissions on generated files according to actual needs.
4. When you train with a non-`root` user, the permissions on your `CKPT_SAVE_DIR` path may be too restrictive, which prevents access to that directory. You can add `chmod -R 660 $CKPT_SAVE_DIR` to the script to modify the access permissions on that directory and ensure that the model weight files can be read and written normally.

**Table 1** Recommended maximum permission limits for files and folders in different scenarios

| Type | Linux permission reference maximum |
| --------------- | --------------------|
| User home directory | 750 (`rwxr-x---`) |
| Program files, including scripts and library files | 550 (`r-xr-x---`) |
| Program file directory | 550 (`r-xr-x---`) |
| Configuration file | 640 (`rw-r-----`) |
| Configuration file directory | 750 (`rwxr-x---`) |
| Log files, after they complete or are archived | 440 (`r--r-----`) |
| Log files, while they are being written | 640 (`rw-r-----`) |
| Log file record directory | 750 (`rwxr-x---`) |
| Debug files | 640 (`rw-r-----`) |
| Debug file directory | 750 (`rwxr-x---`) |
| Temporary file directory | 750 (`rwxr-x---`) |
| Maintenance and upgrade file directory | 770 (`rwxrwx---`) |
| Service data files | 640 (`rw-r-----`) |
| Service data file directory | 750 (`rwxr-x---`) |
| Directory for key components, private keys, certificates, and ciphertext files | 700 (`rwx------`) |
| Key components, private keys, certificates, and encrypted ciphertext | 600 (`rw-------`) |
| Encryption and decryption interfaces and scripts | 500 (`r-x------`) |

## Data Security Statement

1. MindSpeed LLM stores model files in the checkpointing module of Megatron, and some model files use the potentially risky `pickle` module, which may pose data security risks.
2. During program execution, `nltk.download` loads corpora from the path specified by the user. Therefore, ensure network security and that the source of the downloaded package is trusted.

## Runtime Security Statement

1. You are advised to write training scripts that match the available resources. If the training script does not match the resource conditions, for example, if the dataset loading memory exceeds the available memory or if the training script generates more data locally than the available disk space, errors may occur and the process may exit unexpectedly.
2. MindSpeed LLM uses PyTorch internally, and version mismatches may cause runtime errors. For details, see the PyTorch [Security Statement](https://gitcode.com/ascend/pytorch#%E5%AE%89%E5%85%A8%E5%A3%B0%E6%98%8E).
3. This software uses `torch.load` from PyTorch to load models, and the code uses this interface with `weights_only=True`. For PyTorch versions `<= 2.5.1`, a deserialization vulnerability, CVE-2025-32434, exists. Please ensure the safety of the loaded weights and avoid malicious model loading that could attack the execution machine or device.

## Public Internet Address Statement

| Type | Open-source code URL | File name | Public IP address/public URL/domain/email address | Description |
| ---- | ------------ | ----------------------------------------------------------- | ------------------------------------------------------------ | ------------ |
| In-house | Not involved | `mindspeed_llm/model/language_model.py:85` | <https://github.com/kingoflolz/mesh-transformer-jax/> | Details URL |
| In-house | Involved | `tests/test_tools/dist_test.py:6` | <https://github.com/microsoft/DeepSpeed/blob/master/tests/unit/common.py> | Source code URL |
| In-house | Involved | `tests/pipeline/conftest.py:6` | <https://github.com/microsoft/DeepSpeed/blob/master/tests/conftest.py> | Source code URL |
| In-house | Involved | `tests/ut/conftest.py:6` | <https://github.com/microsoft/DeepSpeed/blob/master/tests/conftest.py> | Source code URL |
| In-house | Not involved | `examples/mcore/gemma/data_convert_gemma_pretrain.sh:5` | <https://huggingface.co/datasets/pleisto/wikipedia-cn-20230720-filtered/resolve/main/wikipedia-cn-20230720-filtered.json?download=true> | Data download URL |
| In-house | Not involved | `mindspeed_llm/core/transformer/moe/moe_utils.py:135` | <https://arxiv.org/abs/2101.03961> | Paper URL |
| In-house | Involved | `mindspeed_llm/tasks/data/collator.py:4` | <https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/src/axolotl/monkeypatch/utils.py> | Source code URL |
| In-house | Involved | `mindspeed_llm/core/distributed/distributed_data_parallel.py:126` | <https://github.com/NVIDIA/TransformerEngine/pull/719> | Source code URL |
| In-house | Not involved | `mindspeed_llm/core/datasets/gpt_dataset.py:159, 219` | <https://gitcode.com/ascend/MindSpeed-LLM/wiki/megatron%20data%20helpers%E5%8F%AF%E8%83%BD%E5%BC%95%E5%85%A5%E7%9A%84%E9%97%AE%E9%A2%98> | Details URL |

## Public Interface Statement

MindSpeed LLM has not yet released a wheel package. Therefore, it does not provide any formal public interface. All functionality is invoked through shell scripts. The five entry scripts are [pretrain_gpt.py](https://gitcode.com/ascend/MindSpeed-LLM/blob/26.0.0/pretrain_gpt.py), [inference.py](https://gitcode.com/ascend/MindSpeed-LLM/blob/26.0.0/inference.py), [evaluation.py](https://gitcode.com/ascend/MindSpeed-LLM/blob/26.0.0/evaluation.py), [preprocess_data.py](https://gitcode.com/ascend/MindSpeed-LLM/blob/26.0.0/preprocess_data.py), and [convert_ckpt.py](https://gitcode.com/ascend/MindSpeed-LLM/blob/26.0.0/convert_ckpt.py).

## Communication Security Hardening

[Communication security hardening instructions](https://gitcode.com/Ascend/pytorch/blob/v2.7.1-26.0.0/docs/zh/SECURITYNOTE.md#%E9%80%9A%E4%BF%A1%E5%AE%89%E5%85%A8%E5%8A%A0%E5%9B%BA)

## Communication Matrix

[Communication matrix instructions](https://gitcode.com/Ascend/pytorch/blob/v2.7.1-26.0.0/docs/zh/SECURITYNOTE.md#%E9%80%9A%E4%BF%A1%E7%9F%A9%E9%98%B5)

### Special Scenarios

| Scenario | Usage method | Port | Possible risk |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------| ------------------------------------------------ | ---------- | ---------- |
| When you use MindSpeed LLM for training tasks in the Megatron backend, each time it initializes the model-parallel group, it adds `(3 * NPU count)` random ports by default. When multiple distributed optimizers are enabled, it adds `(number of distributed optimizers * NPU count)` more random ports. At the same time, you need to configure one `master-port` port, which is consistent with the `master-port` of `torch_npu`. | MindSpeed LLM calls the native `mpu.initialize_model_parallel` function of Megatron to initialize the model-parallel group and uses PyTorch distributed training APIs to start any task. | Within [1024, 65520] | Incorrect network configuration may cause port conflicts or connection issues and affect training efficiency. |
| The user downloads corpora through `nltk.download`. | The user uses `nltk.download` inside the code to download corpora. | Random port | If the file source is untrusted, a deserialization vulnerability may exist when the file is loaded, which can lead to file tampering. |
