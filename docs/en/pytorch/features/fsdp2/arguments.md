# Full Parameter Reference

## Model Arguments (ModelArguments)

Contains parameters related to model and tokenizer loading and initialization.

<table>
  <thead>
    <tr>
      <th>Parameter name</th>
      <th>Type</th>
      <th>Default</th>
      <th>Details</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>model_name_or_path</td>
      <td>str</td>
      <td>None (required)</td>
      <td>The local path of the model. This field is required, and the system raises an exception if you do not specify it.</td>
    </tr>
    <tr>
      <td>model_id</td>
      <td>Optional[Literal[&quot;gpt_oss&quot;, &quot;qwen3&quot;, &quot;qwen3_moe&quot;, &quot;qwen3_next&quot;, &quot;step35&quot;, &quot;mamba3&quot;, &quot;minimax_m27&quot;]]</td>
      <td>None</td>
      <td>Model type identifier. If you do not configure it, the system runs the native Transformer model forward pass. If you configure it, the system runs the repository's custom model forward pass. To add a new model type, register it in the <code>ModelRegistry</code> class in <code>mindspeed_llm/fsdp2/models/model_registry.py</code>.</td>
    </tr>
    <tr>
      <td>init_model_with_meta_device</td>
      <td>bool</td>
      <td>False</td>
      <td>Whether to initialize the model with the meta device. When you enable it, you can reduce the GPU memory used during model initialization.</td>
    </tr>
    <tr>
      <td>trust_remote_code</td>
      <td>bool</td>
      <td>False</td>
      <td>Whether to allow loading models from custom modeling files on Hugging Face for custom model architectures.</td>
    </tr>
    <tr>
      <td>train_from_scratch</td>
      <td>bool</td>
      <td>False</td>
      <td>Whether to train the model from scratch with random weights without loading pretrained weights.</td>
    </tr>
    <tr>
      <td>tokenizer_name_or_path</td>
      <td>Optional[str]</td>
      <td>None</td>
      <td>The path or name of the Tokenizer. Specify it when it differs from <code>model_name_or_path</code>.</td>
    </tr>
    <tr>
      <td>cache_dir</td>
      <td>Optional[str]</td>
      <td>None</td>
      <td>The model cache directory, used to store models downloaded from Hugging Face and ModelScope.</td>
    </tr>
    <tr>
      <td>use_fast_tokenizer</td>
      <td>bool</td>
      <td>True</td>
      <td>Whether to use the fast Tokenizer implementation from the tokenizers library. It tokenizes faster than the traditional Tokenizer.</td>
    </tr>
    <tr>
      <td>resize_vocab</td>
      <td>bool</td>
      <td>False</td>
      <td>Whether to resize the Tokenizer and the corresponding output_layer/lm_head dimensions of the model for vocabulary expansion after adding tokens.</td>
    </tr>
    <tr>
      <td>split_special_tokens</td>
      <td>bool</td>
      <td>False</td>
      <td>Whether to split special tokens during tokenization. By default, the system does not split them and keeps special tokens intact.</td>
    </tr>
    <tr>
      <td>add_tokens</td>
      <td>Optional[str]</td>
      <td>None</td>
      <td>Non-special tokens to add to the Tokenizer. Separate multiple tokens with commas.</td>
    </tr>
    <tr>
      <td>add_special_tokens</td>
      <td>Optional[str]</td>
      <td>None</td>
      <td>Special tokens to add to the Tokenizer. Separate multiple tokens with commas.</td>
    </tr>
    <tr>
      <td>new_special_tokens_config</td>
      <td>Optional[str]</td>
      <td>None</td>
      <td>The path to the YAML configuration file for semantic initialization of special tokens. The format is <code>{&#39;&lt;token&gt;&#39;: &#39;description text&#39;}</code>. This field takes precedence over <code>add_special_tokens</code>.</td>
    </tr>
    <tr>
      <td>init_special_tokens</td>
      <td>Literal[&quot;noise_init&quot;, &quot;desc_init&quot;, &quot;desc_init_w_noise&quot;]</td>
      <td>noise_init</td>
      <td>The initialization method for newly added special tokens. 1. noise_init (default): initialize with random noise based on the mean value. 2. desc_init: initialize from the semantics described in <code>new_special_tokens_config</code> and requires that parameter. 3. desc_init_w_noise: semantic initialization plus random noise.</td>
    </tr>
    <tr>
      <td>model_revision</td>
      <td>str</td>
      <td>main</td>
      <td>The model version to use. You can specify a branch name, tag name, or commit ID.</td>
    </tr>
    <tr>
      <td>hf_hub_token</td>
      <td>Optional[str]</td>
      <td>None</td>
      <td>The authentication token for Hugging Face Hub, suitable for downloading or uploading models from Hugging Face.</td>
    </tr>
    <tr>
      <td>ms_hub_token</td>
      <td>Optional[str]</td>
      <td>None</td>
      <td>The authentication token for ModelScope Hub, suitable for downloading or uploading models from ModelScope.</td>
    </tr>
    <tr>
      <td>om_hub_token</td>
      <td>Optional[str]</td>
      <td>None</td>
      <td>The authentication token for Modelers Hub, suitable for downloading or uploading models from Modelers.</td>
    </tr>
    <tr>
      <td>quant_recipe_name</td>
      <td>Literal[&quot;mxfp8&quot;]</td>
      <td>None</td>
      <td>The quantization strategy.</td>
    </tr>
    <tr>
      <td>quant_apply_modules</td>
      <td>List[str]</td>
      <td>['model.layers.{*}']</td>
      <td>The layers or modules to which quantization applies.</td>
    </tr>
    <tr>
      <td>quant_ignored_modules</td>
      <td>List[str]</td>
      <td>['*lm_head', '*gate']</td>
      <td>The list of submodules that do not use quantization.</td>
    </tr>
    <tr>
      <td>quant_converters</td>
      <td>List[str]</td>
      <td>["quantize.linear.mx"]</td>
      <td>The list of quantization converters to use.</td>
    </tr>
    <tr>
      <td>enable_fsdp_low_precision_all_gather</td>
      <td>bool</td>
      <td>True</td>
      <td>Whether to enable low-precision communication.</td>
    </tr>
    <tr>
      <td>fsdp_low_precision_all_gather_mode</td>
      <td>Literal["on-demand", "all"]</td>
      <td>on-demand</td>
      <td>FSDP low-precision all-gather, which aggregates forward or backward weights on demand.</td>
    </tr>
  </tbody>
</table>

## Data Arguments (DataArguments)

Contains parameters related to dataset loading, preprocessing, and data formats.

<table>
  <thead>
    <tr>
      <th>Parameter name</th>
      <th>Type</th>
      <th>Default</th>
      <th>Details</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>template</td>
      <td>Optional[str]</td>
      <td>None</td>
      <td>The template name used to build prompts. `None` means the system parses the template from the tokenizer. To add a new template type, register it in <code>mindspeed_llm/fsdp2/data/template.py</code>. It currently supports gpt and qwen3.</td>
    </tr>
    <tr>
      <td>dataset</td>
      <td>Optional[Union[Dict[str, Any], str]]</td>
      <td>None</td>
      <td>Training dataset. For fine-tuning scenarios, dataset configuration supports inline configuration and registration through <code>dataset_info.json</code>. For pretraining scenarios, enter the raw dataset path directly. For detailed configuration examples, see <a href="../../training/finetune/fsdp2/finetune_fsdp2.md">FSDP2 backend training guide</a>. A simple example is shown here:<pre style="text-align: left;"><code>dataset:<br>&nbsp;&nbsp;file_name: "./my_data.json"   # Data file path.<br>&nbsp;&nbsp;formatting: "alpaca"          # Data format.</code></pre></td>
    </tr>
    <tr>
      <td>eval_dataset</td>
      <td>Optional[Union[Dict[str, Any], str]]</td>
      <td>None</td>
      <td>Evaluation dataset. Use the same format as <code>dataset</code>. This field is mutually exclusive with <code>val_size</code>. Therefore, you cannot set both at the same time.</td>
    </tr>
    <tr>
      <td>dataset_dir</td>
      <td>str</td>
      <td>./configs/fsdp2/data</td>
      <td>The directory that stores dataset configuration files.</td>
    </tr>
    <tr>
      <td>cutoff_len</td>
      <td>int</td>
      <td>2048</td>
      <td>The truncation length of the input sequence after tokenization. Sequences that exceed this length are truncated.</td>
    </tr>
    <tr>
      <td>train_on_prompt</td>
      <td>bool</td>
      <td>False</td>
      <td>Whether to remove the mask from the prompt part. This field cannot be True at the same time as <code>mask_history</code>.</td>
    </tr>
    <tr>
      <td>mask_history</td>
      <td>bool</td>
      <td>False</td>
      <td>Whether to mask conversation history and train only on the final response. When this field is True, the system does not compute loss for the conversation history. This field cannot be True at the same time as <code>train_on_prompt</code>.</td>
    </tr>
    <tr>
      <td>streaming</td>
      <td>bool</td>
      <td>False</td>
      <td>Whether to enable streaming dataset loading. This field is mutually exclusive with <code>max_samples</code>, and <code>val_size</code> must be an integer.</td>
    </tr>
    <tr>
      <td>buffer_size</td>
      <td>int</td>
      <td>16384</td>
      <td>The size of the random sampling buffer during streaming loading.</td>
    </tr>
    <tr>
      <td>mix_strategy</td>
      <td>Literal[&quot;concat&quot;, &quot;interleave_under&quot;, &quot;interleave_over&quot;]</td>
      <td>concat</td>
      <td>Multi-dataset mixing strategy: concatenation, undersampling, or oversampling.</td>
    </tr>
    <tr>
      <td>interleave_probs</td>
      <td>Optional[str]</td>
      <td>None</td>
      <td>The interleaving sampling probabilities for multiple datasets, separated by commas. This field takes effect only when <code>mix_strategy</code> is interleave_under or interleave_over.</td>
    </tr>
    <tr>
      <td>overwrite_cache</td>
      <td>bool</td>
      <td>False</td>
      <td>Whether to overwrite the cached preprocessed dataset.</td>
    </tr>
    <tr>
      <td>preprocessing_batch_size</td>
      <td>int</td>
      <td>1000</td>
      <td>The number of samples in each batch during data preprocessing.</td>
    </tr>
    <tr>
      <td>preprocessing_num_workers</td>
      <td>Optional[int]</td>
      <td>None</td>
      <td>The number of processes used for data preprocessing.</td>
    </tr>
    <tr>
      <td>max_samples</td>
      <td>Optional[int]</td>
      <td>None</td>
      <td>For debugging. Use it to truncate the number of samples in each dataset. This field is mutually exclusive with <code>streaming</code>.</td>
    </tr>
    <tr>
      <td>eval_num_beams</td>
      <td>Optional[int]</td>
      <td>None</td>
      <td>The number of beams that <code>model.generate</code> uses during evaluation.</td>
    </tr>
    <tr>
      <td>ignore_pad_token_for_loss</td>
      <td>bool</td>
      <td>True</td>
      <td>Whether to ignore tokens that correspond to padding labels when computing loss.</td>
    </tr>
    <tr>
      <td>val_size</td>
      <td>float</td>
      <td>0.0</td>
      <td>The size of the validation set, as an integer or a float between 0 and 1. You must specify <code>dataset</code> as well. This field is mutually exclusive with <code>eval_dataset</code>.</td>
    </tr>
    <tr>
      <td>eval_on_each_dataset</td>
      <td>bool</td>
      <td>False</td>
      <td>Whether to evaluate on each dataset separately.</td>
    </tr>
    <tr>
      <td>packing</td>
      <td>bool</td>
      <td>False</td>
      <td>Whether to enable sequence packing and pack multiple short sequences into one long sequence.</td>
    </tr>
    <tr>
      <td>neat_packing</td>
      <td>bool</td>
      <td>False</td>
      <td>Whether to enable sequence packing without cross attention. When you enable it, the system automatically sets <code>packing=True</code>.</td>
    </tr>
    <tr>
      <td>tool_format</td>
      <td>Optional[str]</td>
      <td>None</td>
      <td>The tool format used to build function call examples. It adapts the document to tool-calling tasks and standardizes function call formatting.</td>
    </tr>
    <tr>
      <td>default_system</td>
      <td>Optional[str]</td>
      <td>None</td>
      <td>Override the default system prompt in the template for a custom system prompt.</td>
    </tr>
    <tr>
      <td>enable_thinking</td>
      <td>Optional[bool]</td>
      <td>True</td>
      <td>Whether to enable thinking mode for inference models. When enabled, the model generates intermediate reasoning steps. True means enabled, False means disabled, and None means the system does not remove the <code>cot</code> tag from the raw data. This is suitable when the raw data contains mixed types.</td>
    </tr>
    <tr>
      <td>tokenized_path</td>
      <td>Optional[str]</td>
      <td>None</td>
      <td>The path for saving or loading a tokenized dataset. If the path does not exist, the system saves the tokenized dataset. If the path exists, the system loads the tokenized dataset.</td>
    </tr>
    <tr>
      <td>data_shared_file_system</td>
      <td>bool</td>
      <td>False</td>
      <td>Whether to store the dataset on a shared file system. This is suitable for distributed training scenarios.</td>
    </tr>
    <tr>
      <td>data_manager_type</td>
      <td>Literal[&quot;lf&quot;, &quot;mg&quot;]</td>
      <td>lf</td>
      <td>The data manager type used to build different dataset managers and adapt different dataset loading and processing logic. lf means fine-tuning data processing, and mg means pretraining.</td>
    </tr>
    <tr>
      <td>split</td>
      <td>str</td>
      <td>100,0,0</td>
      <td>The split ratio for the training set, validation set, and test set, separated by commas.</td>
    </tr>
    <tr>
      <td>create_attention_mask_in_dataloader</td>
      <td>Optional[bool]</td>
      <td>False</td>
      <td>Whether to create the attention mask in the data loader.</td>
    </tr>
    <tr>
      <td>no_shared_storage</td>
      <td>Optional[bool]</td>
      <td>False</td>
      <td>Whether to enable shared storage.</td>
    </tr>
    <tr>
      <td>dataloader_type</td>
      <td>Literal["single"]</td>
      <td>single</td>
      <td>The data loader type. <code>single</code> means sequential reading.</td>
    </tr>
    <tr>
      <td>reset_attention_mask</td>
      <td>Optional[bool]</td>
      <td>False</td>
      <td>For pretraining pack scenarios, when enabled, the system generates <code>actual_seq_len</code> based on the eod position and passes it to the model for training. Enabling <code>reset_attention_mask</code> requires enabling <code>append_eod</code>.</td>
    </tr>
    <tr>
      <td>append_eod</td>
      <td>Optional[bool]</td>
      <td>False</td>
      <td>For pretraining data processing, append the EOD marker to the end of the document.</td>
    </tr>
  </tbody>
</table>

## Parallel Arguments (ParallelArguments)

Contains parameters related to distributed parallel strategies and memory optimization.

<table>
  <thead>
    <tr>
      <th>Parameter name</th>
      <th>Type</th>
      <th>Default</th>
      <th>Details</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>tp_size</td>
      <td>int</td>
      <td>1</td>
      <td>Tensor Parallel size. It splits model tensors across multiple GPUs by column or row.</td>
    </tr>
    <tr>
      <td>fsdp_size</td>
      <td>int</td>
      <td>1</td>
      <td>Fully Sharded Data Parallel (FSDP) size. It shards model parameters across multiple GPUs.</td>
    </tr>
    <tr>
      <td>recompute</td>
      <td>bool</td>
      <td>False</td>
      <td>Whether to enable recomputation. When you enable it, the system saves GPU memory at the cost of some performance.</td>
    </tr>
    <tr>
      <td>ep_size</td>
      <td>int</td>
      <td>1</td>
      <td>Expert Parallel size. This applies to Mixture of Experts (MoE) models and splits different experts across multiple GPUs.</td>
    </tr>
    <tr>
      <td>ep_fsdp_size</td>
      <td>int</td>
      <td>1</td>
      <td>The FSDP size inside the expert parallel group. On top of expert parallelism, it shards the parameters of each expert.</td>
    </tr>
    <tr>
      <td>cp_size</td>
      <td>int</td>
      <td>1</td>
      <td>Context Parallel size. It splits the context of the input sequence across multiple GPUs.</td>
    </tr>
    <tr>
      <td>cp_type</td>
      <td>Literal[&quot;ulysses&quot;,&quot;ring&quot;]</td>
      <td>ulysses</td>
      <td>The algorithm type for Context Parallel. It currently supports only the ulysses and ring algorithms.</td>
    </tr>
    <tr>
      <td>fsdp_modules</td>
      <td>List[str]</td>
      <td>[&quot;model.layers.{}&quot;, &quot;model.embed_tokens&quot;, &quot;lm_head&quot;]</td>
      <td>The model layer structure that enables FSDP. This field is required and cannot be an empty list.</td>
    </tr>
    <tr>
      <td>ignored_modules</td>
      <td>List[str]</td>
      <td>None</td>
      <td>The list of modules that do not enable FSDP.</td>
    </tr>
    <tr>
      <td>reshard_after_forward</td>
      <td>bool</td>
      <td>True</td>
      <td>Whether to reshard the parameters of the FSDP main module after the forward pass.</td>
    </tr>
    <tr>
      <td>shard_placement_fn</td>
      <td>Optional[str]</td>
      <td>None</td>
      <td>A custom shard placement function for the FSDP main module. Use it to customize parameter shard placement logic.</td>
    </tr>
    <tr>
      <td>efsdp_shard_placement_fn</td>
      <td>Optional[str]</td>
      <td>shard_by_dim_1</td>
      <td>The shard placement logic for FSDP modules inside the expert parallel group.</td>
    </tr>
    <tr>
      <td>tp_colwise</td>
      <td>List[str]</td>
      <td>[&quot;*.q_proj&quot;, &quot;*.k_proj&quot;, &quot;*.v_proj&quot;, &quot;*.gate_proj&quot;, &quot;*.up_proj&quot;]</td>
      <td>The model layer structure that uses column-wise tensor parallelism.</td>
    </tr>
    <tr>
      <td>tp_rowwise</td>
      <td>List[str]</td>
      <td>[&quot;*.o_proj&quot;, &quot;*.down_proj&quot;]</td>
      <td>The model layer structure that uses row-wise tensor parallelism.</td>
    </tr>
    <tr>
      <td>ep_modules</td>
      <td>List[str]</td>
      <td>[&quot;model.layers.{*}.mlp.experts&quot;]</td>
      <td>The model layer structure that enables expert parallelism. This applies only to MoE models.</td>
    </tr>
    <tr>
      <td>ep_fsdp_modules</td>
      <td>List[str]</td>
      <td>[&quot;model.layers.{*}.mlp.experts&quot;]</td>
      <td>The model layer structure that enables FSDP inside the expert parallel group.</td>
    </tr>
    <tr>
      <td>ep_dispatcher</td>
      <td>Literal[&quot;eager&quot;, &quot;fused&quot;, &quot;mc2&quot;]</td>
      <td>eager</td>
      <td>The scheduling strategy for MoE expert parallelism.</td>
    </tr>
    <tr>
      <td>recompute_modules</td>
      <td>List[str]</td>
      <td>[&quot;model.layers.{*}&quot;]</td>
      <td>The model layer structure that enables activation recomputation. Use it together with <code>recompute=True</code>.</td>
    </tr>
    <tr>
      <td>param_dtype</td>
      <td>Literal[&quot;bf16&quot;, &quot;fp16&quot;, &quot;fp32&quot;]</td>
      <td>bf16</td>
      <td>The data type used to store FSDP parameters.</td>
    </tr>
    <tr>
      <td>reduce_dtype</td>
      <td>Literal[&quot;bf16&quot;, &quot;fp16&quot;, &quot;fp32&quot;]</td>
      <td>fp32</td>
      <td>The data type used for FSDP gradient reduction. The default is fp32, which ensures numerical stability during gradient reduction and helps avoid gradient vanishing or explosion.</td>
    </tr>
    <tr>
      <td>num_to_forward_prefetch</td>
      <td>int</td>
      <td>1</td>
      <td>The number of modules prefetched during the FSDP forward pass. This helps optimize pipeline efficiency.</td>
    </tr>
    <tr>
      <td>num_to_backward_prefetch</td>
      <td>int</td>
      <td>1</td>
      <td>The number of modules prefetched during the FSDP backward pass. This helps optimize pipeline efficiency.</td>
    </tr>
  </tbody>
</table>

## Training Arguments (TrainingArguments)

Contains parameters related to training hyperparameters, optimizers, model saving, and logging.

<table>
  <thead>
    <tr>
      <th>Parameter name</th>
      <th>Type</th>
      <th>Default</th>
      <th>Details</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>output_dir</td>
      <td>str</td>
      <td>None (required)</td>
      <td>The output directory for training results, used to save model weights, logs, prediction results, and more. This field is required, and the system raises an exception if you do not specify it.</td>
    </tr>
    <tr>
      <td>optimizer</td>
      <td>Literal["adamw", "muon"]</td>
      <td>adamw</td>
      <td>The optimizer. It currently supports AdamW and Muon.</td>
    </tr>
    <tr>
      <td>lr</td>
      <td>float</td>
      <td>1e-5</td>
      <td>The initial learning rate.</td>
    </tr>
    <tr>
      <td>weight_decay</td>
      <td>float</td>
      <td>0.01</td>
      <td>The weight decay coefficient.</td>
    </tr>
    <tr>
      <td>adam_beta1</td>
      <td>float</td>
      <td>0.9</td>
      <td>The beta1 parameter of the AdamW optimizer. It controls the exponential decay rate of the first-order momentum.</td>
    </tr>
    <tr>
      <td>adam_beta2</td>
      <td>float</td>
      <td>0.95</td>
      <td>The beta2 parameter of the AdamW optimizer. It controls the exponential decay rate of the second-order momentum.</td>
    </tr>
    <tr>
      <td>adam_epsilon</td>
      <td>float</td>
      <td>1e-8</td>
      <td>The epsilon parameter of the AdamW optimizer.</td>
    </tr>
    <tr>
      <td>max_grad_norm</td>
      <td>float</td>
      <td>1.0</td>
      <td>The maximum norm for gradient clipping, which helps prevent gradient explosion.</td>
    </tr>
    <tr>
      <td>lr_scheduler_type</td>
      <td>Literal[&quot;cosine&quot;, &quot;linear&quot;, &quot;constant&quot;]</td>
      <td>cosine</td>
      <td>The learning rate scheduler type: cosine annealing, linear decay, or constant learning rate.</td>
    </tr>
    <tr>
      <td>warmup_ratio</td>
      <td>float</td>
      <td>0.03</td>
      <td>The proportion of total training steps used for linear warmup.</td>
    </tr>
    <tr>
      <td>min_lr</td>
      <td>float</td>
      <td>1e-6</td>
      <td>The minimum learning rate. This field takes effect only when you use the cosine scheduler.</td>
    </tr>
    <tr>
      <td>num_train_epochs</td>
      <td>float</td>
      <td>3.0</td>
      <td>The total number of training epochs. If <code>max_steps&gt;0</code>, that field overrides this one.</td>
    </tr>
    <tr>
      <td>max_steps</td>
      <td>int</td>
      <td>-1</td>
      <td>The total number of training steps. When it is greater than 0, it overrides <code>num_train_epochs</code>.</td>
    </tr>
    <tr>
      <td>gradient_accumulation_steps</td>
      <td>int</td>
      <td>1</td>
      <td>The number of gradient accumulation steps. The system accumulates gradients from multiple batches before it runs backpropagation and parameter updates.</td>
    </tr>
    <tr>
      <td>disable_shuffling</td>
      <td>bool</td>
      <td>False</td>
      <td>Whether to disable shuffling for the training set.</td>
    </tr>
    <tr>
      <td>seed</td>
      <td>int</td>
      <td>42</td>
      <td>The random seed set at the start of training.</td>
    </tr>
    <tr>
      <td>save_steps</td>
      <td>int</td>
      <td>500</td>
      <td>Save model weights every X steps.</td>
    </tr>
    <tr>
      <td>save_total_limit</td>
      <td>Optional[int]</td>
      <td>3</td>
      <td>The limit on the total number of weight checkpoints. If the count exceeds this number, the system deletes the oldest checkpoint.</td>
    </tr>
    <tr>
      <td>resume_from_checkpoint</td>
      <td>Optional[str]</td>
      <td>None</td>
      <td>The checkpoint path used to resume training.</td>
    </tr>
    <tr>
      <td>logging_steps</td>
      <td>int</td>
      <td>1</td>
      <td>Record training logs every X steps.</td>
    </tr>
    <tr>
      <td>stage</td>
      <td>Literal[&quot;pt&quot;, &quot;sft&quot;]</td>
      <td>sft</td>
      <td>Training stage: pretraining or instruction fine-tuning.</td>
    </tr>
    <tr>
      <td>calculate_per_token_loss</td>
      <td>bool</td>
      <td>False</td>
      <td>Whether to scale cross-entropy loss by the number of non-padding tokens in the global batch.</td>
    </tr>
    <tr>
      <td>dataloader_num_workers</td>
      <td>int</td>
      <td>0</td>
      <td>The number of data loading subprocesses. 0 means the main process loads data.</td>
    </tr>
    <tr>
      <td>dataloader_prefetch_factor</td>
      <td>Optional[int]</td>
      <td>None</td>
      <td>The number of batches prefetched by each data loading subprocess.</td>
    </tr>
    <tr>
      <td>dataloader_pin_memory</td>
      <td>bool</td>
      <td>True</td>
      <td>Whether to enable pinned memory for the data loader.</td>
    </tr>
    <tr>
      <td>dataloader_persistent_workers</td>
      <td>bool</td>
      <td>False</td>
      <td>Whether to keep data loading subprocesses alive.</td>
    </tr>
    <tr>
      <td>dataloader_drop_last</td>
      <td>bool</td>
      <td>False</td>
      <td>Whether to drop the last incomplete batch when the dataset size is not divisible by the batch size.</td>
    </tr>
    <tr>
      <td>per_device_train_batch_size</td>
      <td>int</td>
      <td>8</td>
      <td>The training batch size per GPU.</td>
    </tr>
    <tr>
      <td>save_only_model</td>
      <td>bool</td>
      <td>False</td>
      <td>When saving checkpoints, whether to save only model parameters and not optimizer state, scheduler state, or random number state.</td>
    </tr>
    <tr>
      <td>save_async</td>
      <td>bool</td>
      <td>False</td>
      <td>Whether to save checkpoints asynchronously.</td>
    </tr>
    <tr>
      <td>save_epochs</td>
      <td>int</td>
      <td>1</td>
      <td>The interval, in epochs, for saving checkpoints.</td>
    </tr>
    <tr>
      <td>save_hf_weights</td>
      <td>bool</td>
      <td>True</td>
      <td>Whether to save Hugging Face format weights after training finishes.</td>
    </tr>
  </tbody>
</table>

## Profiling Parameters

Contains parameters related to performance profiling data collection.

<table>
  <thead>
    <tr>
      <th>Parameter name</th>
      <th>Type</th>
      <th>Default</th>
      <th>Details</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>profile</td>
      <td>bool</td>
      <td>False</td>
      <td>Whether to enable profiling data collection.</td>
    </tr>
    <tr>
      <td>profile_step_start</td>
      <td>int</td>
      <td>0</td>
      <td>The step at which to start collecting data, inclusive. This value must be greater than or equal to 0.</td>
    </tr>
    <tr>
      <td>profile_step_end</td>
      <td>int</td>
      <td>-1</td>
      <td>The step at which to stop collecting data, exclusive. -1 means collect data until training ends.</td>
    </tr>
    <tr>
      <td>profile_ranks</td>
      <td>List[int]</td>
      <td>[-1]</td>
      <td>The GPU ranks from which to collect data. -1 means collect profiling data from all ranks.</td>
    </tr>
    <tr>
      <td>profile_level</td>
      <td>str</td>
      <td>level0</td>
      <td>The profiling level: level_none, level0, level1, or level2. Higher levels collect more information.</td>
    </tr>
    <tr>
      <td>profile_export_type</td>
      <td>str</td>
      <td>text</td>
      <td>The export type for profiling results: text format or database format.</td>
    </tr>
    <tr>
      <td>profile_data_simplification</td>
      <td>bool</td>
      <td>False</td>
      <td>Whether to enable data simplification mode for profiling.</td>
    </tr>
    <tr>
      <td>profile_with_cpu</td>
      <td>bool</td>
      <td>False</td>
      <td>Whether to collect CPU data during profiling.</td>
    </tr>
    <tr>
      <td>profile_with_stack</td>
      <td>bool</td>
      <td>False</td>
      <td>Whether to collect call stack information during profiling.</td>
    </tr>
    <tr>
      <td>profile_with_memory</td>
      <td>bool</td>
      <td>False</td>
      <td>Whether to collect memory allocation and usage during profiling.</td>
    </tr>
    <tr>
      <td>profile_record_shapes</td>
      <td>bool</td>
      <td>False</td>
      <td>Whether to collect tensor shape information during profiling.</td>
    </tr>
    <tr>
      <td>profile_save_path</td>
      <td>str</td>
      <td>./profile</td>
      <td>The save path for profiling data collection.</td>
    </tr>
  </tbody>
</table>

## Inference Arguments (InferenceArguments)

Contains inference-related parameters.

<table>
<thead>
<tr>
<th>Parameter name</th>
<th>Type</th>
<th>Default</th>
<th>Details</th>
</tr>
</thead>
<tbody>
<tr>
<td>infer_backend</td>
<td>Literal["huggingface"]</td>
<td>"huggingface"</td>
<td>Specifies the inference engine backend to use. It currently supports only Hugging Face.</td>
</tr>
<tr>
<td>max_new_tokens</td>
<td>int</td>
<td>512</td>
<td>The maximum number of tokens to generate. This limit applies only to the generated response and does not include tokens in the input prompt itself.</td>
</tr>
<tr>
<td>do_sample</td>
<td>bool</td>
<td>False</td>
<td>Whether to use random sampling for generation. If you set this field to False, the system uses Greedy Decoding by default. In FSDP multi-GPU inference, to avoid deadlocks caused by inconsistent output across GPUs during sampling, you are usually advised to keep the default False.</td>
</tr>
</tbody>
</table>

## Optimization Feature Parameters (OptimizationArguments)

Contains parameters related to fused operator enablement, memory optimization features, and performance optimization features.

<table>
  <thead>
    <tr>
      <th>Parameter name</th>
      <th>Type</th>
      <th>Default</th>
      <th>Details</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>use_fused_rmsnorm</td>
      <td>bool</td>
      <td>False</td>
      <td>Whether to enable the fused rmsnorm operator.</td>
    </tr>
    <tr>
      <td>moe_grouped_gemm</td>
      <td>bool</td>
      <td>False</td>
      <td>Whether to enable the fused GMM operator.</td>
    </tr>
    <tr>
      <td>use_fused_rotary_pos_emb</td>
      <td>bool</td>
      <td>False</td>
      <td>Whether to enable the fused RoPE operator.</td>
    </tr>
    <tr>
      <td>use_flash_attn</td>
      <td>bool</td>
      <td>False</td>
      <td>Whether to enable the FA operator.</td>
    </tr>
    <tr>
      <td>use_triton_gdn</td>
      <td>bool</td>
      <td>False</td>
      <td>Whether to enable the Triton fused operator to accelerate Gated DeltaNet network computation.</td>
    </tr>
    <tr>
      <td>use_flash_gdn</td>
      <td>bool</td>
      <td>False</td>
      <td>Whether to enable the AscendC fused operator to accelerate Gated DeltaNet network computation. At most one of this parameter and <code>use_triton_gdn</code> can be enabled.</td>
    </tr>
    <tr>
    <td>chunk_loss_size</td>
    <td>int</td>
    <td>None</td>
    <td>The sequence length used each time loss is computed. Chunked loss computation can save memory.</td>
    </tr>
    <tr>
    <td>gdn_chunk_size</td>
    <td>int</td>
    <td>64</td>
    <td>The number of matrix chunks used during Gated DeltaNet network computation.</td>
    </tr>
    <tr>
    <td>use_triton_rmsnormgated</td>
    <td>bool</td>
    <td>False</td>
    <td>Whether to enable the Triton fused operator to accelerate RMSNorm_gated computation.</td>
    </tr>
    <tr>
    <td>fix_router</td>
    <td>bool</td>
    <td>False</td>
    <td>Fix expert assignment to balance load. Use this for performance tuning only.</td>
    </tr>
  </tbody>
</table>

# Megatron Parameter Mapping

<table>
  <thead>
    <tr>
      <th>Parameter type</th>
      <th>FSDP2 parameter</th>
      <th>Megatron parameter</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2">ModelArguments</td>
      <td>model_name_or_path</td>
      <td>--load</td>
    </tr>
    <tr>
      <td>tokenizer_name_or_path</td>
      <td>--tokenizer-name-or-path</td>
    </tr>
    <tr>
      <td rowspan="4">DataArguments</td>
      <td>dataset</td>
      <td>--data-path</td>
    </tr>
    <tr>
      <td>cutoff_len</td>
      <td>--seq-length</td>
    </tr>
    <tr>
      <td>preprocessing_num_workers</td>
      <td>--workers</td>
    </tr>
    <tr>
      <td>packing</td>
      <td>--pack</td>
    </tr>
    <tr>
      <td rowspan="5">ParallelArguments</td>
      <td>tp_size</td>
      <td>--tensor-model-parallel-size</td>
    </tr>
    <tr>
      <td>ep_size</td>
      <td>--expert-model-parallel-size</td>
    </tr>
    <tr>
      <td>cp_size</td>
      <td>--context-parallel-size</td>
    </tr>
    <tr>
      <td>cp_type</td>
      <td>--context-parallel-algo</td>
    </tr>
    <tr>
      <td>recompute</td>
      <td>--recompute-granularity</td>
    </tr>
    <tr>
      <td rowspan="9">TrainingArguments</td>
      <td>per_device_train_batch_size</td>
      <td>--micro-batch-size</td>
    </tr>
    <tr>
      <td>disable_shuffling</td>
      <td>--no-shuffle</td>
    </tr>
    <tr>
      <td>lr_scheduler_type</td>
      <td>--lr-decay-style</td>
    </tr>
    <tr>
      <td>warmup_ratio</td>
      <td>--lr-warmup-fraction</td>
    </tr>
    <tr>
      <td>recompute</td>
      <td>--recompute-method</td>
    </tr>
    <tr>
      <td>max_steps</td>
      <td>--train-iters</td>
    </tr>
    <tr>
      <td>save_steps</td>
      <td>--save-interval</td>
    </tr>
    <tr>
      <td>logging_steps</td>
      <td>--log-interval</td>
    </tr>
    <tr>
      <td>Number of GPUs * per_device_train_batch_size * gradient_accumulation_steps</td>
      <td>--global-batch-size</td>
    </tr>
  </tbody>
</table>
