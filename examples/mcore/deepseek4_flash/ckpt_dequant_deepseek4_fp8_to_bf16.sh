pip install torchao

python tests/tools/ckpt_dequant/deepseekv4_ckpt_dequant.py \
  --input_fp8_hf_path ./model_from_hf/deepseek4-hf \
  --output_hf_path ./model_from_hf/deepseek4-hf-bf16 \
  --quant_type bfloat16
