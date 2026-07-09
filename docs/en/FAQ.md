# MindSpeed LLM FAQ

- **Question 1**

  Q: Why does the training log show "Checkpoint path not found"?
  A: Check if `CKPT_LOAD_DIR` points to the correct weight conversion path. Confirm that the folder contains `.ckpt` or `.bin` files. Otherwise, correct the weight path setting.

- **Question 2**

  Q: Why does dataset loading show an "out of range" error?
  A: The fine-tuning script fails to load the dataset. Check if `DATA_PATH` in the script conforms to the example specifications.

  ![img_3.png](./pytorch/figures/quick_start/img_3.png)

- **Question 3**

  Q: Why is no runtime log file generated?
  A: You need to create the `logs` folder manually.

  ![img_1.png](./pytorch/figures/quick_start/img_1.png)
