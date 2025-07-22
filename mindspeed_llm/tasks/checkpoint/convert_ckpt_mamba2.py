import copy
import os
import json
import re
import shutil
from collections import OrderedDict
import logging as logger
import argparse
import torch
import safetensors.torch
logger.basicConfig(format="")
logger.getLogger().setLevel(logger.INFO)


class CheckpointConverter:
    def __init__(self, args):
        self.args = args
        self.args.mamba_d_inner = self.args.d_model * 2
        self.args.mamba2_n_heads = self.args.mamba_d_inner // self.args.mamba2_head_dim
        self._valid_parameter()
        self.tp_split_dim = {
            'word_embeddings.weight': 0,
            'norm.weight': -1,
            'final_norm.weight': -1,
            'output_layer.weight': 0,
            'A_log': 0,
            'D': 0,
            'dt_bias': 0,
            'in_proj.weight': 0,
            'conv1d.weight': 0,
            'conv1d.bias': 0,
            'x_proj.weight': 1,
            'dt_proj.weight': 0,
            'dt_proj.bias': 0,
            'out_proj.weight': 1,
            'mixer.norm.weight': 0,
            'linear_fc1.layer_norm_weight': -1,
            'linear_fc1.weight': 0,
            'linear_fc2.weight': 1,
            'self_attention.linear_proj.weight': 1,
            'self_attention.linear_qkv.layer_norm_weight': -1,
            'self_attention.linear_qkv.weight': 0,
        }
        
    def get_split_dim(self, tensor_name):
        if 'norm.weight' in tensor_name:
            if 'mixer.norm.weight' in tensor_name:
                return self.tp_split_dim['mixer.norm.weight']
            else:
                return self.tp_split_dim['norm.weight']
        for key in self.tp_split_dim.keys():
            if key in tensor_name:
                return self.tp_split_dim[key]
        raise Exception(f"Unknown tensor name {tensor_name}")

    def _valid_parameter(self):
        if self.args.mamba2_n_groups % self.args.target_tp_size != 0:
            raise ValueError("target_tp_size must can divide n_groups. Please adjust values.")

    @staticmethod
    def load_hf_files_to_dict(directory_path):
        model_dict = {}
        loaded = False  # Flag to indicate if any file was successfully loaded

        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)

            try:
                if filename.endswith(".bin"):
                    cur_weights = torch.load(file_path, map_location=torch.device('cpu'))
                    model_dict.update(cur_weights)
                    print(f"Successfully loaded: {filename}")
                    loaded = True

                elif filename.endswith(".safetensors"):
                    try:
                        from safetensors.torch import load_file as safe_load
                    except ImportError as e:
                        raise ImportError(
                            "Detected a .safetensors file but the 'safetensors' package is not installed. "
                            "Please install it using `pip install safetensors`."
                        ) from e
                    
                    cur_weights = safe_load(file_path)
                    model_dict.update(cur_weights)
                    print(f"Successfully loaded: {filename}")
                    loaded = True

            except Exception as e:
                print(f"Error loading {filename}: {e}")

        if not loaded:
            raise RuntimeError("No valid weight files (.bin or .safetensors) were found or loaded.")

        return model_dict

    @staticmethod
    def load_config_and_get_n_layer(directory_path):
        config_file_path = os.path.join(directory_path, "config.json")

        if not os.path.exists(config_file_path):
            raise FileNotFoundError(f"{config_file_path} not found!")

        with open(config_file_path, 'r') as f:
            config = json.load(f)
        n_layer = config.get("num_hidden_layers", config.get("n_layer", None))

        if n_layer is None:
            raise KeyError("Neither 'num_hidden_layers' nor 'n_layers' key found in the config file")
        
        return n_layer

    @staticmethod
    def modify_keys_hf2mg(model_dict):

        modified_dict = {}
        for key, value in model_dict.items():
            new_key = key
            if "norm_f" in key:
                new_key = key.replace("backbone.norm_f", "decoder.final_norm")
            elif "lm_head" in key:
                new_key = key.replace("lm_head", "output_layer")
            elif "embedding" in key:
                if "backbone.embeddings" in key:
                    new_key = key.replace("backbone.embeddings", "embedding.word_embeddings")
                else:
                    new_key = key.replace("backbone.embedding", "embedding.word_embeddings")
            elif "backbone" in key:
                new_key = key.replace("backbone", "decoder")
            modified_dict[new_key] = value

        return modified_dict

    @staticmethod
    def modify_keys_mg2hf(modified_dict):

        restored_dict = {}
        for key, value in modified_dict.items():
            new_key = key
            if "final_norm" in key:
                new_key = key.replace("decoder.final_norm", "backbone.norm_f")
            elif "output_layer" in key:
                new_key = key.replace("output_layer", "lm_head")
            elif "embedding" in key:
                new_key = key.replace("embedding.word_embeddings", "backbone.embeddings")
            elif "decoder" in key:
                new_key = key.replace("decoder", "backbone")
            restored_dict[new_key] = value

        return restored_dict

    def combine_tp_tensors(self, params, key, dim, tensors):
        tp_size = len(tensors)

        if 'mixer.in_proj.weight' in key:
            xs = []
            zs = []
            Bs = []
            Cs = []
            dts = []
            for tensor in tensors:
                x, z, B, C, dt = torch.split(tensor, [params.mamba_d_inner // tp_size,
                                                    params.mamba_d_inner // tp_size,
                                                    (params.mamba2_n_groups // tp_size) * self.args.mamba_d_state,
                                                    (params.mamba2_n_groups // tp_size) * self.args.mamba_d_state,
                                                    params.mamba2_n_heads // tp_size], dim=dim)
                xs.append(x)
                zs.append(z)
                Bs.append(B)
                Cs.append(C)
                dts.append(dt)

            for tensor_index, (B_tensor, C_tensor) in enumerate(zip(Bs, Cs)):
                Bs[tensor_index] = torch.reshape(B_tensor, (-1, params.mamba_d_state, B_tensor.shape[-1]))
                Cs[tensor_index] = torch.reshape(C_tensor, (-1, params.mamba_d_state, C_tensor.shape[-1]))

            B = torch.cat(Bs, dim=dim)
            C = torch.cat(Cs, dim=dim)
            x = torch.cat(xs, dim=dim)
            z = torch.cat(zs, dim=dim)
            dt = torch.cat(dts, dim=dim)

            return torch.cat([x, z, B.flatten(0, 1), C.flatten(0, 1), dt], dim=dim)

        elif 'mixer.conv1d' in key:
            xs = []
            Bs = []
            Cs = []
            for tensor in tensors:
                x, B, C = torch.split(tensor, [params.mamba_d_inner // tp_size,
                                            (params.mamba2_n_groups // tp_size) * params.mamba_d_state,
                                            (params.mamba2_n_groups // tp_size) * params.mamba_d_state], dim=dim)
                xs.append(x)
                Bs.append(B)
                Cs.append(C)

            for tensor_index, (B_tensor, C_tensor) in enumerate(zip(Bs, Cs)):
                if 'weight' in key:
                    Bs[tensor_index] = torch.reshape(B_tensor, (-1, params.mamba_d_state, B_tensor.shape[-2], B_tensor.shape[-1]))
                    Cs[tensor_index] = torch.reshape(C_tensor, (-1, params.mamba_d_state, C_tensor.shape[-2], C_tensor.shape[-1]))
                elif 'bias' in key:
                    Bs[tensor_index] = torch.reshape(B_tensor, (-1, params.mamba_d_state))
                    Cs[tensor_index] = torch.reshape(C_tensor, (-1, params.mamba_d_state))
                else:
                    raise Exception("Unknown key")
            B = torch.cat(Bs, dim=dim)
            C = torch.cat(Cs, dim=dim)
            x = torch.cat(xs, dim=dim)

            return torch.cat([x, B.flatten(0, 1), C.flatten(0, 1)], dim=dim)

        else:
            return torch.cat(tensors, dim=dim)

    @staticmethod
    def split_tensor_for_tp(params, key, dim, tensor):
        tp_size = params.target_tp_size
        tensor_sliced = []

        if 'mixer.in_proj.weight' in key:
            x, z, B, C, dt = torch.split(tensor, [params.mamba_d_inner, params.mamba_d_inner,
                                                        params.mamba2_n_groups * params.mamba_d_state,
                                                        params.mamba2_n_groups * params.mamba_d_state,
                                                        params.mamba2_n_heads], dim=dim)
            B = torch.reshape(B, (-1, params.mamba_d_state, B.shape[-1]))
            C = torch.reshape(C, (-1, params.mamba_d_state, C.shape[-1]))

            B_sliced = torch.chunk(B, tp_size, dim=dim)
            C_sliced = torch.chunk(C, tp_size, dim=dim)
            x_sliced = torch.chunk(x, tp_size, dim=dim)
            z_sliced = torch.chunk(z, tp_size, dim=dim)
            dt_sliced = torch.chunk(dt, tp_size, dim=dim)

            tensor_sliced = []
            for (x, z, B, C, dt) in zip(x_sliced, z_sliced, B_sliced, C_sliced, dt_sliced):
                tensor_sliced.append(torch.cat((x, z, B.flatten(0, 1), C.flatten(0, 1), dt), dim=dim))

        elif 'mixer.conv1d' in key:
            x, B, C = torch.split(tensor, [params.mamba_d_inner,
                                                params.mamba2_n_groups * params.mamba_d_state,
                                                params.mamba2_n_groups * params.mamba_d_state], dim=dim)
            if 'weight' in key:
                B = torch.reshape(B, (-1, params.mamba_d_state, B.shape[-2], B.shape[-1]))
                C = torch.reshape(C, (-1, params.mamba_d_state, C.shape[-2], C.shape[-1]))
            elif 'bias' in key:
                B = torch.reshape(B, (-1, params.mamba_d_state))
                C = torch.reshape(C, (-1, params.mamba_d_state))
            else:
                raise Exception("Unknown key")

            B_sliced = torch.chunk(B, tp_size, dim=dim)
            C_sliced = torch.chunk(C, tp_size, dim=dim)
            x_sliced = torch.chunk(x, tp_size, dim=dim)

            tensor_sliced = []
            for (x, B, C) in zip(x_sliced, B_sliced, C_sliced):
                tensor_sliced.append(torch.cat((x, B.flatten(0, 1), C.flatten(0, 1)), dim=dim))

        else:
            tensor_sliced = torch.chunk(tensor, tp_size, dim=dim)

        return tensor_sliced

    @staticmethod
    def finalize_checkpoint(src_model, model, params, verbose=False):
        if 'args' in src_model.keys():
            model['args'] = copy.deepcopy(src_model['args'])
            model['args'].tensor_model_parallel_size = params.target_tp_size
            model['args'].pipeline_model_parallel_size = params.target_pp_size

        if 'checkpoint_version' in src_model.keys():
            model['checkpoint_version'] = copy.deepcopy(src_model['checkpoint_version'])
        else:
            model['checkpoint_version'] = 3.0

        if 'iteration' in src_model.keys():
            model['iteration'] = copy.deepcopy(src_model['iteration'])
        else:
            model['iteration'] = 1 

        if 'opt_param_scheduler' in src_model.keys():
            model['opt_param_scheduler'] = copy.deepcopy(src_model['opt_param_scheduler'])

        if 'rng_state' in src_model.keys():
            model['rng_state'] = copy.deepcopy(src_model['rng_state'])

        if verbose:
            original_args = src_model['args'].__dict__
            final_args = model['args'].__dict__
            for key in original_args:
                if key in final_args:
                    if final_args[key] != original_args[key]:
                        logger.info("KEY MISMATCH: {}".format(key))
                        logger.info("\toriginal: {}\n\tfinal: {}".format(original_args[key], final_args[key]))
                else:
                    logger.info("KEY MISSING from final: {}, value {}".format(key, original_args[key]))

            for key in final_args:
                if key not in original_args:
                    logger.info("KEY ADDED to final: {}, value {}".format(key, final_args[key]))

        return model

    @staticmethod
    def get_latest_checkpoint_model_file(load_dir, model_filename="model_optim_rng.pt"):
        """
        Get the sub-model path for the latest iteration (any sub-model) to extract structure or parameter information.

        Parameters:
        - load_dir: Main directory to load the model
        - model_filename: Sub-model filename (default is "model_optim_rng.pt")

        Returns:
        - out_iteration: Latest iteration number
        - input_model_dir: Iteration directory path
        - src_model_file: Full file path of any sub-model
        """
        tracker_filename = os.path.join(load_dir, 'latest_checkpointed_iteration.txt')

        try:
            with open(tracker_filename, 'r') as f:
                metastring = f.read().strip()
                iteration = int(metastring)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Checkpoint tracker file not found at {tracker_filename}") from e
        except ValueError as e:
            raise ValueError(f"Invalid iteration value in {tracker_filename}: '{metastring}' is not an integer.") from e

        out_iteration = iteration
        input_model_dir = os.path.join(load_dir, f'iter_{iteration:07d}')
        input_sub_models = os.listdir(input_model_dir)

        if not input_sub_models:
            raise RuntimeError(f"No sub-models found under {input_model_dir}")

        src_model_file = os.path.join(input_model_dir, input_sub_models[0], model_filename)
        return out_iteration, input_model_dir, src_model_file


    def load_mg_model(self):
        """
        Load the latest iteration number and retrieve a model checkpoint from the model directory to obtain structure and other information.

        Returns:
        - out_iteration: Iteration number
        - input_model_dir: Corresponding iteration directory
        - src_model: Sample model dict
        """

        out_iteration, input_model_dir, src_model_file = self.get_latest_checkpoint_model_file(self.args.load_dir)

        src_model = torch.load(src_model_file, map_location='cpu', weights_only=False)

        logger.info(f"Sample model {src_model_file} is loaded.\n")
        return out_iteration, input_model_dir, src_model

    @staticmethod
    def get_model_file_path(input_model_dir, tp_index, pp_index=0, input_pp_rank=1, filename="model_optim_rng.pt"):

        dir_name = f"mp_rank_{tp_index:02d}"
        if input_pp_rank > 1:
            dir_name += f"_{pp_index:03d}"
        return os.path.join(input_model_dir, dir_name, filename)

    def merge_checkpoints(self, input_model_dir, input_tp_rank, input_pp_rank, num_layers_per_pipeline_rank, args):
        """
        Load and merge Megatron-style TP+PP model weights, and return the merged full model (OrderedDict)
        """

        full_model = OrderedDict()

        for pp_index in range(input_pp_rank):
            logger.info(f"Processing input pipeline rank {pp_index}")
            tp_models = []

            for tp_index in range(input_tp_rank):
                model_file = self.get_model_file_path(
                    input_model_dir,
                    tp_index,
                    pp_index,
                    input_pp_rank
                )

                tp_models.append(torch.load(model_file, map_location='cpu', weights_only=False))
                logger.info(f"Model {model_file} is loaded.")

            if input_tp_rank > 1:
                combined_tp_model = OrderedDict()
                for key, tensor in tp_models[0]['model'].items():
                    if "_extra_state" in key:
                        combined_tp_model[key] = tensor
                        continue

                    split_dim = self.get_split_dim(key)
                    if split_dim != -1:
                        combined_tensor = self.combine_tp_tensors(
                            args, key, split_dim,
                            [tp_models[i]['model'][key].cpu() for i in range(input_tp_rank)]
                        )
                        combined_tp_model[key] = combined_tensor
                    else:
                        combined_tp_model[key] = tensor
            else:
                combined_tp_model = tp_models[0]['model']

            for key, tensor in combined_tp_model.items():
                try:
                    layer_num = int(re.findall(r'\d+', key)[0])
                    new_key = key.replace(str(layer_num), str(layer_num + pp_index * num_layers_per_pipeline_rank), 1)
                except Exception:
                    new_key = key
                full_model[new_key] = tensor

        logger.info("Loaded combined model.")
        return full_model

    def split_model_by_pp_tp(self, full_model, args):
        """
        split Megatron model by pipeline rank and tensor parallel rank
        """
        split_models = {}
        layers_per_pp = args.num_layers // args.target_pp_size

        for pp_idx in range(args.target_pp_size):
            tp_models = [{'model': OrderedDict()} for _ in range(args.target_tp_size)]

            for key, tensor in full_model.items():
                try:
                    layer_num = int(re.findall(r'\d+', key)[0])
                    if layer_num >= layers_per_pp * (pp_idx + 1):
                        continue
                    new_key = key.replace(str(layer_num), str(layer_num - layers_per_pp * pp_idx), 1)
                except Exception:
                    new_key = key

                if "_extra_state" in key:
                    for tp_model in tp_models:
                        tp_model['model'][new_key] = tensor
                    continue

                split_dim = self.get_split_dim(new_key)
                if split_dim != -1:
                    slices = self.split_tensor_for_tp(args, new_key, split_dim, tensor)
                    for i in range(args.target_tp_size):
                        tp_models[i]['model'][new_key] = slices[i]
                else:
                    for tp_model in tp_models:
                        tp_model['model'][new_key] = tensor

            for tp_idx, tp_model in enumerate(tp_models):
                split_models[(pp_idx, tp_idx)] = tp_model['model']

        return split_models

    @staticmethod
    def build_model_save_path(args, out_iteration, tp_idx, pp_idx=0):
        """
        Constructs and creates the path to save the model, and returns the full model file path.

        Parameters:
        - save_dir: Top-level directory to save
        - out_iteration: Current iteration number (int)
        - tp_idx: Tensor parallel rank
        - pp_idx: Pipeline parallel rank (default is 0)
        - target_pp_size: Pipeline parallel size (default is 1)
        - filename: Filename (default is 'model_optim_rng.pt')

        Returns:
        - model_file: The full model file path in the save directory
        """

        filename = "model_optim_rng.pt"
        dir_name = f"mp_rank_{tp_idx:02d}"
        if args.target_pp_size > 1:
            dir_name += f"_{pp_idx:03d}"

        save_path = os.path.join(args.save_dir, f"iter_{out_iteration:07d}", dir_name)
        os.makedirs(save_path, exist_ok=True)

        return os.path.join(save_path, filename)

    def save_split_models(self, split_models, src_model, args, out_iteration):
        """
        save splited model
        """
        for (pp_idx, tp_idx), model_dict in split_models.items():
            model_file = self.build_model_save_path(
                args,
                out_iteration,
                tp_idx,
                pp_idx,
            )

            finalized = self.finalize_checkpoint(src_model, {'model': model_dict}, args, verbose=False)
            torch.save(finalized, model_file)
            logger.info(f"Model {model_file} is saved.")

        tracker_file = os.path.join(args.save_dir, "latest_checkpointed_iteration.txt")
        with open(tracker_file, "w") as f:
            f.write(str(out_iteration))

    def convert_mg_checkpoint(self, args):
        logger.info("====RUNNING CHECKPOINT CONVERSION====")

        out_iteration, input_model_dir, src_model = self.load_mg_model()

        # input tensor and pipeline parallel size
        if 'args' in src_model.keys():
            input_tp_rank = src_model['args'].tensor_model_parallel_size
            input_pp_rank = src_model['args'].pipeline_model_parallel_size
            num_layers_per_pipeline_rank = src_model['args'].num_layers // input_pp_rank
        else:
            input_tp_rank = self.args.input_tp_rank
            input_pp_rank = self.args.input_pp_rank
            num_layers = self.args.num_layers
            num_layers_per_pipeline_rank = num_layers // input_pp_rank

        # construct full model
        full_model = self.merge_checkpoints(
                    input_model_dir=input_model_dir,
                    input_tp_rank=input_tp_rank,
                    input_pp_rank=input_pp_rank,
                    num_layers_per_pipeline_rank=num_layers_per_pipeline_rank,
                    args=args)

        if args.save_model_type == 'hf':
            hf_model = self.modify_keys_mg2hf(full_model)
            # Ensure the parent directory exists
            model_file = os.path.join(args.save_dir, "torch_model.bin")
            torch.save(hf_model, model_file)
            logger.info(f"Model saved to {model_file}.")

        else:
            if 'args' in src_model:
                args.num_layers = src_model['args'].num_layers

            split_models = self.split_model_by_pp_tp(full_model, args)
            self.save_split_models(split_models, src_model, args, out_iteration)

    def convert_hf2mg(self, args):
        logger.info("====RUNNING CHECKPOINT CONVERSION====")

        hf_model = self.load_hf_files_to_dict(args.load_dir)
        hf_model_new = self.modify_keys_hf2mg(hf_model)
        args.num_layers = self.load_config_and_get_n_layer(args.load_dir)

        split_models = self.split_model_by_pp_tp(hf_model_new, args)
        self.save_split_models(split_models, hf_model, args, out_iteration=1)

    def main(self):
        if self.args.load_model_type == "mg":
            self.convert_mg_checkpoint(self.args)
        else:
            self.convert_hf2mg(self.args)


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load-dir', type=str,
                        help='Directory from which to load the model checkpoint')
    parser.add_argument('--save-dir', type=str,
                        help='Directory to save the converted model checkpoint')
    parser.add_argument('--target-tp-size', type=int, default=1,
                        help='Target Tensor Parallel (TP) size for the output model')
    parser.add_argument('--target-pp-size', type=int, default=1,
                        help='Target Pipeline Parallel (PP) size for the output model')
    parser.add_argument('--reset-iterations', action='store_true',
                        help='Whether to reset the iteration count in the saved model')
    parser.add_argument('--input-tp-rank', type=int,
                        help='Tensor Parallel rank of the input model shard')
    parser.add_argument('--input-pp-rank', type=int,
                        help='Pipeline Parallel rank of the input model shard')
    parser.add_argument('--num-layers', type=int,
                        help='Number of layers in the model')
    parser.add_argument('--d-model', type=int, default=4096,
                        help='Model dimension (hidden size)')
    parser.add_argument('--mamba-d-state', type=int, default=128,
                        help='State dimension used in the Mamba model')
    parser.add_argument('--mamba2-n-groups', type=int, default=8,
                        help='Number of groups in Mamba v2 model')
    parser.add_argument('--mamba2-head-dim', type=int, default=64,
                        help='Head dimension in Mamba v2 model')
    parser.add_argument('--load-model-type', type=str, nargs='?',
                        default=None, const=None, choices=['hf', 'mg'],
                        help='Checkpoint format to load: "hf" for HuggingFace or "mg" for Megatron')
    parser.add_argument('--save-model-type', type=str, default='mg',
                        choices=['mg', 'hf'],
                        help='Checkpoint format to save: "hf" for HuggingFace or "mg" for Megatron')

    args, _ = parser.parse_known_args()

    converter = CheckpointConverter(args)
    converter.main()


if __name__ == "__main__":
    run()
