set -e

python -c "from mindspeed.op_builder import GMMOpBuilder; GMMOpBuilder().load()"
python -c "from mindspeed.op_builder import GMMV2OpBuilder; GMMV2OpBuilder().load()"
python -c "from mindspeed.op_builder import MatmulAddOpBuilder; MatmulAddOpBuilder().load()"
python -c "from mindspeed.op_builder import MoeTokenPermuteOpBuilder; MoeTokenPermuteOpBuilder().load()"
python -c "from mindspeed.op_builder import MoeTokenUnpermuteOpBuilder; MoeTokenUnpermuteOpBuilder().load()"
python -c "from mindspeed.op_builder import RotaryPositionEmbeddingOpBuilder; RotaryPositionEmbeddingOpBuilder().load()"
python -c "from mindspeed.op_builder import GroupMatmulAddOpBuilder; GroupMatmulAddOpBuilder().load()"
python -c "from mindspeed.op_builder.fused_adamw_v2_builder import FusedAdamWV2OpBuilder; FusedAdamWV2OpBuilder().load()"

compile_cann_op() {
    python - "$1" <<'PY'
import importlib
import inspect
import sys

name = sys.argv[1]
module = importlib.import_module(name)
builders = {}

for value in vars(module).values():
    values = [value]
    if inspect.isfunction(value):
        values.extend(value.__globals__.values())

    for obj in values:
        if not inspect.isclass(obj):
            load = getattr(obj, "load", None)
            if callable(load) and "builder" in type(obj).__name__.lower():
                builders[id(obj)] = obj

if not builders:
    raise RuntimeError(f"No Builder found in {name}")

for builder in builders.values():
    print(f"[compile] {name}: {type(builder).__name__}")
    builder.load()
PY
}

for op in \
    cann_ops_transformer.ops.attention.compressor \
    cann_ops_transformer.ops.attention.lightning_indexer_v2 \
    cann_ops_transformer.ops.attention.sparse_flash_mla \
    cann_ops_transformer.ops.attention.sparse_flash_mla_grad \
    cann_ops_transformer.ops.attention.sparse_lightning_indexer_kl_loss_grad \
    cann_ops_transformer.ops.mhc.mhc_pre_sinkhorn \
    cann_ops_transformer.ops.mhc.mhc_pre_sinkhorn_backward \
    cann_ops_transformer.ops.mhc.mhc_post \
    cann_ops_transformer.ops.mhc.mhc_post_backward
do
    compile_cann_op "$op"
done

echo "Training operators precompiled successfully."
