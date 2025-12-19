# This file loads the parameters and returns them as a list of TTNN tensors

import ttnn
from . import utils
import os

# Get the directory where this file is located
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_TENSORS_DIR = os.path.join(_CURRENT_DIR, "tensors")


def load_inputs_for_clipvision_ttnn(mesh_device):

    utils_load_tensor_0 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg0.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_1 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg1.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_2 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg2.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_3 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg3.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_4 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg4.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_5 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg5.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_6 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg6.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_7 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg7.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_8 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg8.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_9 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg9.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_10 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg10.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_11 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg11.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_12 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg12.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_13 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg13.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_14 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg14.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_15 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg15.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_16 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg16.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_17 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg17.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_18 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg18.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_19 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg19.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_20 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg20.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_21 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg21.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_22 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg22.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_23 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg23.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_24 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg24.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_25 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg25.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_26 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg26.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_27 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg27.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_28 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg28.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_29 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg29.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_30 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg30.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_31 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg31.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_32 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg32.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_33 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg33.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_34 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg34.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_35 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg35.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_36 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg36.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_37 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg37.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_38 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg38.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_39 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg39.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_40 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg40.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_41 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg41.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_42 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg42.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_43 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg43.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_44 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg44.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_45 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg45.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_46 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg46.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_47 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg47.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_48 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg48.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_49 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg49.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_50 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg50.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_51 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg51.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_52 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg52.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_53 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg53.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_54 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg54.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_55 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg55.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_56 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg56.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_57 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg57.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_58 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg58.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_59 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg59.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_60 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg60.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_61 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg61.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_62 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg62.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_63 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg63.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_64 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg64.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_65 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg65.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_66 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg66.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_67 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg67.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_68 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg68.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_69 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg69.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_70 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg70.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_71 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg71.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_72 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg72.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_73 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg73.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_74 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg74.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_75 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg75.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_76 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg76.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_77 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg77.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_78 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg78.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_79 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg79.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_80 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg80.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_81 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg81.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_82 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg82.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_83 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg83.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_84 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg84.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_85 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg85.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_86 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg86.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_87 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg87.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_88 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg88.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_89 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg89.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_90 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg90.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_91 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg91.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_92 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg92.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_93 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg93.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_94 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg94.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_95 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg95.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_96 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg96.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_97 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg97.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_98 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg98.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_99 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg99.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_100 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg100.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_101 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg101.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_102 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg102.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_103 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg103.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_104 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg104.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_105 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg105.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_106 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg106.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_107 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg107.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_108 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg108.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_109 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg109.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_110 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg110.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_111 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg111.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_112 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg112.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_113 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg113.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_114 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg114.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_115 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg115.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_116 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg116.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_117 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg117.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_118 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg118.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_119 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg119.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_120 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg120.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_121 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg121.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_122 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg122.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_123 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg123.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_124 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg124.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_125 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg125.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_126 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg126.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_127 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg127.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_128 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg128.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_129 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg129.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_130 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg130.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_131 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg131.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_132 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg132.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_133 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg133.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_134 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg134.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_135 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg135.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_136 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg136.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_137 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg137.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_138 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg138.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_139 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg139.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_140 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg140.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_141 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg141.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_142 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg142.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_143 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg143.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_144 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg144.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_145 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg145.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_146 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg146.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_147 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg147.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_148 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg148.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_149 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg149.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.INT32,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_150 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg150.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_151 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg151.tensorbin"),
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    # Don't load activation tensor, it will be loaded in main() function
    utils_load_tensor_152 = None
    # utils_load_tensor_152 = utils.load_tensor(
    #     os.path.join(_TENSORS_DIR, "arg152.tensorbin"),
    #     ttnn.Layout.TILE,
    #     ttnn.DataType.BFLOAT16,
    #     mesh_device,
    #     ttnn.MemoryConfig(
    #         ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
    #     ),
    # )
    utils_load_tensor_153 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg153.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_154 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg154.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_155 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg155.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_156 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg156.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_157 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg157.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_158 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg158.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_159 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg159.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_160 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg160.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_161 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg161.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_162 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg162.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_163 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg163.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_164 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg164.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_165 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg165.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_166 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg166.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_167 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg167.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_168 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg168.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_169 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg169.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_170 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg170.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_171 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg171.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_172 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg172.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_173 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg173.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_174 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg174.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_175 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg175.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_176 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg176.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_177 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg177.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_178 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg178.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_179 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg179.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_180 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg180.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_181 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg181.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_182 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg182.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_183 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg183.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_184 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg184.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_185 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg185.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_186 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg186.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_187 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg187.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_188 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg188.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_189 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg189.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_190 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg190.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_191 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg191.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_192 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg192.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_193 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg193.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_194 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg194.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_195 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg195.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_196 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg196.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_197 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg197.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_198 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg198.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_199 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg199.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_200 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg200.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_201 = utils.load_tensor(
        os.path.join(_TENSORS_DIR, "arg201.tensorbin"),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_262 = [
        utils_load_tensor_0,
        utils_load_tensor_1,
        utils_load_tensor_2,
        utils_load_tensor_3,
        utils_load_tensor_4,
        utils_load_tensor_5,
        utils_load_tensor_6,
        utils_load_tensor_7,
        utils_load_tensor_8,
        utils_load_tensor_9,
        utils_load_tensor_10,
        utils_load_tensor_11,
        utils_load_tensor_12,
        utils_load_tensor_13,
        utils_load_tensor_14,
        utils_load_tensor_15,
        utils_load_tensor_16,
        utils_load_tensor_17,
        utils_load_tensor_18,
        utils_load_tensor_19,
        utils_load_tensor_20,
        utils_load_tensor_21,
        utils_load_tensor_22,
        utils_load_tensor_23,
        utils_load_tensor_24,
        utils_load_tensor_25,
        utils_load_tensor_26,
        utils_load_tensor_27,
        utils_load_tensor_28,
        utils_load_tensor_29,
        utils_load_tensor_30,
        utils_load_tensor_31,
        utils_load_tensor_32,
        utils_load_tensor_33,
        utils_load_tensor_34,
        utils_load_tensor_35,
        utils_load_tensor_36,
        utils_load_tensor_37,
        utils_load_tensor_38,
        utils_load_tensor_39,
        utils_load_tensor_40,
        utils_load_tensor_41,
        utils_load_tensor_42,
        utils_load_tensor_43,
        utils_load_tensor_44,
        utils_load_tensor_45,
        utils_load_tensor_46,
        utils_load_tensor_47,
        utils_load_tensor_48,
        utils_load_tensor_49,
        utils_load_tensor_50,
        utils_load_tensor_51,
        utils_load_tensor_52,
        utils_load_tensor_53,
        utils_load_tensor_54,
        utils_load_tensor_55,
        utils_load_tensor_56,
        utils_load_tensor_57,
        utils_load_tensor_58,
        utils_load_tensor_59,
        utils_load_tensor_60,
        utils_load_tensor_61,
        utils_load_tensor_62,
        utils_load_tensor_63,
        utils_load_tensor_64,
        utils_load_tensor_65,
        utils_load_tensor_66,
        utils_load_tensor_67,
        utils_load_tensor_68,
        utils_load_tensor_69,
        utils_load_tensor_70,
        utils_load_tensor_71,
        utils_load_tensor_72,
        utils_load_tensor_73,
        utils_load_tensor_74,
        utils_load_tensor_75,
        utils_load_tensor_76,
        utils_load_tensor_77,
        utils_load_tensor_78,
        utils_load_tensor_79,
        utils_load_tensor_80,
        utils_load_tensor_81,
        utils_load_tensor_82,
        utils_load_tensor_83,
        utils_load_tensor_84,
        utils_load_tensor_85,
        utils_load_tensor_86,
        utils_load_tensor_87,
        utils_load_tensor_88,
        utils_load_tensor_89,
        utils_load_tensor_90,
        utils_load_tensor_91,
        utils_load_tensor_92,
        utils_load_tensor_93,
        utils_load_tensor_94,
        utils_load_tensor_95,
        utils_load_tensor_96,
        utils_load_tensor_97,
        utils_load_tensor_98,
        utils_load_tensor_99,
        utils_load_tensor_100,
        utils_load_tensor_101,
        utils_load_tensor_102,
        utils_load_tensor_103,
        utils_load_tensor_104,
        utils_load_tensor_105,
        utils_load_tensor_106,
        utils_load_tensor_107,
        utils_load_tensor_108,
        utils_load_tensor_109,
        utils_load_tensor_110,
        utils_load_tensor_111,
        utils_load_tensor_112,
        utils_load_tensor_113,
        utils_load_tensor_114,
        utils_load_tensor_115,
        utils_load_tensor_116,
        utils_load_tensor_117,
        utils_load_tensor_118,
        utils_load_tensor_119,
        utils_load_tensor_120,
        utils_load_tensor_121,
        utils_load_tensor_122,
        utils_load_tensor_123,
        utils_load_tensor_124,
        utils_load_tensor_125,
        utils_load_tensor_126,
        utils_load_tensor_127,
        utils_load_tensor_128,
        utils_load_tensor_129,
        utils_load_tensor_130,
        utils_load_tensor_131,
        utils_load_tensor_132,
        utils_load_tensor_133,
        utils_load_tensor_134,
        utils_load_tensor_135,
        utils_load_tensor_136,
        utils_load_tensor_137,
        utils_load_tensor_138,
        utils_load_tensor_139,
        utils_load_tensor_140,
        utils_load_tensor_141,
        utils_load_tensor_142,
        utils_load_tensor_143,
        utils_load_tensor_144,
        utils_load_tensor_145,
        utils_load_tensor_146,
        utils_load_tensor_147,
        utils_load_tensor_148,
        utils_load_tensor_149,
        utils_load_tensor_150,
        utils_load_tensor_151,
        utils_load_tensor_152,
        utils_load_tensor_153,
        utils_load_tensor_154,
        utils_load_tensor_155,
        utils_load_tensor_156,
        utils_load_tensor_157,
        utils_load_tensor_158,
        utils_load_tensor_159,
        utils_load_tensor_160,
        utils_load_tensor_161,
        utils_load_tensor_162,
        utils_load_tensor_163,
        utils_load_tensor_164,
        utils_load_tensor_165,
        utils_load_tensor_166,
        utils_load_tensor_167,
        utils_load_tensor_168,
        utils_load_tensor_169,
        utils_load_tensor_170,
        utils_load_tensor_171,
        utils_load_tensor_172,
        utils_load_tensor_173,
        utils_load_tensor_174,
        utils_load_tensor_175,
        utils_load_tensor_176,
        utils_load_tensor_177,
        utils_load_tensor_178,
        utils_load_tensor_179,
        utils_load_tensor_180,
        utils_load_tensor_181,
        utils_load_tensor_182,
        utils_load_tensor_183,
        utils_load_tensor_184,
        utils_load_tensor_185,
        utils_load_tensor_186,
        utils_load_tensor_187,
        utils_load_tensor_188,
        utils_load_tensor_189,
        utils_load_tensor_190,
        utils_load_tensor_191,
        utils_load_tensor_192,
        utils_load_tensor_193,
        utils_load_tensor_194,
        utils_load_tensor_195,
        utils_load_tensor_196,
        utils_load_tensor_197,
        utils_load_tensor_198,
        utils_load_tensor_199,
        utils_load_tensor_200,
        utils_load_tensor_201,
    ]
    return util_create_list_262

