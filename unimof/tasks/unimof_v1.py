import logging
import os

import numpy as np
from unicore.data import (AppendTokenDataset, Dictionary, EpochShuffleDataset,
                          LMDBDataset, NestedDictionaryDataset,
                          PrependTokenDataset, RawLabelDataset,
                          RightPadDataset, RightPadDataset2D, TokenizeDataset)
from unicore.tasks import UnicoreTask, register_task
from unimof.data import (CroppingDataset, DistanceDataset, EdgeTypeDataset,
                         KeyDataset, LatticeNormalizeDataset,
                         MaskPointsDataset, NormalizeDataset,
                         NumericalTransformDataset, PrependAndAppend2DDataset,
                         RemoveHydrogenDataset, RightPadDatasetCoord,
                         ToTorchDataset)

logger = logging.getLogger(__name__)


@register_task("unimof_v1")
class UniMOFV1Task(UnicoreTask):
    """Task for training transformer auto-encoder models."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument("data", help="downstream data path")
        parser.add_argument(
            "--task-name", type=str, default="", help="downstream task name"
        )
        parser.add_argument(
            "--classification-head-name",
            default="classification",
            help="finetune downstream task name",
        )
        parser.add_argument(
            "--num-classes",
            default=1,
            type=int,
            help="finetune downstream task classes numbers",
        )
        parser.add_argument(
            "--max-atoms",
            type=int,
            default=512,
            help="selected maximum number of atoms in a molecule",
        )
        parser.add_argument("--dict-name", default="dict.txt", help="dictionary file")
        parser.add_argument(
            "--remove-hydrogen", action="store_true", help="remove hydrogen atoms"
        )
    def __init__(self, args, dictionary):
        super().__init__(args)
        self.dictionary = dictionary
        self.seed = args.seed
        self.mask_idx = dictionary.add_symbol("[MASK]", is_special=True)

    @classmethod
    def setup_task(cls, args, **kwargs):
        dictionary = Dictionary.load(os.path.join(args.data, args.dict_name))
        logger.info("dictionary: {} types".format(len(dictionary)))
        return cls(args, dictionary)

    def load_dataset(self, split, **kwargs):
        """Load a given dataset split。
        Args:
            split (str): name of the data scoure (e.g., train)
        """
        # 拼接数据集的路径
        split_path = os.path.join(self.args.data, self.args.task_name, split + ".lmdb")
        # 加载LMDB数据集
        dataset = LMDBDataset(split_path)
        # 获取目标数据集
        tgt_dataset = KeyDataset(dataset, "target")
        # 将目标数据集转换为PyTorch数据集
        tgt_dataset = ToTorchDataset(tgt_dataset, dtype="float32")
        # 如果参数中指定移除氢原子，则移除氢原子
        if self.args.remove_hydrogen:
            dataset = RemoveHydrogenDataset(dataset, "atoms", "coordinates")
        # 对数据集进行裁剪
        dataset = CroppingDataset(
            dataset, self.seed, "atoms", "coordinates", self.args.max_atoms
        )
        # 对数据集进行归一化处理
        dataset = NormalizeDataset(dataset, "coordinates")
        # 获取源数据集
        src_dataset = KeyDataset(dataset, "atoms")
        # 对源数据集进行分词处理
        src_dataset = TokenizeDataset(
            src_dataset, self.dictionary, max_seq_len=self.args.max_seq_len
        )
        # 获取坐标数据集
        coord_dataset = KeyDataset(dataset, "coordinates")

        # 定义一个辅助函数，用于在数据集的前后添加特定的token
        def PrependAndAppend(dataset, pre_token, app_token):
            # 在数据集前面添加token
            dataset = PrependTokenDataset(dataset, pre_token)
            # 在数据集后面添加token
            return AppendTokenDataset(dataset, app_token)

        # 在源数据集的前后添加特定的token
        src_dataset = PrependAndAppend(
            src_dataset, self.dictionary.bos(), self.dictionary.eos()
        )
        # 获取边类型数据集
        edge_type = EdgeTypeDataset(src_dataset, len(self.dictionary))
        # 将坐标数据集转换为PyTorch数据集
        coord_dataset = ToTorchDataset(coord_dataset, "float32")
        # 计算坐标数据集之间的距离
        distance_dataset = DistanceDataset(coord_dataset)
        # 在坐标数据集的前后添加特定的值
        coord_dataset = PrependAndAppend(coord_dataset, 0.0, 0.0)
        # 在距离数据集的前后添加特定的值
        distance_dataset = PrependAndAppend2DDataset(distance_dataset, 0.0)
        # 创建一个嵌套的字典数据集
        nest_dataset = NestedDictionaryDataset(
            {
                "net_input": {
                    # 对源token进行右填充
                    "src_tokens": RightPadDataset(
                        src_dataset, pad_idx=self.dictionary.pad()
                    ),
                    # 对坐标数据集进行右填充
                    "src_coord": RightPadDatasetCoord(coord_dataset, pad_idx=0),
                    # 对距离数据集进行二维右填充
                    "src_distance": RightPadDataset2D(distance_dataset, pad_idx=0),
                    # 对边类型数据集进行二维右填充
                    "src_edge_type": RightPadDataset2D(edge_type, pad_idx=0),
                },
                # 设置目标数据集
                "target": {"finetune_target": tgt_dataset},
            }
        )
        # 如果split是train或train.small，则对嵌套的数据集进行epoch shuffle
        if split in ["train", "train.small"]:
            nest_dataset = EpochShuffleDataset(
                nest_dataset, len(nest_dataset), self.args.seed
            )
        # 将处理好的数据集保存到self.datasets中
        self.datasets[split] = nest_dataset

    def build_model(self, args):
        # 从unicore库中导入models模块
        from unicore import models

        # 使用args和self作为参数调用models模块中的build_model函数来构建模型
        model = models.build_model(args, self)

        # 使用args中的classification_head_name和num_classes参数
        # 在模型中注册分类头
        model.register_classification_head(
            # 使用args中的classification_head_name参数
            self.args.classification_head_name,
            # 使用args中的num_classes参数
            num_classes=self.args.num_classes
        )

        # 返回构建好的模型
        return model
