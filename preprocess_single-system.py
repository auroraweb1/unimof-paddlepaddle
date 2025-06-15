import os
import warnings
from multiprocessing import Pool, Process, Queue

from pymatgen.core import Structure
from pymatgen.io.cif import CifParser
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from tqdm import tqdm

warnings.filterwarnings("ignore")
import glob
import pickle
import re
import sys

import lmdb
import numpy as np
import pandas as pd


def normalize_atoms(atom):
    """
    对原子符号进行规范化处理，去除数字。

    Args:
        atom (str): 待规范化的原子符号，例如 'H1', 'C2', 'O3'。

    Returns:
        str: 规范化后的原子符号，例如 'H', 'C', 'O'。

    """
    return re.sub("\\d+", "", atom)


def cif_parser(cif_path, primitive=False):
    """
    解析单个cif文件

    Args:
        cif_path (str): cif文件的路径。
        primitive (bool, optional): 是否将结构简化为原胞。默认为False。

    Returns:
        dict: 包含解析结果的字典，包含以下字段：
            - ID (str): cif文件的文件名（不包含扩展名）。
            - atoms (list of str): 原子种类列表，已进行标准化。
            - coordinates (np.ndarray): 原子坐标（以笛卡尔坐标表示）。
            - abc (tuple of float): 晶格常数（a, b, c）。
            - angles (tuple of float): 晶胞夹角（alpha, beta, gamma）。
            - volume (float): 晶胞体积。
            - lattice_matrix (np.ndarray): 晶格矩阵（3x3）。
            - abc_coordinates (np.ndarray): 原子在晶胞坐标系中的坐标。

    Raises:
        AssertionError: 如果解析得到的原子种类列表长度与坐标列表长度不一致，抛出此异常。
    """
    """
    Parser for single cif file
    """
    s = Structure.from_file(cif_path, primitive=primitive)
    id = cif_path.split("/")[-1][:-4]
    lattice = s.lattice
    abc = lattice.abc
    angles = lattice.angles
    volume = lattice.volume
    lattice_matrix = lattice.matrix
    df = s.as_dataframe()
    atoms = df["Species"].astype(str).map(normalize_atoms).tolist()
    coordinates = df[["x", "y", "z"]].values.astype(np.float32)
    abc_coordinates = df[["a", "b", "c"]].values.astype(np.float32)
    assert len(atoms) == coordinates.shape[0]
    assert len(atoms) == abc_coordinates.shape[0]
    return {
        "ID": id,
        "atoms": atoms,
        "coordinates": coordinates,
        "abc": abc,
        "angles": angles,
        "volume": volume,
        "lattice_matrix": lattice_matrix,
        "abc_coordinates": abc_coordinates,
    }


def single_parser(content):
    """
    解析单个cif文件，并将其与目标值打包成pickle格式的数据。

    Args:
        content (tuple): 包含cif文件名和目标值的元组，格式为 (cif_name, targets)。

    Returns:
        bytes or None: 解析后的pickle格式数据，如果文件不存在则返回None。

    """
    dir_path = "/home/unimof/mof_database"
    cif_name, targets = content
    cif_path = os.path.join(dir_path, cif_name + ".cif")
    if os.path.exists(cif_path):
        data = cif_parser(cif_path, primitive=False)
        data["target"] = targets
        return pickle.dumps(data, protocol=-1)
    else:
        print(f"{cif_path} does not exit!")
        return None


def get_data(path):
    """
    从指定路径读取CSV文件，提取数据，并计算目标列的平均值和标准差。

    Args:
        path (str): CSV文件的路径。

    Returns:
        list of tuples: 包含MOF名称和目标值的元组列表。

    """
    data = pd.read_csv(path)
    columns = "target"
    cif_names = "mof-name"
    value = data[columns]
    _mean, _std = value.mean(), value.std()
    print(f"mean and std of target values are: {_mean}, {_std}")
    return [(item[0], item[1]) for item in zip(data[cif_names], data[columns].values)]


def train_valid_test_split(data, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1):
    """
    将数据集分割为训练集、验证集和测试集。

    Args:
        data (list of tuples): 包含多个元组的数据集，每个元组的第一项是唯一的标识符，其余项为数据。
        train_ratio (float, optional): 训练集的比例，默认为0.8。
        valid_ratio (float, optional): 验证集的比例，默认为0.1。
        test_ratio (float, optional): 测试集的比例，默认为0.1。

    Returns:
        tuple: 包含三个列表的元组，分别为训练集、验证集和测试集。

    Raises:
        ValueError: 如果传入的参数比例之和不为1，则抛出异常。

    """
    np.random.seed(42)
    id_list = [item[0] for item in data]
    unique_id_list = list(set(id_list))
    unique_id_list = np.random.permutation(unique_id_list)
    print(f"length of data is {len(data)}")
    print(f"length of unique_id_list is {len(unique_id_list)}")
    train_size = int(len(unique_id_list) * train_ratio)
    valid_size = int(len(unique_id_list) * valid_ratio)
    train_id_list = unique_id_list[:train_size]
    valid_id_list = unique_id_list[train_size : train_size + valid_size]
    test_id_list = unique_id_list[train_size + valid_size :]
    train_data = [item for item in data if item[0] in train_id_list]
    valid_data = [item for item in data if item[0] in valid_id_list]
    test_data = [item for item in data if item[0] in test_id_list]
    print(f"train_len:{len(train_data)}")
    print(f"valid_len:{len(valid_data)}")
    print(f"test_len:{len(test_data)}")
    return train_data, valid_data, test_data


def write_lmdb(inpath="./", outpath="./", nthreads=40):
    """
    将数据集写入LMDB格式。

    Args:
        inpath (str): 输入数据集路径，默认为当前目录（'./'）。
        outpath (str): 输出LMDB文件路径，默认为当前目录（'./'）。
        nthreads (int): 用于并行处理的线程数，默认为40。

    Returns:
        None

    """
    data = get_data(inpath)
    train_data, valid_data, test_data = train_valid_test_split(data)
    print(len(train_data), len(valid_data), len(test_data))
    for name, content in [
        ("train.lmdb", train_data),
        ("valid.lmdb", valid_data),
        ("test.lmdb", test_data),
    ]:
        outputfilename = os.path.join(outpath, name)
        try:
            os.remove(outputfilename)
        except:
            pass
        env_new = lmdb.open(
            outputfilename,
            subdir=False,
            readonly=False,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=1,
            map_size=int(100000000000.0),
        )
        txn_write = env_new.begin(write=True)
        with Pool(nthreads) as pool:
            i = 0
            for inner_output in tqdm(
                pool.imap(single_parser, content), total=len(content)
            ):
                if inner_output is not None:
                    txn_write.put(f"{i}".encode("ascii"), inner_output)
                    i += 1
                    if i % 1000 == 0:
                        print(f"已处理 {i} 条数据")
                        txn_write.commit()
                        txn_write = env_new.begin(write=True)
            print("{} process {} lines".format(name, i))
            txn_write.commit()
            env_new.close()


if __name__ == "__main__":
    inpath = "/home/data/filtered_CoRE_MOF_structure_data.csv"
    outpath = "/home/data/CoRE_volume"
    write_lmdb(inpath=inpath, outpath=outpath, nthreads=8)
