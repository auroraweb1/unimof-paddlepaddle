import os
import pickle
import sys

import lmdb


def wirte_to_lmbd(filename, outfilename):
    try:
        os.remove(outfilename)
    except:
        pass
    env_new = lmdb.open(
        outfilename,
        subdir=False,
        readonly=False,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=1,
        map_size=int(100000000000.0),
    )
    txn_write = env_new.begin(write=True)
    with open(filename, "r") as input:
        i = 0
        for line in input.readlines():
            line = line.strip()
            if line:
                txn_write.put(f"{i}".encode("ascii"), pickle.dumps(line))
                i += 1
    print("process {} lines".format(i))
    txn_write.commit()
    env_new.close()


if __name__ == "__main__":
    wirte_to_lmbd(sys.argv[1], sys.argv[2])
