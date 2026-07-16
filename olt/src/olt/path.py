from pathlib import Path

from cloudpathlib import CloudPath

RemotePath = Path | CloudPath


def remote_mkdir(p: RemotePath):
    if isinstance(p, Path):
        p.mkdir(exist_ok=True, parents=True)
