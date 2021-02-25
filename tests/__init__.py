import os

# libgomp issue, must import n2 before torch. See: https://github.com/kakao/n2/issues/42
import n2  # noqa: F401
import torch

dtypes_from_environ = os.environ.get("TEST_DTYPES", "float16,float32,float64").split(",")
device_from_environ = os.environ.get("TEST_DEVICE", "cuda")

TEST_DTYPES = [getattr(torch, x) for x in dtypes_from_environ]
TEST_DEVICE = torch.device(device_from_environ)
