# -*- coding: utf-8 -*-
# author: kuangdd
# date: 2020/8/13
"""
__init__
"""
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(Path(__name__).stem)


if __name__ == "__main__":
    print(__file__)