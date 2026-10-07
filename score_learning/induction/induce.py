# -*- coding: utf-8 -*-
"""入口转到 RAS-FR（分解残差），避免走项集/lift 那条非原创路径。"""
from score_learning.induction.fri import main

if __name__ == '__main__':
    main()
