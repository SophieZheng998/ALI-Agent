import os
import torch


class abstract_arena:
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.llm_name = args.llm_name
        self.device = torch.device(args.cuda)