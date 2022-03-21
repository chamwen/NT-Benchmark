# -*- coding: utf-8 -*-
# A Survey on Negative Transfer
# https://github.com/chamwen/NT-Benchmark
import os.path as osp
from datetime import datetime
from datetime import timedelta, timezone
from utils.utils import create_folder


class LogRecord:
    def __init__(self, args):
        self.args = args
        self.result_dir = args.result_dir
        self.data_env = 'gpu'
        self.data_name = args.dset
        self.method = args.method

    def log_init(self):
        create_folder(self.result_dir, self.args.data_env, self.args.local_dir)

        if self.data_env == 'local':
            time_str = datetime.utcnow().replace(tzinfo=timezone.utc).astimezone(
                timezone(timedelta(hours=8), name='Asia/Shanghai')).strftime("%Y-%m-%d_%H_%M_%S")
        if self.data_env == 'gpu':
            time_str = datetime.utcnow().replace(tzinfo=timezone.utc).strftime("%Y-%m-%d_%H_%M_%S")
        file_name_head = 'log_' + self.method + '_' + self.data_name + '_'
        self.args.out_file = open(osp.join(self.args.result_dir, file_name_head + time_str + '.txt'), 'w')
        self.args.out_file.write(self._print_args() + '\n')
        self.args.out_file.flush()
        return self.args

    def record(self, log_str):
        self.args.out_file.write(log_str + '\n')
        self.args.out_file.flush()
        return self.args

    def _print_args(self):
        s = "==========================================\n"
        for arg, content in self.args.__dict__.items():
            s += "{}:{}\n".format(arg, content)
        return s
