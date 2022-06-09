import sys
from collections.abc import Iterable
from multiprocessing import Pool
from shutil import get_terminal_size

from mmcv.utils import Timer

class Progress_ui_show:
    """A progress bar which can print the progress."""

    def __init__(self, task_num=0, bar_width=50, start=True, file=sys.stdout):
        self.task_num = task_num
        self.bar_width = bar_width
        self.completed = 0
        self.file = file
        #if start:
        #    self.start()

    @property
    def terminal_width(self):
        width, _ = get_terminal_size()
        return width

    def start(self):
        info = ' '
        if self.task_num > 0:
            self.file.write(f'[{" " * self.bar_width}] 0/{self.task_num}, '
                            'elapsed: 0s, ETA:')

            info = f'0/{self.task_num}, ' + 'elapsed: 0s'
        else:
            self.file.write('completed: 0, elapsed: 0s')
            info = 'completed: 0, elapsed: 0s'
        self.file.flush()
        self.timer = Timer()
        return info

    def update(self, num_tasks=1):
        assert num_tasks > 0
        info = ''
        self.completed += num_tasks
        elapsed = self.timer.since_start()
        infer_time = int(elapsed + 0.5)
        if elapsed > 0:
            fps = self.completed / elapsed
        else:
            fps = float('inf')
        if self.task_num > 0:
            percentage = self.completed / float(self.task_num)
            eta = int(elapsed * (1 - percentage) / percentage + 0.5)
            msg = f'\r[{{}}] {self.completed}/{self.task_num}, ' \
                  f'{fps:.1f} task/s, elapsed: {int(elapsed + 0.5)}s, ' \
                  f'ETA: {eta:5}s'

            bar_width = min(self.bar_width,
                            int(self.terminal_width - len(msg)) + 2,
                            int(self.terminal_width * 0.6))
            bar_width = max(2, bar_width)
            mark_width = int(bar_width * percentage)
            bar_chars = '>' * mark_width + ' ' * (bar_width - mark_width)
            self.file.write(msg.format(bar_chars))
            # ui show
            info = f'{self.completed}/{self.task_num}, ' \
                  f'{fps:.1f} task/s, elapsed: {int(elapsed + 0.5)}s, ' \
                  f'ETA: {eta:5}s'
        else:
            self.file.write(
                f'completed: {self.completed}, elapsed: {int(elapsed + 0.5)}s,'
                f' {fps:.1f} tasks/s')

            msg = f'completed: {self.completed}, elapsed: {int(elapsed + 0.5)}s,' \
                f' {fps:.1f} tasks/s'
            # ui show
            info = msg
        self.file.flush()
        if self.completed == self.task_num:
            return [info, str(infer_time)]
        else:
            return info