import datetime
import os
import time
import json

import torch
import wandb

class Tracker:
    def __init__(self, entity=None, project=None, args=None):
        pass

    def add_histogram(self, tag, data, i):
        pass

    def add_dictionary(self, dict):
        pass

    def add_image(self, tag, value, i):
        pass

    def set_summary(self, key, value):
        pass

    def add_scalar(self, tag, value, i):
        pass

    def log_iteration_time(self, batch_size, i):
        pass

class WandBTracker(Tracker):
    def __init__(self, entity=None, project=None, args=None):
        super().__init__(entity,project,args)
        wandb.init(entity=entity ,project=name, config=args)

    def add_dictionary(self, dict):
        wand.log(dict)

    def add_histogram(self, tag, data, i):
        if type(data) == torch.Tensor:
            data = data.cpu().detach()
        wandb.log({tag: wandb.Histogram(data)}, step=i)

    def add_scalar(self, tag, value, i):
        wandb.log({tag: value}, step=i)

    def add_image(self, tag, value, i):
        wandb.log({tag: [wandb.Image(value, caption="Label")]}, step=i)

    def log_iteration_time(self, batch_size, i):
        """Call this once per training iteration."""
        try:
            dt = time.time() - self.last_time  #noqa
            self.last_time = time.time()
            if i % 10 == 0:
                self.add_scalar("timings/iterations-per-sec", 1/dt, i)
                self.add_scalar("timings/samples-per-sec", batch_size/dt, i)
        except AttributeError:
            self.last_time = time.time()

    def set_summary(self, key, value):
        wandb.run.summary[key] = value


class ConsoleTracker(Tracker):
    def __init__(self, entity=None, project=None, args=None):
        super().__init__(entity,project,args)
        pass

    def add_scalar(self, tag, value, i):
        print(f"{i}  {tag}: {value}")

    def add_dictionary(self, dict):
        print(json.dumps(dict))

    def log_iteration_time(self, batch_size, i):
        """Call this once per training iteration."""
        try:
            dt = time.time() - self.last_time  #noqa
            self.last_time = time.time()
            if i % 10 == 0:
                print(f"{i}  iterations-per-sec: {1/dt}")
                print(f"{i}  samples-per-sec: {batch_size/dt}")
        except AttributeError:
            self.last_time = time.time()

    def set_summary(self, key, value):
        print(f"{key}: {value}")
