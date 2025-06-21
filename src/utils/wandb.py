def wandb_only(func):
    def wrapper(self, *args, **kwargs):
        if self.logger is None:
            return
        return func(self, *args, **kwargs)
    return wrapper
