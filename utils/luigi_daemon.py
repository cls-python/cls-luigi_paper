import subprocess
from time import sleep


class LuigiDaemon:

    def __init__(self, logger=None):
        self.logger = logger

    def __enter__(self):
        subprocess.run(["luigid", "--background", "--logdir", "~/"])
        if self.logger is not None:
            self.logger.warning("Started Luigi daemon...")
        else:
            print("Started Luigi daemon...")
        sleep(0.5)

    def __exit__(self, *args):
        subprocess.run(["pkill", "-f", "luigid"])
        if self.logger is not None:
            self.logger.warning("Killed Luigi daemon...")
        else:
            print("Killed Luigi daemon...")
        sleep(0.5)
