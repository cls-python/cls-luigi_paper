import os
import subprocess
from time import sleep
import platform
import getpass


class LuigiDaemon:

    def __init__(self, logger=None, logdir=None):
        self.logger = logger
        self.system = platform.system()
        self.logdir = logdir

        if self.logdir is None:

            if self.system == 'Windows':
                self.logdir = f"/Users/{getpass.getuser()}/AppData/Local/Temp/"
            else:
                self.logdir = "/tmp/"

    def __enter__(self):
        if self.system == "Windows":
            subprocess.Popen(
                ["luigid", "--logdir", self.logdir],
            )
        else:
            subprocess.run(["luigid", "--background", "--logdir", self.logdir])

        if self.logger is not None:
            self.logger.warning("Started Luigi daemon...")
        else:
            print("Started Luigi daemon...")
        sleep(1)

    def __exit__(self, *args):
        if self.system == "Windows":
            subprocess.run(["taskkill", "-f", "-im", "luigid.exe"])
        else:
            subprocess.run(["pkill", "-f", "luigid"])
        if self.logger is not None:
            self.logger.warning("Killed Luigi daemon...")
        else:
            print("Killed Luigi daemon...")
        sleep(1)
