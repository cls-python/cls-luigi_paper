import subprocess
from time import sleep

from abc import ABC, abstractmethod
import platform


class LuigiDaemonHandler(ABC):
    
    @abstractmethod
    def start_luigi_daemon(self):
        pass
    
    @abstractmethod
    def kill_luigi_daemon(self):
        pass
    

class LinuxLuigiDaemonHandler(LuigiDaemonHandler):
    
        def __init__(self, logdir="/tmp/"):
            self.logdir = logdir
            self.luigi_process = None
        
        def start_luigi_daemon(self):
            try:    
                self.luigi_process = subprocess.Popen(["luigid", "--background", "--logdir", self.logdir])
                self.luigi_process.wait()
                sleep(0.5)
            except Exception as e:
                raise e
                
        def kill_luigi_daemon(self):
            try:
                self.luigi_process = subprocess.Popen(["pkill", "-f", "luigid"])
                self.luigi_process.wait()
                sleep(0.5)
            except Exception as e:
                raise e
            
class MacOSLuigiDaemonHandler(LuigiDaemonHandler):
    
    def __init__(self, logdir="/tmp/"):
        raise NotImplementedError
    
    
class WindowsLuigiDaemonHandler(LuigiDaemonHandler):
        
        def __init__(self, logdir="/tmp/"):
            raise NotImplementedError
                

class LuigiDaemon:

    def __init__(self, logger=None, logdir="/tmp/"):
        self.logger = logger
        self.logdir = logdir
        self.daemon_handler = self._get_luigi_daemon_handler(self.logdir)
        
        
    def _get_luigi_daemon_handler(self, logdir):
        system = platform.system()

        match system:
            case "Linux":
                return LinuxLuigiDaemonHandler(logdir=logdir)
            case "Darwin":
                return MacOSLuigiDaemonHandler(logdir=logdir)
            case "Windows":
                return WindowsLuigiDaemonHandler(logdir=logdir)
            case _:
                raise NotImplementedError

    def __enter__(self):
        try:
            self.daemon_handler.start_luigi_daemon()
        except Exception as e:
            self.log_or_print(f"Could not start Luigi daemon: {e}")
            raise e

    def __exit__(self, *args):
        try:
            self.daemon_handler.kill_luigi_daemon()
        except Exception as e: 
            self.log_or_print(f"Could not kill Luigi daemon: {e}")
            raise e
        
    def log_or_print(self, msg, level="warning"):
        if self.logger is not None:
            
            if level == "warning":
                self.logger.warning(msg)
            elif level == "error":
                self.logger.error(msg)
            elif level == "info":
                self.logger.info(msg)
                
        else:
            print(msg)
