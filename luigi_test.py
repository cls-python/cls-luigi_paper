import luigi


class Task1(luigi.Task):
    param1 = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(f'TASK1_{self.param1}.txt')

    def run(self):
        with self.output().open('w') as f:
            f.write(self.param1)


class Task2(luigi.Task):
    param1 = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(f'Task2_{self.param1}.txt')

    def run(self):
        with self.output().open('w') as f:
            f.write(self.param1)

    def requires(self):
        return Task1(param1=self.param1)



if __name__ == '__main__':
    from utils.luigi_daemon import LuigiDaemon

    from configparser import ConfigParser

    for i in range(10):
        # config_object = {}
        #
        # config_object["Task1"] = {
        #     "param1": str(i)
        # }
        #
        # config_object["Task2"] = {
        #     "param1": str(i)
        # }
        #
        # out_string = ""
        #
        # for key, value in config_object.items():
        #     _str = f"[{key}]\n"
        #     for k, v in value.items():
        #         _str += k + "=" + str(v) + "\n"
        #
        #     out_string +=_str
        #
        # with open("luigi.cfg", "w+") as f:
        #     f.write(out_string)

        t = Task2(str(i))

        with LuigiDaemon():
            luigi.build([t], local_scheduler=False)

