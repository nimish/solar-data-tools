"""
TODO: Change documentation to sphinx format
"""
from pprint import pprint
import os
import platform
import psutil
import dask.config
from dask.distributed import Client
from sdt_dask.clients.clients import Clients
import logging

logger = logging.getLogger(__name__)


class Local(Clients):
    def __init__(
        self,
        n_workers: int = 2,
        threads_per_worker: int = 2,
        memory_per_worker: int = 5,
        verbose: bool = False,
    ):
        self.verbose = verbose
        self.n_workers = n_workers
        self.threads_per_worker = threads_per_worker
        self.memory_per_worker = memory_per_worker
        self.dask_config = dask.config

    def _get_sys_var(self):
        self.system = platform.system().lower()
        self.cpu_count = os.cpu_count()
        self.memory = psutil.virtual_memory().total // (1024.0**3)

    def _config_init(self):
        tmp_dir = dask.config.get("temporary_directory")
        if not tmp_dir:
            self.dask_config.update(
                {
                    "distributed.worker.memory.spill": False,
                    "distributed.worker.memory.pause": False,
                    "distributed.worker.memory.target": 0.8,
                    "distributed.worker.memory.terminate": False,
                }
            )

    def _check(self):
        self._get_sys_var()
        # workers and threads need to be less than cpu core count
        # memory per worker >= 5 GB but total memory use should be less than the system memory available
        if self.n_workers * self.threads_per_worker > self.cpu_count:
            raise ValueError(
                f"workers and threads exceed local resources, {self.cpu_count} cores present"
            )
        elif self.memory_per_worker < 5:
            raise ValueError(
                "memory per worker too small, minimum memory size per worker 5 GB"
            )
        if self.n_workers * self.memory_per_worker > self.memory:
            self.dask_config.set({"distributed.worker.memory.spill": True})
            print(
                f"[!] memory per worker exceeds system memory ({self.memory} GB), activating memory spill fraction"
            )

    def init_client(self) -> Client:
        self._config_init()
        self._check()
        self.client = Client(
            processes=True,
            n_workers=self.n_workers,
            threads_per_worker=self.threads_per_worker,
            memory_limit=f"{self.memory_per_worker:.2f}GiB",
        )

        if self.verbose:
            logger.info(f"[i] System: {self.system}")
            logger.info(f"[i] CPU Count: {self.cpu_count}")
            logger.info(f"[i] System Memory: {self.memory}")
            logger.info(f"[i] Workers: {self.n_workers}")
            logger.info(f"[i] Threads per Worker: {self.threads_per_worker}")
            logger.info(f"[i] Memory per Worker: {self.memory_per_worker}")
            logger.info("[i] Dask worker config:")
            logger.info(self.dask_config.get("distributed.worker"))

        logger.info(f"\n[>] Dask Dashboard: {self.client.dashboard_link}\n")

        return self.client
