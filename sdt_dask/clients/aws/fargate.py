from sdt_dask.clients.clients import Clients
from dask_cloudprovider.aws import FargateCluster
from dask.distributed import Client
import logging

logger = logging.getLogger(__name__)


class Fargate(Clients):
    def __init__(
        self,
        image: str = "",
        tags: dict = {},  # optional
        vpc: str = "",
        region_name: str = "",
        environment: dict = {},
        n_workers: int = 10,
        threads_per_worker: int = 2,
    ):
        self.image = image
        self.tags = tags
        self.vpc = vpc
        self.region_name = region_name
        self.environment = environment
        self.n_workers = n_workers
        self.threads_per_worker = threads_per_worker

    def _check_versions(self):
        data = self.client.get_versions(check=True)
        scheduler_pkgs = data["scheduler"]["packages"]
        client_pkgs = data["client"]["packages"]

        for (c_pkg, c_ver), (s_pkg, s_ver) in zip(
            scheduler_pkgs.items(), client_pkgs.items()
        ):
            if c_ver != s_ver:
                msg = "Please Update the client version to match the Scheduler version"
                raise EnvironmentError(
                    f"{c_pkg} version Mismatch:\n\tScheduler: {s_ver} vs Client: {c_ver}\n{msg}"
                )

    def init_client(self) -> tuple:
        logger.info("[i] Initilializing Fargate Cluster ...")

        cluster = FargateCluster(
            tags=self.tags,
            image=self.image,
            vpc=self.vpc,
            region_name=self.region_name,
            environment=self.environment,
            n_workers=self.n_workers,
            worker_nthreads=self.threads_per_worker,
        )

        logger.info("[i] Initialized Fargate Cluster")
        logger.info("[i] Initilializing Dask Client ...")

        self.client = Client(cluster)

        self._check_versions()

        logger.info(f"[>] Dask Dashboard: {self.client.dashboard_link}")

        return self.client
