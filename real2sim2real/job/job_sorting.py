"""Pro-M6 sorting job using the shared finite, synchronized LeRobot v3 collector."""

from pydantic import Field

from real2sim2real.job.job_physic_solver import PhysicSolverJob, PhysicSolverJobConfig


class SortingJobConfig(PhysicSolverJobConfig):
    """One continuous relay episode; physical failures remain diagnostic data."""

    backend: str = "sorting_package"
    output_dir: str = "outputs/sorting"
    timeout_s: float = Field(default=14400, gt=0)


class SortingJob(PhysicSolverJob):
    """Keep the standard Job lifecycle and native station robot placement."""

    config_cls = SortingJobConfig

    def before_place_robot_hook(self) -> None:
        """The Pro-M6 USD already contains its station world pose."""
