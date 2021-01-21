import os
from typing import List
from .callback import Callback
from molecules.plot import plot_tsne
import concurrent.futures as cf


class TSNEPlotCallback(Callback):
    """
    Saves t-SNE embedding plots.
    """

    def __init__(
        self,
        out_dir: str,
        interval: int = 1,
        colors: List[str] = ["rmsd", "fnc"],
        projection_type: str = "2d",
        target_perplexity: int = 30,
        perplexities: List[int] = [5, 30, 50, 100, 200],
        tsne_is_blocking: bool = False,
        pca: bool = True,
        pca_dim: int = 50,
        backend: str = "mpl",
        wandb_config=None,
        mpi_comm=None,
    ):
        """
        Parameters
        ----------
        out_dir : str
            Directory to store output plots.
        interval : int
            Plots every interval epochs, default is once per epoch.
        backend: str
            Specify plotting backend as `mpl` for matplotlib or `plotly` for plotly.
        wandb_config : wandb configuration file
        mpi_comm: mpi communicator
        """
        super().__init__(interval, mpi_comm)

        if self.is_eval_node:
            os.makedirs(out_dir, exist_ok=True)

            self.tsne_kwargs = {
                "out_dir": out_dir,
                "wandb_config": wandb_config,
                "colors": colors,
                "projection_type": projection_type,
                "target_perplexity": target_perplexity,
                "perplexities": perplexities,
                "pca": pca,
                "pca_dim": pca_dim,
                "backend": backend,
            }

            self.tnse_is_blocking = tsne_is_blocking

            # Need for async plotting
            self.executor = cf.ThreadPoolExecutor(max_workers=2)
            self.future_tsne = None

    def on_epoch_end(self, epoch, logs):
        if self.is_eval_node and (epoch % self.interval == 0):

            # Wait for the old stuff to finish
            if self.future_tsne is not None:
                try:
                    self.future_tsne.result()
                except Exception as exc:
                    print(f"TSNE plot callback generated an exception: {exc}")

            self.future_tsne = self.executor.submit(
                plot_tsne,
                embeddings_path=logs["embeddings_path"],
                global_step=logs["global_step"],
                epoch=epoch,
                **self.tsne_kwargs,
            )
            if self.tnse_is_blocking:
                if self.future_tsne is not None:
                    try:
                        self.future_tsne.result()
                    except Exception as exc:
                        print(f"TSNE plot callback generated an exception: {exc}")
                    self.future_tsne = None

        # All other nodes wait for node 0 to save
        if self.comm is not None:
            self.comm.barrier()

    def on_train_end(self, logs):
        if self.is_eval_node:
            # Wait for the old stuff to finish
            if self.future_tsne is not None:
                self.future_tsne.result()
        if self.comm is not None:
            self.comm.barrier()
