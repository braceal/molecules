import os
import time
import wandb
from typing import List
from PIL import Image
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from molecules.data.utils import parse_h5


def compute_tsne(
    data: np.ndarray,
    n_components: int = 3,
    n_jobs: int = 4,
    perplexity: float = 30.0,
    backend: str = "sklearn",
) -> np.ndarray:
    r"""Run tsne on `data`."""
    if backend == "sklearn":
        from sklearn.manifold import TSNE

        tsne = TSNE(n_components=n_components, n_jobs=n_jobs, perplexity=perplexity)
    elif backend == "cuml":
        from cuml.manifold import TSNE

        tsne = TSNE(n_components=n_components, method="exact", perplexity=perplexity)
    else:
        raise ValueError(f"TSNE backend {backend} not supported.")

    tsne_embeddings = tsne.fit_transform(data)
    return tsne_embeddings


def compute_pca(embeddings: np.ndarray, dim: int = 50) -> np.ndarray:
    # TODO: use pca to drop embeddings to dim 50
    # TODO: run PCA in pytorch and reduce dimension down to 50 (maybe even lower)
    #       then run tSNE on outputs of PCA. This works for sparse matrices
    #       https://pytorch.org/docs/master/generated/torch.pca_lowrank.html
    return embeddings


def plot_tsne(
    embeddings_path: str,
    out_dir: str = "./",
    colors: List[str] = ["rmsd"],
    pca: bool = True,
    projection_type: str = "2d",
    target_perplexity: int = 30,
    embeddings_dset_name: str = "embeddings",
    perplexities: List[int] = [5, 30, 50, 100, 200],
    pca_dim: int = 50,
    plot_backend: str = "mpl",
    outlier_inds=None,
    wandb_config=None,
    global_step=0,
    epoch=1,
):
    """
    Parameters
    ----------
    plot_backend: str
            Specify plotting backend as `mpl` for matplotlib or `plotly` for plotly.
    """

    color_arrays = parse_h5(embeddings_path, fields=colors + [embeddings_dset_name])
    embeddings = color_arrays.pop(embeddings_dset_name)

    if pca and embeddings.shape[1] > pca_dim:
        embeddings = compute_pca(embeddings, pca_dim)

    if plot_backend == "plotly":
        from plotly.io import to_html

        tsne_embeddings = compute_tsne(embeddings)
        fig = plot_tsne_plotly(tsne_embeddings, df_dict=color_arrays, color=colors[0])
        html_string = to_html(fig)
        if wandb_config is not None:
            wandb.log(
                {"t-SNE interactive scatter": wandb.Html(html_string, inject=False)},
                step=global_step,
            )
        time_stamp = time.strftime(
            f"t-SNE-plotly-{colors[0]}-epoch-{epoch}-%Y%m%d-%H%M%S.html"
        )
        with open(os.path.join(out_dir, time_stamp), "w") as f:
            f.write(html_string)
        return

    # create plot grid
    nrows = len(perplexities)
    ncols = 3 if projection_type == "3d" else 1

    # If outliers are plotted, make them prominent
    alpha = 0.3 if outlier_inds is not None else None

    # Precompute tsne embeddings for each perplexity
    tsne_embeddings = []
    for perplexity in perplexities:
        tsne_embed = compute_tsne(
            embeddings,
            n_components=int(projection_type[0]),
            n_jobs=4,
            perplexity=perplexity,
        )
        tsne_embeddings.append(tsne_embed)

    for color_name, color_arr in color_arrays.items():

        # create colormaps
        cmi = plt.get_cmap("jet")
        vmin, vmax = np.min(color_arr), np.max(color_arr)
        cnorm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        scalar_map = matplotlib.cm.ScalarMappable(norm=cnorm, cmap=cmi)
        scalar_map.set_array(color_arr)

        # create figure
        fig, axs = plt.subplots(
            figsize=(ncols * 4, nrows * 4), nrows=nrows, ncols=ncols
        )

        # set up constants
        color = scalar_map.to_rgba(color_arr)
        if color_name == "rmsd":
            titlestring = f"RMSD to reference state after epoch {epoch}"
        elif color_name == "fnc":
            titlestring = f"Fraction of contacts to reference state after epoch {epoch}"

        for idr, (perplexity, emb_trans) in enumerate(
            zip(perplexities, tsne_embeddings)
        ):

            # plot
            if projection_type == "3d":
                z1, z2, z3 = emb_trans[:, 0], emb_trans[:, 1], emb_trans[:, 2]
                z1mm = np.min(z1), np.max(z1)
                z2mm = np.min(z2), np.max(z2)
                z3mm = np.min(z3), np.max(z3)
                z1mm = (z1mm[0] * 0.95, z1mm[1] * 1.05)
                z2mm = (z2mm[0] * 0.95, z2mm[1] * 1.05)
                z3mm = (z3mm[0] * 0.95, z3mm[1] * 1.05)
                # x-y
                ax1 = axs[idr, 0]
                ax1.scatter(z1, z2, marker=".", c=color, alpha=alpha)
                ax1.set_xlim(z1mm)
                ax1.set_ylim(z2mm)
                ax1.set_xlabel(r"$z_1$")
                ax1.set_ylabel(r"$z_2$")
                # x-z
                ax2 = axs[idr, 1]
                ax2.scatter(z1, z3, marker=".", c=color, alpha=alpha)
                ax2.set_xlim(z1mm)
                ax2.set_ylim(z3mm)
                ax2.set_xlabel(r"$z_1$")
                ax2.set_ylabel(r"$z_3$")
                if idr == 0:
                    ax2.set_title(titlestring)
                # y-z
                ax3 = axs[idr, 2]
                ax3.scatter(z2, z3, marker=".", c=color, alpha=alpha)
                ax3.set_xlim(z2mm)
                ax3.set_ylim(z3mm)
                ax3.set_xlabel(r"$z_2$")
                ax3.set_ylabel(r"$z_3$")
                # colorbar
                divider = make_axes_locatable(axs[idr, 2])
                cax = divider.append_axes("right", size="5%", pad=0.1)
                fig.colorbar(scalar_map, ax=axs[idr, 2], cax=cax)

                if outlier_inds is not None:
                    # Plot outliers as diamonds with no transparency
                    outlier_kwargs = {"marker": "D", "color": color[outlier_inds]}
                    ax1.scatter(z1[outlier_inds], z2[outlier_inds], **outlier_kwargs)
                    ax2.scatter(z1[outlier_inds], z3[outlier_inds], **outlier_kwargs)
                    ax3.scatter(z2[outlier_inds], z3[outlier_inds], **outlier_kwargs)

            else:
                ax = axs[idr]
                z1, z2 = emb_trans[:, 0], emb_trans[:, 1]
                ax.scatter(z1, z2, marker=".", c=color, alpha=alpha)
                z1mm = np.min(z1), np.max(z1)
                z2mm = np.min(z2), np.max(z2)
                z1mm = (z1mm[0] * 0.95, z1mm[1] * 1.05)
                z2mm = (z2mm[0] * 0.95, z2mm[1] * 1.05)
                ax.set_xlim(z1mm)
                ax.set_ylim(z2mm)
                ax.set_xlabel(r"$z_1$")
                ax.set_ylabel(r"$z_2$")
                if idr == 0:
                    ax.set_title(titlestring)
                # colorbar
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.1)
                fig.colorbar(scalar_map, ax=axs, cax=cax)

                if outlier_inds is not None:
                    # Plot outliers as diamonds with no transparency
                    outlier_kwargs = {"marker": "D", "color": color[outlier_inds]}
                    ax.scatter(z1[outlier_inds], z2[outlier_inds], **outlier_kwargs)

            # plot as 3D object on wandb
            if (wandb_config is not None) and (perplexity == target_perplexity):
                point_data = np.concatenate([emb_trans, color[:, :3] * 255.0], axis=1)
                caption = f"perplexity {perplexity} color {color_name}"
                wandb.log(
                    {
                        f"3D t-SNE embeddings {color_name} paint": wandb.Object3D(
                            point_data, caption=caption
                        )
                    },
                    step=global_step,
                )

        # tight layout
        plt.tight_layout()

        # save figure
        time_stamp = time.strftime(
            f"2d-embeddings-{color_name}-epoch-{epoch}-%Y%m%d-%H%M%S.png"
        )
        plt.savefig(os.path.join(out_dir, time_stamp), dpi=300)

        # wandb logging
        if wandb_config is not None:
            img = Image.open(os.path.join(out_dir, time_stamp))
            wandb.log(
                {
                    f"2D t-SNE embeddings {color_name} paint": [
                        wandb.Image(img, caption="Latent Space Visualizations")
                    ]
                },
                step=global_step,
            )

        # close plot
        plt.close(fig)


def plot_tsne_publication(
    embeddings_path,
    out_dir="./",
    colors=["rmsd"],
    embeddings_dset_name: str = "embeddings",
    pca=False,
    pca_dim=50,
    wandb_config=None,
    global_step=0,
    epoch=1,
):
    """Generate publication quality 3d t-SNE plot."""

    color_arrays = parse_h5(embeddings_path, fields=colors + [embeddings_dset_name])
    embeddings = color_arrays.pop(embeddings_dset_name)

    if pca and embeddings.shape[1] > 50:
        embeddings = pca(embeddings, pca_dim)

    embeddings = compute_tsne(embeddings, n_components=3, n_jobs=4)

    z1, z2, z3 = embeddings[:, 0], embeddings[:, 1], embeddings[:, 2]
    z1_min_max = np.min(z1), np.max(z1)
    z2_min_max = np.min(z2), np.max(z2)
    z3_min_max = np.min(z3), np.max(z3)

    # TODO: make grid plot of fnc, rmsd
    for color_name, color_arr in color_arrays.items():

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        cnorm = matplotlib.colors.Normalize(
            vmin=np.min(color_arr), vmax=np.max(color_arr)
        )
        scalar_map = matplotlib.cm.ScalarMappable(norm=cnorm, cmap=plt.get_cmap("jet"))
        scalar_map.set_array(color_arr)
        fig.colorbar(scalar_map)
        color = scalar_map.to_rgba(color_arr)

        ax.scatter3D(z1, z2, z3, marker=".", c=color)
        ax.set_xlim3d(z1_min_max)
        ax.set_ylim3d(z2_min_max)
        ax.set_zlim3d(z3_min_max)
        ax.set_xlabel(r"$z_1$")
        ax.set_ylabel(r"$z_2$")
        ax.set_zlabel(r"$z_3$")

        if color_name == "rmsd":
            ax.set_title(f"RMSD to reference state after epoch {epoch}")
        elif color_name == "fnc":
            ax.set_title(f"Fraction of contacts to reference state after epoch {epoch}")

        time_stamp = time.strftime(
            f"3d-embeddings-{color_name}-epoch-{epoch}-%Y%m%d-%H%M%S.png"
        )
        plt.savefig(os.path.join(out_dir, time_stamp), dpi=300)

        if wandb_config is not None:
            img = Image.open(os.path.join(out_dir, time_stamp))
            wandb.log(
                {
                    f"3D xyz t-SNE embeddings {color_name} paint": [
                        wandb.Image(img, caption="Latent Space Visualizations")
                    ]
                },
                step=global_step,
            )

        ax.clear()
        plt.close(fig)


def plot_tsne_plotly(tsne_embeddings, df_dict={}, color=None):

    import pandas as pd
    import plotly.express as px

    for i, name in enumerate(["x", "y", "z"]):
        df_dict[name] = tsne_embeddings[:, i]

    embeddings_df = pd.DataFrame(df_dict)

    fig = px.scatter_3d(
        embeddings_df,
        x="x",
        y="y",
        z="z",
        color=color,
        width=1000,
        height=1000,
        size_max=7,
        hover_data=list(df_dict.keys()),
    )
    return fig


if __name__ == "__main__":

    # data1 = np.random.normal(size=(100, 6))
    # data2 = np.random.normal(size=(100, 6), loc=10, scale=2)
    # data3 = np.random.normal(size=(100, 6), loc=5, scale=1)
    # data = np.concatenate((data1, data2, data3))
    # rmsd = np.random.normal(size=300)
    # fnc = np.random.normal(size=300)

    # from molecules.utils import open_h5
    # scaler_kwargs = {'fletcher32': True}
    # with open_h5('tmpdir/test_embed.h5', 'w', swmr=False) as h5_file:
    #     h5_file.create_dataset('embeddings', data=data, **scaler_kwargs)
    #     h5_file.create_dataset('rmsd', data=rmsd, **scaler_kwargs)
    #     h5_file.create_dataset('fnc', data=fnc, **scaler_kwargs)

    # outlier_inds = [1,2,3,4,5]

    # plot_tsne('tmpdir/test_embed.h5', './tmpdir',
    #           projection_type='3d', colors=['rmsd'],
    #           outlier_inds=outlier_inds)

    # data_dir="/gpfs/alpine/med110/proj-shared/tkurth/runs/cmaps-3clpro-summit-run-1/model-cmaps-3clpro-summit-run-1/embedddings"
    # embedding_file="embeddings-raw-step-724-20200918-130640.h5"

    # data_dir="/gpfs/alpine/med110/proj-shared/tkurth/runs/cmaps-3clpro-summit-run-2-nnodes4/model-cmaps-3clpro-summit-run-2-nnodes4/embedddings"
    # embedding_file="embeddings-raw-step-1343-20200918-153515.h5"

    # data_dir="/gpfs/alpine/med110/proj-shared/tkurth/runs/cmaps-3clpro-summit-run-2-nnodes1/model-cmaps-3clpro-summit-run-2-nnodes1/embedddings"
    # embedding_file="runs/model-3clpro-aae-test_bs-32_opt_Adam_lr-1e-4_latentdim-48_cutoff-16/embedddings/embeddings-raw-step-37499-20200925-213727.h5"

    # # concat
    # embedding_input=os.path.join(data_dir, embedding_file)

    # outlier_inds = np.arange(200)

    # plot_tsne(embedding_file, colors=['rmsd', 'fnc'], projection_type='3d',
    #          outlier_inds=outlier_inds)
    pass
