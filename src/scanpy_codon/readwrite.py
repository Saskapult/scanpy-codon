from python import h5py
from python import numpy as np
from python import anndata
from python import pathlib

import logging as logg


@python
def _download(url: str, path):
    try:
        import ipywidgets  # noqa: F401
        from tqdm.auto import tqdm
    except ImportError:
        from tqdm import tqdm

    from urllib.error import URLError
    from urllib.request import Request, urlopen

    blocksize = 1024 * 8
    blocknum = 0

    try:
        req = Request(url, headers={"User-agent": "scanpy-user"})

        try:
            open_url = urlopen(req)
        except URLError:
            logg.warning(
                "Failed to open the url with default certificates, trying with certifi."
            )

            from ssl import create_default_context

            from certifi import where

            open_url = urlopen(req, context=create_default_context(cafile=where()))

        with open_url as resp:
            total = resp.info().get("content-length", None)
            with (
                tqdm(
                    unit="B",
                    unit_scale=True,
                    miniters=1,
                    unit_divisor=1024,
                    total=total if total is None else int(total),
                ) as t,
                path.open("wb") as f,
            ):
                block = resp.read(blocksize)
                while block:
                    f.write(block)
                    blocknum += 1
                    t.update(len(block))
                    block = resp.read(blocksize)

    except (KeyboardInterrupt, Exception):
        # Make sure file doesn’t exist half-downloaded
        if path.is_file():
            path.unlink()
        raise


def _check_datafile_present_and_download(path, backup_url=None):
    """Check whether the file is present, otherwise download."""
    path = pathlib.Path(path)
    if path.is_file():
        return True
    if backup_url is None:
        return False
    logg.info(
        f"try downloading from url\n{backup_url}\n"
        "... this may take a while but only happens once"
    )
    if not path.parent.is_dir():
        logg.info(f"creating directory {path.parent}/ for saving data")
        path.parent.mkdir(parents=True)

    _download(backup_url, path)
    return True


def _collect_datasets(dsets: dict, group):
    for k, v in group.items():
        if isinstance(v, h5py.Dataset):
            dsets[k] = v[()]
        else:
            _collect_datasets(dsets, v)


@python
def _read_v3_10x_h5(filename: str):
    import h5py

    def _collect_datasets(dsets: dict, group):
        for k, v in group.items():
            if isinstance(v, h5py.Dataset):
                dsets[k] = v[()]
            else:
                _collect_datasets(dsets, v)

    with h5py.File(str(filename), "r") as f:
        try:
            dsets = {}
            _collect_datasets(dsets, f["matrix"])

            # from python import scipy.sparse
            import scipy.sparse as sparse
            import anndata

            M, N = dsets["shape"]
            data = dsets["data"]
            if dsets["data"].dtype == np.dtype("int32"):
                data = dsets["data"].view("float32")
                data[:] = dsets["data"]
            matrix = sparse.csr_matrix(
                (data, dsets["indices"], dsets["indptr"]),
                shape=(N, M),
            )
            obs_dict = {"obs_names": dsets["barcodes"].astype(str)}
            var_dict = {"var_names": dsets["name"].astype(str)}

            if "gene_id" not in dsets:
                # Read metadata specific to a feature-barcode matrix
                var_dict["gene_ids"] = dsets["id"].astype(str)
            else:
                # Read metadata specific to a probe-barcode matrix
                var_dict.update(
                    {
                        "gene_ids": dsets["gene_id"].astype(str),
                        "probe_ids": dsets["id"].astype(str),
                    }
                )
            var_dict["feature_types"] = dsets["feature_type"].astype(str)
            if "filtered_barcodes" in f["matrix"]:
                obs_dict["filtered_barcodes"] = dsets["filtered_barcodes"].astype(bool)

            if "features" in f["matrix"]:
                var_dict.update(
                    (
                        feature_metadata_name,
                        dsets[feature_metadata_name].astype(
                            bool# if feature_metadata_item.dtype.kind == "b" else str
                        ),
                    )
                    for feature_metadata_name, feature_metadata_item in f["matrix"][
                        "features"
                    ].items()
                    if isinstance(feature_metadata_item, h5py.Dataset)
                    and feature_metadata_name
                    not in [
                        "name",
                        "feature_type",
                        "id",
                        "gene_id",
                        "_all_tag_keys",
                    ]
                )
            else:
                raise ValueError("10x h5 has no features group")
            adata = anndata.AnnData(
                matrix,
                obs=obs_dict,
                var=var_dict,
            )
            # logg.info("", time=start)
            return adata
        except KeyError:
            raise Exception("File is missing one or more required datasets.")


def read_10x_h5(
    filename: str,
    # genome: str = None,
    # gex_only: bool = True,
    # backup_url: str = None,
):
    print("Ladder 2")
    # start = logg.info(f"reading {filename}")
    is_present = _check_datafile_present_and_download(filename)
    if not is_present:
        logg.debug(f"... did not find original file {filename}")
    with h5py.File(str(filename), "r") as f:
        v3 = "/matrix" in f
    adata = _read_v3_10x_h5(filename)
    # if genome:
    #     if genome not in adata.var["genome"].values:
    #         raise ValueError(
    #             f"Could not find data corresponding to genome '{genome}' in '{filename}'. "
    #             f'Available genomes are: {list(adata.var["genome"].unique())}.'
    #         )
    #     adata = adata[:, adata.var["genome"] == genome]
    # if gex_only:
    #     adata = adata[:, adata.var["feature_types"] == "Gene Expression"]
    if adata.is_view:
        adata = adata.copy()
    return adata
