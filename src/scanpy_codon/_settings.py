from time import time
from python import inspect


@dataclass(python=True)
class Verbosity():
    error = 0
    warning = 1
    info = 2
    hint = 3
    debug = 4


@dataclass(python=True)
class ScanpyConfig:
    """\
    Config manager for scanpy.
    """

    N_PCS: int
    """Default number of principal components to use."""
    verbosity: int
    plot_suffix: str
    file_format_data: str
    file_format_figs: str
    autosave: bool
    autoshow: bool
    writedir: str
    cachedir: str
    datasetdir: str
    figdir: str
    cache_compression: str
    max_memory: int
    n_jobs: int
    logfile: str
    categories_to_ignore: List[str]
    _frameon: bool
    _vector_friendly: bool
    _low_resolution_warning: bool
    n_pcs: int

    _start: float
    _previous_time: float
    _previous_memory_usage: int

    def __init__(
        self,
        verbosity: int = Verbosity.warning,
        plot_suffix: str = "",
        file_format_data: str = "h5ad",
        file_format_figs: str = "pdf",
        autosave: bool = False,
        autoshow: bool = True,
        writedir: str = "./write/",
        cachedir: str = "./cache/",
        datasetdir: str = "./data/",
        figdir: str = "./figures/",
        cache_compression: str = "lzf",
        max_memory=15,
        n_jobs=1,
        logfile: str = "",
        categories_to_ignore: List[str] = ["N/A", "dontknow", "no_gate", "?"],
        _frameon: bool = True,
        _vector_friendly: bool = False,
        _low_resolution_warning: bool = True,
        n_pcs=50,
    ):
        # logging
        # self._root_logger = _RootLogger(logging.INFO)  # level will be replaced
        self.logfile = logfile
        self.verbosity = verbosity
        # rest
        self.plot_suffix = plot_suffix
        self.file_format_data = file_format_data
        self.file_format_figs = file_format_figs
        self.autosave = autosave
        self.autoshow = autoshow
        self.writedir = writedir
        self.cachedir = cachedir
        self.datasetdir = datasetdir
        self.figdir = figdir
        self.cache_compression = cache_compression
        self.max_memory = max_memory
        self.n_jobs = n_jobs
        self.categories_to_ignore = categories_to_ignore
        self._frameon = _frameon
        """bool: See set_figure_params."""

        self._vector_friendly = _vector_friendly
        """Set to true if you want to include pngs in svgs and pdfs."""

        self._low_resolution_warning = _low_resolution_warning
        """Print warning when saving a figure with low resolution."""

        self._start = time()
        """Time when the settings module is first imported."""

        self._previous_time = self._start
        """Variable for timing program parts."""

        self._previous_memory_usage = -1
        """Stores the previous memory usage."""

        self.N_PCS = n_pcs

    def set_figure_params(
        self,
        scanpy: bool = True,
        dpi: int = 80,
        dpi_save: int = 150,
        frameon: bool = True,
        vector_friendly: bool = True,
        fontsize: int = 14,
        figsize: int = None,
        color_map: str = None,
        format: str = "pdf",
        facecolor: str = None,
        transparent: bool = False,
        ipython_format: str = "png2x",
    ) -> None:
        if self._is_run_from_ipython():
            from python import IPython

            if isinstance(ipython_format, str):
                ipython_format = [ipython_format]
            IPython.display.set_matplotlib_formats(*ipython_format)

        from python import matplotlib.rcParams

        self._vector_friendly = vector_friendly
        self.file_format_figs = format
        if dpi is not None:
            rcParams["figure.dpi"] = dpi
        if dpi_save is not None:
            rcParams["savefig.dpi"] = dpi_save
        if transparent is not None:
            rcParams["savefig.transparent"] = transparent
        if facecolor is not None:
            rcParams["figure.facecolor"] = facecolor
            rcParams["axes.facecolor"] = facecolor
        if scanpy:
            print("Unimplemented part")
            # from .plotting._rcmod import set_rcParams_scanpy
            # set_rcParams_scanpy(fontsize=fontsize, color_map=color_map)
        if figsize is not None:
            rcParams["figure.figsize"] = figsize
        self._frameon = frameon

    def _is_run_from_ipython():
        from python import builtins
        return getattr(builtins, "__IPYTHON__", False)


settings = ScanpyConfig()
