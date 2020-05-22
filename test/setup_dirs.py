import os


def setup_dirs(fpath: str):
    """
    Args:
        fpath:
            The absolute file path of the .py file given by __file__

    Returns:
        indir:
            The input directory
            e.g. path/to/test_something.py --> path/to/test_something

        workdir:
            The working directory
            e.g. path/to/test_something.py --> path/to/workdir

        outdir:
            The output directory
            e.g. path/to/test_something.py --> path/to/outdir
    """

    indir = os.path.relpath(fpath[:-3], '.')
    basedir = os.path.dirname(indir)
    workdir = f'{basedir}/workdir'
    outdir = f'{basedir}/outdir'

    os.makedirs(workdir, exist_ok=True)
    os.makedirs(outdir, exist_ok=True)

    return indir, workdir, outdir
