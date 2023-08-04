# cython_setup.py
import sys, os, time, platform, subprocess
from setuptools import setup, find_packages
from Cython.Build import cythonize
from traceback import format_exc

from distutils.extension import Extension
import numpy

# compile with:
# python setup_cython.py build_ext --inplace

# USAGE - nisam ovo:
#
#   from cython_setup import run
#   run(pyx_path)

# vcvars = r"C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvars32.bat"

# NOTE: to use visual studio 2017 you must have setuptools version 34+
vcvars = r"C:\Program Files (x86)\Microsoft Visual Studio\2017\BuildTools\VC\Auxiliary\Build\vcvars32.bat"


def _build_ext():
    try:
        # pyx_path = sys.argv.pop(-1)
        # pyx_path = os.path.abspath(pyx_path)
        # if not os.path.exists(pyx_path):
        #     raise FileNotFoundError(f"{pyx_path} does not exist")
        # project_name = sys.argv.pop(-1)
        # os.chdir(os.path.abspath(os.path.dirname(pyx_path)))

        extensions = [
            Extension('im2col_cython', ['im2col_cython.pyx'],
                        include_dirs = [numpy.get_include()]
            ),
        ]

        print("cwd: %s" % os.getcwd())
        print(os.path.abspath("build"))
        setup(
            # cmdclass = {'build_ext': build_ext},
            packages=find_packages(),
            # ext_modules=cythonize(extensions)
            ext_modules=cythonize(extensions,
                                  compiler_directives={'language_level': 3, 'infer_types': True, 'binding': False},
                                  annotate=True),
            # include_dirs = [numpy.get_include()]
            build_dir=os.path.abspath("build")
        )
    except:
        input(format_exc())


def retry(func):
    def wrapper(*args, **kw):
        tries = 0
        while True:
            try:
                return func(*args, **kw)
            except Exception:
                tries += 1
                if tries > 4:
                    raise
                time.sleep(0.4)

    return wrapper


@retry
def cleanup(pyx_path):
    from send2trash import send2trash
    c_file = os.path.splitext(pyx_path)[0] + ".c"
    if os.path.exists(c_file):
        os.remove(c_file)

    if os.path.exists("build"):
        send2trash("build")


def move_pyd_files(pyx_path):
    pyx_dir = os.path.dirname(pyx_path)
    build_dir = os.path.join(pyx_dir, "build")
    if not os.path.exists(build_dir):
        raise RuntimeError(f"build_dir {build_dir} did not exist....")
    found_pyd = False
    for top, dirs, nondirs in os.walk(build_dir):
        for name in nondirs:
            if name.lower().endswith(".pyd") or name.lower().endswith(".so"):
                found_pyd = True
                old_path = os.path.join(top, name)
                new_path = os.path.join(pyx_dir, name)
                if os.path.exists(new_path):
                    print(f"removing {new_path}")
                    os.remove(new_path)
                print(f"file created at {new_path}")
                os.rename(old_path, new_path)
    if not found_pyd:
        raise RuntimeError("Never found .pyd file to move")

def run(pyx_path):
    """
    :param pyx_path:
    :type pyx_path:
    :return: this function creates the batch file, which in turn calls this module, which calls cythonize, once done
    the batch script deletes itself... I'm sure theres a less convoluted way of doing this, but it works
    :rtype:
    """
    try:
        project_name = os.path.splitext(os.path.basename(pyx_path))[0]
        run_script(project_name, os.path.abspath(pyx_path))
    except:
        input(format_exc())


def run_script(project_name, pyx_path):
    dirname = os.path.dirname(pyx_path)
    # ------------------------------
    os.chdir(dirname)
    if os.path.exists(vcvars):
        #  raise RuntimeError(
        # f"Could not find vcvars32.bat at {vcvars}\nis Visual Studio Installed?\nIs setuptools version > 34?")
        subprocess.check_call(f'call "{vcvars}"', shell=True)

    cmd = "python" if platform.system() == "Windows" else "python3"
    subprocess.check_call(f'{cmd} "{__file__}" build_ext "{project_name}" "{pyx_path}"', shell=True)
    move_pyd_files(pyx_path)
    cleanup(pyx_path)


if len(sys.argv) > 2:
    _build_ext()