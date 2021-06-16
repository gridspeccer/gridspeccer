# gridspeccer

*NOTE:* We are currently in the process of transitioning from a "just include the complete script in every paper repository" to a standalone codebase that is not stable yet.

Plotting tool to make plotting with many subfigures easier, especially for publications. 
After installation with `pip install . --local`[1] the `gridspeccer` can be used from the command line on plot scripts:
```
gridspeccer [filename]
```
It can also be used on folder (no argument is equivalent to CWD `.`), in which files that satisfy `fig*.py` are searched.
With the optional argument `--mplrc [file]` one can specify a matplotlibrc to be used for plotting.

A standalone plot file that does not need data is `examples`, this is also used for a unit test (TODO).

[![linting](../../workflows/lint/badge.svg)](../../actions?query=workflow%3Alint)

[![install module](../../workflows/install%20module/badge.svg)](../../actions?query=workflow%3A%22install+module%22)
[![last Release](../../workflows/release_bump/badge.svg)](../../actions?query=workflow%3A%22release_bump%22)
[![gsExample with pseudo tex](../../workflows/gsExample%20with%20pseudo%20tex/badge.svg)](../../actions?query=workflow%3A%22gsExample+with+pseudo+tex%22)
[![gsExample with tex](../../workflows/gsExample%20with%20tex/badge.svg?branch=master)](../../actions?query=workflow%3A%22gsExample+with+tex%22)

Many old examples that are not executable at the moment can be found in `old_examples`, to serve as inspiration for other plots.

### Requirements

* Python 3
* matplotlib
* LaTeX

### Notes
[1] Don't install using `python setup.py install`, as this will create an `.egg`, and the default `matplotlibrc`s will not be accessible.

### Todos
* make true tex standard?
