# gridspeccer

Plotting tool to make plotting with many subfigures easier, especially for publications. 
After installation with `python setup.py install --local` the `gridspeccer` can be used from the command line on plot scripts:
```
gridspeccer [filename]
```
It can also be used on folder (no argument is equivalent to CWD `.`), in which files that satisfy `fig*.py` are searched.
With the optional argument `--mplrc [file]` one can specify a matplotlibrc to be used for plotting.

A standalone plot file that does not need data is `examples`, this is also used for a unit test (TODO).

Many old examples that are not executable at the moment can be found in `old_examples`, to serve as inspiration for other plots.
