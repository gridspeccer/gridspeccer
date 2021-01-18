# gridspeccer

Plotting tool to make plotting with many subfigures easier, especially for publications. 
After installation with `pip install . --local`[1] the `gridspeccer` can be used from the command line on plot scripts:
```
gridspeccer [filename]
```
It can also be used on folder (no argument is equivalent to CWD `.`), in which files that satisfy `fig*.py` are searched.
With the optional argument `--mplrc [file]` one can specify a matplotlibrc to be used for plotting.

A standalone plot file that does not need data is `examples`, this is also used for a unit test (TODO).

![install module](https://github.com/JulianGoeltz/gridspeccer/workflows/install%20module/badge.svg)
![gsExample with pseudo tex](https://github.com/JulianGoeltz/gridspeccer/workflows/gsExample%20with%20pseudo%20tex/badge.svg)
![gsExample with tex](https://github.com/JulianGoeltz/gridspeccer/workflows/gsExample%20with%20tex/badge.svg?branch=master)

Many old examples that are not executable at the moment can be found in `old_examples`, to serve as inspiration for other plots.


### Notes
[1] Don't install using `python setup.py install`, as this will create an `.egg`, and the default `matplotlibrc`s will not be accessible.

### Todos
* make true tex standard?
