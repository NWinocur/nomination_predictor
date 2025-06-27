---
trigger: manual
glob:
description:
---

## Open a notebook

!!! note

    The following assumes you're using a Jupyter notebook, but while the specific commands for another notebook tool may look a little bit different, the process guidance still applies.

Now you're ready to do some analysis! Make sure that your project-specific environment is activated (you can check with `which jupyter`) and run `jupyter notebook notebooks` to open a Jupyter notebook in the `notebooks/` folder. You can start by creating a new notebook and doing some exploratory data analysis. We often name notebooks with a scheme that looks like this:

```
0.01-pjb-data-source-1.ipynb
```

 - `0.01` - Helps keep work in chronological order. The structure is `PHASE.NOTEBOOK`. `NOTEBOOK` is just the Nth notebook in that phase to be created. For phases of the project, we generally use a scheme like the following, but you are welcome to design your own conventions:
    - `0` - Data exploration - often just for exploratory work
    - `1` - Data cleaning and feature creation - often writes data to `data/processed` or `data/interim`
    - `2` - Visualizations - often writes publication-ready viz to `reports`
    - `3` - Modeling - training machine learning models
    - `4` - Publication - Notebooks that get turned directly into reports
- `pjb` - Your initials; this is helpful for knowing who created the notebook and prevents collisions from people working in the same notebook.
- `data-source-1` - A description of what the notebook covers

Now that you have your notebook going, start your analysis!

## Refactoring code into shared modules

As your project goes on, you'll want to refactor your code in a way that makes it easy to share between notebooks and scripts. We recommend creating a module in the `{{ cookiecutter.module_name }}` folder that contains the code you use in your project. This is a good way to make sure that you can use the same code in multiple places without having to copy and paste it.

Because the default structure is a Python package and is installed by default, you can do the following to make that code available to you within a Jupyter notebook.

First, we recommend turning on the `autoreload` extension. This will make Jupyter always go back to the source code for the module rather than caching it in memory. If your notebook isn't reflecting the latest changes from your changes to a `.py` file, try restarting the kernel and make sure `autoreload` is on. We add a cell at the top of the notebook with the following:

```
%load_ext autoreload
%autoreload 2
```

Now all your code should be importable. At the start of the CCDS project, you picked a module name. It's the same name as the folder that is in the root project directory. For example, if the module name were `my_project` you could use code by importing it like:

```python
from my_project.data import make_dataset

data = make_dataset()
```

Now it should be easy to do any refactoring you need to do to make your code more modular and reusable.


## Make your code reviewable

We try to review every line of code written at DrivenData. Data science code in particular has the risk of executing without erroring, but not being "correct" (for example, you use standard deviation in a calculation rather than variance). We've found the best way to catch these kinds of mistakes is a second set of eyes looking at the code.

Right now on GitHub, it is hard to observe and comment on changes that happen in Jupyter notebooks. We develop and maintain a tool called [`nbautoexport`](https://nbautoexport.drivendata.org/stable/) that automatically exports a `.py` version of your Jupyter notebook every time you save it. This means that you can commit both the `.ipynb` and the `.py` to source control so that reviewers can leave line-by-line comments on your notebook code. To use it, you will need to add `nbautoexport` to your requirements file and then run `make requirements` to install it.

Once `nbautoexport` is installed, you can setup the nbautoexport tool for your project with the following commands at the commandline:

```
nbautoexport install
nbautoexport configure notebooks
```

Once you're done with your work, you'll want to add it to a commit and push it to GitHub so you can open a pull request. You can do that with the following commandline commands

```
git add .  # stage all changed files to include them in the commit
git commit -m "Initial exploration"  # commit the changes with a message
git push  # publish the changes
```

Now you'll be able to [create a Pull Request in GitHub](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request).