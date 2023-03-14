
Contributing to watex 
=======================

General support
----------------

General support queries ("how do I do X?") go to [StackOverflow], which has a bigger audience 
of individuals who will notice your post and may be able to help. Include runnable code, a 
specific definition of what you want to accomplish, and a detailed explanation of the challenges 
that you have experienced to increase your chances of receiving a timely response.

Bug reporting
---------------

If you believe you have found a problem in watex, please file a bug report on 
the [Github issue tracker](https://github.com/WEgeophysics/watex/issues/new). Bug reports must contain the 
following information in order to be useful:

- A reproducible code sample demonstrating the issue
- The output that you are viewing (a plot picture or an error message) - A detailed explanation of why you 
  believe something is incorrect - The versions of watex and matplotlib with which you are working

Bug reports are easier to handle if they can be proven using one of the watex documentation' 
example datasets (i.e. ``watex.datasets.load_XXX`` (``XXX`` refers to the datasets, i.e. ``watex.datasets.load_tankesse``). 
Instead, your example should create synthetic data to replicate the issue. If you can only show the problem 
with your real dataset, you must provide it, preferably as a csv. You may directly submit a csv to a github 
issue thread, however it must have a '.txt' suffix.

If you come across an error, examining the particular wording of the notice before posting a new 
issue will frequently help you address the problem quickly and prevent creating a duplicate report.


New features
-------------

If you believe a new feature should be added to ``watex``, you may create an issue to discuss it. Nevertheless, 
please keep in mind that present development efforts are primarily focused on standardizing the API and internals, 
and there may be limited excitement for unique features that do not fit well into short-and medium-term development 
plans.

Development  
-------------

For the development, improve/adding new algorithms, please refer to [Development guide](https://watex.readthedocs.io/en/latest/development.html). 
to follow the ``watex API`` recommended syntaxes. 
