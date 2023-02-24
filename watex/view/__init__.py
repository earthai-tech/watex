""" 
View is the visualization sub-package. It is divised into the learning plot
(:mod:`~watex.view.mlplot`) and, data analysis and exploratory modules 
(:mod:`~watex.view.plot`).
"""

from .mlplot import ( 
    pobj,
    biPlot, 
    EvalPlot, 
    plotLearningInspection, 
    plotLearningInspections, 
    plotSilhouette,
    plotDendrogram, 
    plotDendroheat, 
    plotProjection, 
    plotModel, 
    plot_reg_scoring, 
    plot_matshow, 
    plot_model_scores, 
    plot2d,
    )
from .plot import ( 
    QuickPlot , 
    ExPlot,
    TPlot, 
    viewtemplate, 
    )

__all__=[
    "pobj",
    "biPlot", 
    "EvalPlot", 
    "QuickPlot" , 
    "ExPlot",
    "TPlot",
    "plotLearningInspection", 
    "plotLearningInspections", 
    "plotSilhouette", 
    "plotDendrogram", 
    "plotDendroheat", 
    "viewtemplate", 
    "plotProjection", 
    "plotModel", 
    "plot_reg_scoring", 
    "plot_matshow", 
    "plot_model_scores", 
    "plot2d"
    ]