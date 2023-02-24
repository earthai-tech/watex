# -*- coding: utf-8 -*-
#   Licence:BSD 3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
#   Created date: Fri Apr 15 10:46:56 2022

"""
:code:`watex` property objects. It is composed of base classes that are inherited 
by methods implemented throughout the package. It also inferred properties to 
data objects. 

.. _WATex: https://github.com/WEgeophysics/watex/
.. |ERP| replace:: Electrical resistivity profiling 
.. |VES| replace:: Vertical Electrical Sounding 
.. _interpol_imshow: https://matplotlib.org/stable/gallery/images_contours_and_fields/interpolation_methods.html

"""   

# import warnings 
from __future__ import annotations 
import os 
from abc import ( 
    ABC, 
    abstractmethod, 
    )
import pandas as pd 

from .decorators import refAppender 
from ._docstring import refglossary  
from .exceptions import ( 
    FileHandlingError, 
    EDIError 
    )

__all__ = [ 
    "Water",
    "BasePlot", 
    'P',
    'BagoueNotes',
    "ElectricalMethods", 
    "IsEdi",
    "Config", 
    "UTM_DESIGNATOR", 
    "_EDI", 
    "_ZRP_COMPS", 
    "_TIP_COMPS", 
    "Software", 
    "Copyright", 
    "References", 
    "Person", 
]


UTM_DESIGNATOR ={
    'X':[72,84], 
    'W':[64,72], 
    'V':[56,64],
    'U':[48,56],
    'T':[40,48],
    'S':[32,40], 
    'R':[24,32], 
    'Q':[16,24], 
    'P':[8,16],
    'N':[0,8], 
    'M':[-8,0],
    'L':[-16, 8], 
    'K':[-24,-16],
    'J':[-32,-24],
    'H':[-40,-32],
    'G':[-48,-40], 
    'F':[-56,-48],
    'E':[-64, -56],
    'D':[-72,-64], 
    'C':[-80,-72],
    'Z':[-80,84]
}
    
_EDI =[
    #Head=Infos-Frequency-Rhorot
    # Zrot and end blocks
    '>HEAD','>INFO',
    
    #Definitions Measurments Blocks
    '>=DEFINEMEAS',
    '>=MTSECT',
    '>FREQ ORDER=INC', 
    '>ZROT',
    '>RHOROT',
    '>!Comment',
    '>FREQ',
    '>HMEAS',
    '>EMEAS',
    
    #Apparents Resistivities
    #and Phase Blocks
    '>RHOXY','>PHSXY',
    '>RHOXY.VAR','>PHSXY.VAR',
    '>RHOXY.ERR','>PHSXY.ERR',
    '>RHOXY.FIT','>PHSXY.FIT', 
    '>RHOYX','>PHSYX',
    '>RHOYX.VAR','>PHSYX.VAR',
    '>RHOYX.ERR','>PHSYX.ERR',
    '>RHOYX.FIT','>PHSYX.FIT',
    '>RHOXX','>PHSXX',
    '>RHOXX.VAR','>PHSXX.VAR',
    '>RHOXX.ERR','>PHSXX.ERR',
    '>RHOXX.FIT','>PHSXX.FIT',
    '>RHOYY','>PHSYY',
    '>RHOYY.VAR','>PHSYY.VAR',
    '>RHOYY.ERR','>PHSYY.ERR',
    '>RHOYY.FIT','>PHSYY.FIT',
    '>FRHOXY','>FPHSXY',
    '>FRHOXY.VAR','>FPHSXY.VAR',
    '>FRHOXY.ERR','>FPHSXY.ERR',
    '>FRHOXY.FIT','>FPHSXY.FIT', 
    '>FRHOXX','>FPHSXX',
    '>FRHOXX.VAR','>FPHSXX.VAR',
    '>FRHOXX.ERR','>FPHSXX.ERR',
    '>FRHOXX.FIT','>FPHSXX.FIT',
    
     #Time series-Sepctra 
     #and EM/OTHERSECT
    '>TSERIESSECT', '>TSERIES', 
    '>=SPECTRASECT', '>=EMAPSECT',
    '>=OTHERSECT',
    
    #Impedance Data Blocks

    '>ZXYR','>ZXYI',
    '>ZXY.VAR', '>ZXYR.VAR',
    '>ZXYI.VAR', '>ZXY.COV', 
    '>ZYXR','>ZYXI',
    '>ZYX.VAR', '>ZYXR.VAR',
    '>ZYXI.VAR', '>ZYX.COV',
    '>ZYYR','>ZYYI',
    '>ZYY.VAR', '>ZYYR.VAR',
    '>ZYYI.VAR', '>ZYY.COV',
    '>ZXXR', '>ZXXI',
    '>ZXXR.VAR','>ZXXI.VAR',
    '>ZXX.VAR','>ZXX.COV',
    '>FZXXR','>FZXXI',
    '>FZXYR','>FZXYI', 
    
    #Continuous 1D inversion 
    '>RES1DXX', '>DEP1DXX', 
    '>RES1DXY', '>DEP1DXY',
    '>RES1DYX', '>DEP1DYX', 
    '>RES1DYY', '>DEP1DYY',
    '>FRES1DXX', '>FDEP1DXX',
    '>FRES1DXY', '>FDEP1DXY',
    
    # Coherency and 
    #Signal Data Blocks
    '>COH','>EPREDCOH',
    '>HPREDCOH','>SIGAMP',
    '>SIGNOISE', 
    
    #Tipper Data blocks 
    '>TIPMAG','>TIPPHS',
    'TIPMAG.ERR', '>TIPMAG.FIT',
    '>TIPPHS.FIT', '>TXR.EXP',
    '>TXI.EXP', '>TXVAR.EXP',
    '>TYR.EXP','>TYI.EXP',
    '>TYVAR.EXP',
    
     # Strike, Skew, and
     # Ellipticity Data Blocks 
    '>ZSTRIKE','>ZSKEW',
    '>ZELLIP', '>TSTRIKE',
    '>TSKEW','>TELLIP',
    
    #Spatial filter blocks
    '>FILWIDTH','>FILANGLE',
    '>EQUIVLEN' , '>END'
] 
#
_ZRP_COMPS =[
    #z
    [
         ['zxxr', 'zxxi', 'zxx.var'],
         ['zxyr', 'zxyi', 'zxy.var'],
         ['zyxr', 'zyxi', 'zyx.var'],
         ['zyyr', 'zyyi', 'zyy.var']
    ],
    #Rho
    [
         ['rhoxx', 'rhoxx.var','rhoxx.err', 'rhoxx.fit'],
         ['rhoxy','rhoxy.var','rhoxy.err', 'rhoxy.fit']
    ],
                                    
    [
         ['phxx','phsxx.var', 'phsxx.err', 'phsxx.fit'],
         ['phxy','phsxy.var', 'phsxy.err', 'phsxy.fit']
    ], 
    # filtered 
    [
         ['frhoxx','frhoxx.var','frhoxx.err', 'frhoxx.fit'],
         ['frhoxy','frhoxy.var', 'frhoxy.err', 'frhoxy.fit']
    ],
                                         
    [
         ['fphsxx','fphsxx.var', 'fphsxx.err', 'fphsxx.fit'],
         ['fphsxy','fphsxy.var', 'fphsxy.err', 'fphsxy.fit']
     ],
 ]
                                                    
_TIP_COMPS =[
    ['txr.exp', 'txi.exp', 'txvar.exp'],
    ['tyr.exp', 'tyi.exp', 'tyvar.exp']
    ]

@refAppender(refglossary.__doc__) 
class Water (ABC): 
    r""" Should be a SuperClass for methods classes which deals with water 
    properties and components. 
    
    Instanciate the class shoud raise an error.  
    
    Water (H2O) is a polar inorganic compound that is at room temperature a 
    tasteless and odorless liquid, which is nearly colorless apart from an 
    inherent hint of blue. It is by far the most studied chemical compound 
    and is described as the "universal solvent"and the "solvent of life".
    It is the most abundant substance on the surface of Earth and the only 
    common substance to exist as a solid, liquid, and gas on Earth's surface.
    It is also the third most abundant molecule in the universe 
    (behind molecular hydrogen and carbon monoxide).
    
    The Base class initialize arguments for different methods such as the 
    |ERP| and for |VES|. The `Water` should set the attributes and check 
    whether attributes are suitable for what the specific class expects to. 
    
    Hold some properties informations: 
        
    =================   =======================================================
    Property            Description        
    =================   =======================================================
    state               official names for the chemical compound r"$H_2O$". It 
                        can be a matter of ``solid``, ``ice``, ``gaseous``, 
                        ``water vapor`` or ``steam``. The *default* is ``None``.
    taste               water from ordinary sources, including bottled mineral 
                        water, usually has many dissolved substances, that may
                        give it varying tastes and odors. Humans and other 
                        animals have developed senses that enable them to
                        evaluate the potability of water in order to avoid 
                        water that is too ``salty`` or ``putrid``. 
                        The *default* is ``potable``.    
    odor                Pure water is usually described as tasteless and odorless, 
                        although humans have specific sensors that can feel 
                        the presence of water in their mouths,and frogs are known
                        to be able to smell it. The *default* is ``pure``.
    color               The color can be easily observed in a glass of tap-water
                        placed against a pure white background, in daylight.
                        The **default** is ``pure white background``. 
    appearance          Pure water is visibly blue due to absorption of light 
                        in the region ca. 600 nm – 800 nm. The *default* is 
                        ``visible``. 
    density             Water differs from most liquids in that it becomes
                        less dense as it freezes. In 1 atm pressure, it reaches 
                        its maximum density of ``1.000 kg/m3`` (62.43 lb/cu ft)
                        at 3.98 °C (39.16 °F). The *default* units and values
                        are ``kg/m3``and ``1.`` 
    magnetism           Water is a diamagnetic material. Though interaction
                        is weak, with superconducting magnets it can attain a 
                        notable interaction. the *default* value is 
                        :math:`-0.91 \chi m`". Note that the  magnetism  
                        succeptibily has no unit. 
    capacity            stands for `heat capacity`. In thermodynamics, the 
                        specific heat capacity (symbol cp) of a substance is the
                        heat capacity of a sample of the substance divided by 
                        the mass of the sample. Water has a very high specific
                        heat capacity of 4184 J/(kg·K) at 20 °C 
                        (4182 J/(kg·K) at 25 °C).The *default* is is ``4182 ``
    vaporization        stands for `heat of vaporization`. Indeed, the enthalpy  
                        of vaporization (symbol :math:`\delta H_{vap}`), also  
                        known as the (latent) heat of vaporization or heat of 
                        evaporation, is the amount of energy (enthalpy) that  
                        must be added to a liquid substance to transform a 
                        quantity of that substance into a gas. Water has a high 
                        heat of vaporization i.e. 40.65 kJ/mol or 2257 kJ/kg 
                        at the normal boiling point), both of which are a  
                        result of the extensive hydrogen bonding between its 
                        molecules. The *default* is ``2257 kJ/kg``. 
    fusion              stands for `enthalpy of fusion` more commonly known as 
                        latent heat of water is 333.55 kJ/kg at 0 °C. The 
                        *default* is ``33.55``.
    miscibility         Water is miscible with many liquids, including ethanol
                        in all proportions. Water and most oils are immiscible 
                        usually forming layers according to increasing density
                        from the top. *default* is ``True``.                    
    condensation        As a gas, water vapor is completely miscible with air so 
                        the vapor's partial pressure is 2% of atmospheric 
                        pressure and the air is cooled from 25 °C, starting at
                        about 22 °C, water will start to condense, defining the
                        dew point, and creating fog or dew. The *default* is the 
                        degree of condensation set to ``22°C``. 
    pressure            stands for `vapour pressure` of water. It is the pressure 
                        exerted by molecules of water vapor in gaseous form 
                        i.e. whether pure or in a mixture with other gases such
                        as air.  The vapor pressure is given as a list from the 
                        temperature T, 0°C (0.6113kPa) to 100°C(101.3200kPa). 
                        *default* is ``(0.611, ..., 101.32)``.
    compressibility     The compressibility of water is a function of pressure 
                        and temperature. At 0 °C, at the limit of zero pressure,
                        the compressibility is ``5.1x10^−10 P^{a^−1}``. 
                        The *default* value is the value at 0 °C.
    triple              stands for `triple point`. The temperature and pressure
                        at which ordinary solid, liquid, and gaseous water 
                        coexist in equilibrium is a triple point of water. The 
                        `triple point` are set to (.001°C,611.657 Pa) and 
                        (100 , 101.325kPa) for feezing (0°C) and boiling point
                        (100°C) points. In addition, the `triple point` can be
                        set as ``(20. , 101.325 kPa)`` for 20°C. By *default*,
                        the `triple point` solid/liquid/vapour is set to 
                        ``(.001, 611.657 Pa )``.
    melting             stands for `melting point`. Water can remain in a fluid
                        state down to its homogeneous nucleation point of about
                        231 K (−42 °C; −44 °F). The melting point of ordinary
                        hexagonal ice falls slightly under moderately high 
                        pressures, by 0.0073 °C (0.0131 °F)/atm[h] or about 
                        ``0.5 °C`` (0.90 °F)/70 atm considered as the 
                        *default* value.                   
    conductivity        In pure water, sensitive equipment can detect a very 
                        slight electrical conductivity of 0.05501 ± 0.0001 
                        μS/cm at 25.00 °C. *default* is  ``.05501``.  
    polarity            An important feature of water is its polar nature. The
                        structure has a bent molecular geometry for the two 
                        hydrogens from the oxygen vertex. The *default* is 
                        ``bent molecular geometry`` or ``angular or V-shaped``. 
                        Other possibility is ``covalent bonds `` 
                        ``VSEPR theory`` for Valence Shell Electron Repulsion.
    cohesion            stands for the collective action of hydrogen bonds 
                        between water molecules. The *default* is ``coherent``
                        for the water molecules staying close to each other. 
                        In addition, the `cohesion` refers to the tendency of
                        similar or identical particles/surfaces to cling to
                        one another.
    adhesion            stands for the tendency of dissimilar particles or 
                        surfaces to cling to one another. It can be 
                        ``chemical adhesion``, ``dispersive adhesion``, 
                        ``diffusive adhesion`` and ``disambiguation``.
                        The *default* is ``disambiguation``.
    tension             stands for the tendency of liquid surfaces at rest to 
                        shrink into the minimum surface area possible. Water 
                        has an unusually high surface tension of 71.99 mN/m 
                        at 25 °C[63] which is caused by the strength of the
                        hydrogen bonding between water molecules. This allows
                        insects to walk on water. The *default*  value is to 
                        ``71.99 mN/m at 25 °C``. 
    action              stands for `Capillary action`. Water has strong cohesive
                        and adhesive forces, it exhibits capillary action. 
                        Strong cohesion from hydrogen bonding and adhesion 
                        allows trees to transport water more than 100 m upward.
                        So the *default* value is set to ``100.``meters. 
    issolvent           Water is an excellent solvent due to its high dielectric
                        constant. Substances that mix well and dissolve in water
                        are known as hydrophilic ("water-loving") substances,
                        while those that do not mix well with water are known
                        as hydrophobic ("water-fearing") substances.           
    tunnelling          stands for `quantum tunneling`. It is a quantum 
                        mechanical phenomenon whereby a wavefunction can 
                        propagate through a potential barrier. It can be 
                        ``monomers`` for the motions which destroy and 
                        regenerate the weak hydrogen bond by internal rotations, 
                        or ``hexamer`` involving the concerted breaking of two 
                        hydrogen bond. The *default* is ``hexamer`` discovered 
                        on 18 March 2016.
    reaction            stands for `acide-base reactions`. Water is 
                        ``amphoteric`` i.e. it has the ability to act as either
                        an acid or a base in chemical reactions.
    ionization          In liquid water there is some self-ionization giving 
                        ``hydronium`` ions and ``hydroxide`` ions. *default* is 
                        ``hydroxide``. 
    earthmass           stands for the earth mass ration in "ppm" unit. Water 
                        is the most abundant substance on Earth and also the 
                        third most abundant molecule in the universe after the 
                        :math:`H_2 \quad \text{and} \quad CO` . The *default* 
                        value is ``0.23``ppm of the earth's mass. 
    occurence           stands for the abundant molecule in the Earth. Water 
                        represents ``97.39%`` of the global water volume of
                        1.38×109 km3 is found in the oceans considered as the 
                        *default* value.
    pH                  stands for `Potential of Hydrogens`. It also shows the 
                        acidity in nature of water. For instance the "rain" is
                        generally mildly acidic, with a pH between 5.2 and 5.8 
                        if not having any acid stronger than carbon dioxide. At
                        neutral pH, the concentration of the hydroxide ion 
                        (:math:`OH^{-}`) equals that of the (solvated) hydrogen 
                        ion(:math:`H^{+}`), with a value close to ``10^−7 mol L^−1`` 
                        at 25 °C. The *default* is ``7.`` or ``neutral`` or the
                        name of any substance `pH` close to.
    nommenclature       The accepted IUPAC name of water is ``oxidane`` or 
                        simply ``water``. ``Oxidane`` is only intended to be 
                        used as the name of the mononuclear parent hydride used
                        for naming derivatives of water by substituent 
                        nomenclature. The *default* name is ``oxidane``.                    
    =================   =======================================================                        
    
    
    Notes  
    ----------
    Water (chemical formula H2O) is an inorganic, transparent, tasteless, 
    odorless, and nearly colorless chemical substance, which is the main 
    constituent of Earth's hydrosphere and the fluids of all known living 
    organisms (in which it acts as a solvent). It is vital for all known 
    forms of life, even though it provides neither food, energy, nor organic 
    micronutrients. Its chemical formula, H2O, indicates that each of its 
    molecules contains one oxygen and two hydrogen atoms, connected by covalent
    bonds. The hydrogen atoms are attached to the oxygen atom at an angle of
    104.45°. "Water" is the name of the liquid state of H2O at standard 
    temperature and pressure.

    """
    
    @abstractmethod 
    def __init__(self, 
                 state: str = None, 
                 taste: str  = 'potable', 
                 odor: int | str = 'pure', 
                 appearance: str = 'visible',
                 color: str = 'pure white background', 
                 capacity: float = 4184. , 
                 vaporization: float  = 2257.,  
                 fusion: float = 33.55, 
                 density: float = 1. ,
                 magnetism: float = -.91, 
                 miscibility: bool  =True , 
                 condensation: float = 22, 
                 pressure: tuple = (.6113, ..., 101.32), 
                 compressibility: float  =5.1e-10, 
                 triple: tuple = (.001, 611.657 ),
                 conductivity: float = .05501,
                 melting: float = .5,       
                 polarity: str  ='bent molecular geometry ', 
                 cohesion: str = 'coherent', 
                 adhesion: str  ='disambiguation', 
                 tension: float  = 71.99, 
                 action: float  = 1.e2 ,
                 issolvent: bool =True, 
                 reaction:str  = 'amphoteric', # 
                 ionisation:str  = "hydroxide", 
                 tunneling: str  = 'hexamer' ,
                 nommenclature: str ='oxidane', 
                 earthmass: float =.23 , 
                 occurence: float = .9739,
                 pH: float| str = 7., 
                 ): 
       
        self.state=state 
        self.taste=taste 
        self.odor=odor
        self.appearance=appearance
        self.color=color
        self.capacity=capacity 
        self.vaporization=vaporization   
        self.fusion=fusion 
        self.density=density  
        self.magnetism=magnetism 
        self.miscibility=miscibility 
        self.condensation=condensation 
        self.pressure=pressure, 
        self.compressibility=compressibility 
        self.triple=triple 
        self.conductivity=conductivity
        self.melting=melting      
        self.polarity=polarity  
        self.cohesion=cohesion 
        self.adhesion=adhesion 
        self.tension=tension 
        self.action=action 
        self.issolvent=issolvent 
        self.reaction=reaction
        self.ionisation=ionisation 
        self.tunneling=tunneling 
        self.nommenclature=nommenclature
        self.earthmass=earthmass 
        self.occurence=occurence 
        self.pH=pH
     

class BasePlot(ABC): 
    r""" Base class  deals with Machine learning and conventional Plots. 
    
    The `BasePlot` can not be instanciated. It is build on the top of other 
    plotting classes  and its attributes are used for external plots.
    
    Hold others optional informations: 
        
    ==================  =======================================================
    Property            Description        
    ==================  =======================================================
    fig_dpi             dots-per-inch resolution of the figure
                        *default* is 300
    fig_num             number of the figure instance. *default* is ``1``
    fig_aspect          ['equal'| 'auto'] or float, figure aspect. Can be 
                        rcParams["image.aspect"]. *default* is ``auto``.
    fig_size            size of figure in inches (width, height)
                        *default* is [5, 5]
    savefig             savefigure's name, *default* is ``None``
    fig_orientation     figure orientation. *default* is ``landscape``
    fig_title           figure title. *default* is ``None``
    fs                  size of font of axis tick labels, axis labels are
                        fs+2. *default* is 6 
    ls                  [ '-' | '.' | ':' ] line style of mesh lines
                        *default* is '-'
    lc                  line color of the plot, *default* is ``k``
    lw                  line weight of the plot, *default* is ``1.5``
    alpha               transparency number, *default* is ``0.5``  
    font_weight         weight of the font , *default* is ``bold``.        
    ms                  size of marker in points. *default* is 5
    marker              style  of marker in points. *default* is ``o``.
    marker_facecolor    facecolor of the marker. *default* is ``yellow``
    marker_edgecolor    edgecolor of the marker. *default* is ``cyan``.
    marker_edgewidth    width of the marker. *default* is ``3``.
    xminorticks         minortick according to x-axis size and *default* is 1.
    yminorticks         minortick according to y-axis size and *default* is 1.
    font_size           size of font in inches (width, height)
                        *default* is 3.
    font_style          style of font. *default* is ``italic``
    bins                histograms element separation between two bar. 
                         *default* is ``10``. 
    xlim                limit of x-axis in plot. *default* is None 
    ylim                limit of y-axis in plot. *default* is None 
    xlabel              label name of x-axis in plot. *default* is None 
    ylabel              label name  of y-axis in plot. *default* is None 
    rotate_xlabel       angle to rotate `xlabel` in plot. *default* is None 
    rotate_ylabel       angle to rotate `ylabel` in plot. *default* is None 
    leg_kws             keyword arguments of legend. *default* is empty dict.
    plt_kws             keyword arguments of plot. *default* is empty dict
    plt_style           keyword argument of 2d style. *default* is ``pcolormesh``
    plt_shading         keyword argument of Axes pycolormesh shading. It can be 
                        ['flat'|'nearest'|'gouraud'|'auto'].*default* is 
                        'auto'
    imshow_interp       ['bicubic'|'nearest'|'bilinear'|'quadractic' ] kind of 
                        interpolation for 'imshow' plot. Click `interpol_imshow`_ 
                        to get furher details about the interpolation method. 
                        *default* is ``None``.
    rs                  [ '-' | '.' | ':' ] line style of `Recall` metric
                        *default* is '--'
    ps                  [ '-' | '.' | ':' ] line style of `Precision `metric
                        *default* is '-'
    rc                  line color of `Recall` metric *default* is ``(.6,.6,.6)``
    pc                  line color of `Precision` metric *default* is ``k``
    s                   size of items in scattering plots. default is ``fs*40.``
    cmap                matplotlib colormap. *default* is `jet_r`
    gls                 [ '-' | '.' | ':' ] line style of grid  
                        *default* is '--'.
    glc                 line color of the grid plot, *default* is ``k``
    glw                 line weight of the grid plot, *default* is ``2``
    galpha              transparency number of grid, *default* is ``0.5``  
    gaxis               axis to plot grid.*default* is ``'both'``
    gwhich              type of grid to plot. *default* is ``major``
    tp_axis             axis  to apply ticks params. default is ``both``
    tp_labelsize        labelsize of ticks params. *default* is ``italic``
    tp_bottom           position at bottom of ticks params. *default*
                        is ``True``.
    tp_top              position at the top  of ticks params. *default*
                        is ``True``.
    tp_labelbottom      see label on the bottom of the ticks. *default* 
                        is ``False``
    tp_labeltop         see the label on the top of ticks. *default* is ``True``
    cb_orientation      orientation of the colorbar. *default* is ``vertical``
    cb_aspect           aspect of the colorbar. *default* is 20.
    cb_shrink           shrink size of the colorbar. *default* is ``1.0``
    cb_pad              pad of the colorbar of plot. *default* is ``.05``
    cb_anchor           anchor of the colorbar. *default* is ``(0.0, 0.5)``
    cb_panchor          proportionality anchor of the colorbar. *default* is 
                        `` (1.0, 0.5)``.
    cb_label            label of the colorbar. *default* is ``None``.      
    cb_spacing          spacing of the colorbar. *default* is ``uniform``
    cb_drawedges        draw edges inside of the colorbar. *default* is ``False``
    cb_format           format of the colorbar values. *default* is ``None``.
    sns_orient          seaborn fig orientation. *default* is ``v`` which refer
                        to vertical 
    sns_style           seaborn style 
    sns_palette         seaborn palette 
    sns_height          seaborn height of figure. *default* is ``4.``. 
    sns_aspect          seaborn aspect of the figure. *default* is ``.7``
    sns_theme_kws       seaborn keywords theme arguments. default is ``{
                        'style':4., 'palette':.7}``
    verbose             control the verbosity. Higher value, more messages.
                        *default* is ``0``.
    ==================  =======================================================
    
    """
    
    @abstractmethod 
    def __init__(self,
                 savefig: str = None,
                 fig_num: int =  1,
                 fig_size: tuple =  (12, 8),
                 fig_dpi:int = 300, 
                 fig_legend: str =  None,
                 fig_orientation: str ='landscape',
                 fig_title:str = None,
                 fig_aspect:str='auto',
                 font_size: float =3.,
                 font_style: str ='italic',
                 font_weight: str = 'bold',
                 fs: float = 5.,
                 ms: float =3.,
                 marker: str = 'o',
                 markerfacecolor: str ='yellow',
                 markeredgecolor: str = 'cyan',
                 markeredgewidth: float =  3.,
                 lc: str =  'k',
                 ls: str = '-',
                 lw: float = 1.,
                 alpha: float =  .5,
                 bins: int =  10,
                 xlim: list = None, 
                 ylim: list= None,
                 xminorticks: int=1, 
                 yminorticks: int =1,
                 xlabel: str  =  None,
                 ylabel: str = None,
                 rotate_xlabel: int = None,
                 rotate_ylabel: int =None ,
                 leg_kws: dict = dict(),
                 plt_kws: dict = dict(), 
                 plt_style:str="pcolormesh",
                 plt_shading: str="auto", 
                 imshow_interp:str =None,
                 s: float=  40.,
                 cmap:str='jet_r',
                 show_grid: bool = False,
                 galpha: float = .5,
                 gaxis: str = 'both',
                 gc: str = 'k',
                 gls: str = '--',
                 glw: float = 2.,
                 gwhich: str = 'major',               
                 tp_axis: str = 'both',
                 tp_labelsize: float = 3.,
                 tp_bottom: bool =True,
                 tp_top: bool = True,
                 tp_labelbottom: bool = False,
                 tp_labeltop: bool = True,               
                 cb_orientation: str = 'vertical',
                 cb_aspect: float = 20.,
                 cb_shrink: float =  1.,
                 cb_pad: float =.05,
                 cb_anchor: tuple = (0., .5),
                 cb_panchor: tuple=  (1., .5),              
                 cb_label: str = None,
                 cb_spacing: str = 'uniform' ,
                 cb_drawedges: bool = False,
                 cb_format: float = None ,   
                 sns_orient: str ='v', 
                 sns_style: str = None, 
                 sns_palette: str= None, 
                 sns_height: float=4. , 
                 sns_aspect:float =.7, 
                 sns_theme_kws: dict = None,
                 verbose: int=0, 
                 ): 
        
        self.savefig=savefig
        self.fig_num=fig_num
        self.fig_size=fig_size
        self.fig_dpi=fig_dpi
        self.fig_legend=fig_legend
        self.fig_orientation=fig_orientation
        self.fig_title=fig_title
        self.fig_aspect=fig_aspect
        self.font_size=font_size
        self.font_style=font_style
        self.font_weight=font_weight
        self.fs=fs
        self.ms=ms
        self.marker=marker
        self.marker_facecolor=markerfacecolor
        self.marker_edgecolor=markeredgecolor
        self.marker_edgewidth=markeredgewidth
        self.lc=lc
        self.ls=ls
        self.lw=lw
        self.alpha=alpha
        self.bins=bins
        self.xlim=xlim
        self.ylim=ylim
        self.x_minorticks=xminorticks
        self.y_minorticks=yminorticks
        self.xlabel=xlabel
        self.ylabel=ylabel
        self.rotate_xlabel=rotate_xlabel
        self.rotate_ylabel=rotate_ylabel
        self.leg_kws=leg_kws
        self.plt_kws=plt_kws
        self.plt_style=plt_style
        self.plt_shading=plt_shading
        self.imshow_interp=imshow_interp
        self.s=s 
        self.cmap=cmap
        self.show_grid=show_grid
        self.galpha=galpha
        self.gaxis=gaxis
        self.gc=gc
        self.gls=gls
        self.glw=glw
        self.gwhich=gwhich
        self.tp_axis=tp_axis
        self.tp_labelsize=tp_labelsize  
        self.tp_bottom=tp_bottom
        self.tp_top=tp_top
        self.tp_labelbottom=tp_labelbottom
        self.tp_labeltop=tp_labeltop
        self.cb_orientation=cb_orientation
        self.cb_aspect=cb_aspect
        self.cb_shrink=cb_shrink
        self.cb_pad=cb_pad
        self.cb_anchor=cb_anchor
        self.cb_panchor=cb_panchor
        self.cb_label=cb_label
        self.cb_spacing=cb_spacing
        self.cb_drawedges=cb_drawedges
        self.cb_format=cb_format  
        self.sns_orient=sns_orient
        self.sns_style=sns_style
        self.sns_palette=sns_palette
        self.sns_height=sns_height
        self.sns_aspect=sns_aspect
        self.verbose=verbose
        self.sns_theme_kws=sns_theme_kws or {'style':self.sns_style, 
                                         'palette':self.sns_palette, 
                                                      }
        self.cb_props = {
            pname.replace('cb_', '') : pvalues
                         for pname, pvalues in self.__dict__.items() 
                         if pname.startswith('cb_')
                         }
       
         
        
class ElectricalMethods (ABC) : 
    """ Base class of geophysical electrical methods 

    The electrical geophysical methods are used to determine the electrical
    resistivity of the earth's subsurface. Thus, electrical methods are 
    employed for those applications in which a knowledge of resistivity 
    or the resistivity distribution will solve or shed light on the problem 
    at hand. The resolution, depth, and areal extent of investigation are 
    functions of the particular electrical method employed. Once resistivity 
    data have been acquired, the resistivity distribution of the subsurface 
    can be interpreted in terms of soil characteristics and/or rock type and 
    geological structure. Resistivity data are usually integrated with other 
    geophysical results and with surface and subsurface geological data to 
    arrive at an interpretation. Get more infos by consulting this
    `link <https://wiki.aapg.org/Electrical_methods>`_ . 
    
    
    The :class:`watex.methods.electrical.ElectricalMethods` compose the base 
    class of all the geophysical methods that images the underground using 
    the resistivity values. Is another Base class of :mod:`~.methods.electrical` 
    especially the :class:`~.methods.electrical.ResistivityProfiling` and 
    :class:`~.methods.electrical.VerticalSounding`. It is composed of the 
    details of geolocalisation of the survey area and the array configuration. 
    
    Holds on others optionals infos in ``kws`` arguments: 
       
    ======================  ==============  ===================================
    Attributes              Type                Description  
    ======================  ==============  ===================================
    AB                      float, array    Distance of the current electrodes
                                            in meters. `A` and `B` are used 
                                            as the first and second current 
                                            electrodes by convention. Note that
                                            `AB` is considered as an array of
                                            depth measurement when using the
                                            vertical electrical sounding |VES|
                                            method i.e. AB/2 half-space. Default
                                            is ``200``meters. 
    MN                      float, array    Distance of the current electrodes 
                                            in meters. `M` and `N` are used as
                                            the first and second potential 
                                            electrodes by convention. Note that
                                            `MN` is considered as an array of
                                            potential electrode spacing when 
                                            using the collecting data using the 
                                            vertical electrical sounding |VES|
                                            method i.e MN/2 halfspace. Default 
                                            is ``20.``meters. 
    arrangement             str             Type of dipoles `AB` and `MN`
                                            arrangememts. Can be *schlumberger*
                                            *Wenner- alpha / wenner beta*,
                                            *Gradient rectangular* or *dipole-
                                            dipole*. Default is *schlumberger*.
    area                    str             The name of the survey location or
                                            the exploration area. 
    fromlog10               bool            Set to ``True`` if the given 
                                            resistivities values are collected 
                                            on base 10 logarithm.
    utm_zone                str             string (##N or ##S). utm zone in 
                                            the form of number and North or South
                                            hemisphere, 10S or 03N. 
    datum                   str             well known datum ex. WGS84, NAD27,
                                            etc.         
    projection              str             projected point in lat and lon in 
                                            Datum `latlon`, as decimal degrees 
                                            or 'UTM'. 
    epsg                    int             epsg number defining projection (see 
                                            http://spatialreference.org/ref/ 
                                            for moreinfo). Overrides utm_zone
                                            if both are provided.                           
    ======================  ==============  ===================================
               
    
    Notes
    -------
    The  `ElectricalMethods` consider the given resistivity values as 
    a normal values and not on base 10 logarithm. So if log10 values 
    are given, set the argument `fromlog10` value to ``True``.
    
    .. |VES| replace:: Vertical Electrical Sounding 
    
    """
    
    @abstractmethod 
    def __init__(self, 
                AB: float = 200. , 
                MN: float = 20.,
                arrangement: str  = 'schlumberger', 
                area : str = None, 
                projection: str ='UTM', 
                datum: str ='WGS84', 
                epsg: int =None, 
                utm_zone: str = None,  
                fromlog10:bool =False, 
                verbose: int = 0, 
                ) -> None:
        
        self.AB=AB 
        self.MN=MN 
        self.arrangememt=Config.arrangement(arrangement) 
        self.utm_zone=utm_zone 
        self.projection=projection 
        self.datum=datum
        self.epsg=epsg 
        self.area=area 
        self.fromlog10=fromlog10 
        self.verbose=verbose 
        


class IsEdi(ABC): 
    """ Assert SEG MT/EMAP Data Interchange Standard EDI-file .
    
    Is an abstract Base class for control the valid EDI [1]_. It is also 
    used to ckeck whether object is an instance of EDI object.
    
    EDI stands for Electrical Data Interchange module can read and write an *.edi 
    file as the 'standard ' format of magnetotellurics. Each section of the .edi 
    file belongs to a class object, thus the elements of each section are attributes 
    for easy access. Edi is outputted  following the SEG documentation and rules  
    of EMAP (Electromagnetic  Array Profiling) and MT sections. 
    
    Examples 
    --------
    >>> import pycsamt
    >>> from watex.property import IsEdi 
    >>> from watex.methods.em import EM
    >>> IsEdi.register (pycsamt.core.edi.Edi )
    >>> ediObj= EM().fit(r'data/edis').ediObjs_ [0] # one edi-file for assertion 
    >>> isinstance (ediObj, IsEdi)
    ... True 
    
    References 
    ------------
    .. [1]  Wight, D.E., Drive, B., 1988. MT/EMAP Data Interchange Standard, 
        1rt ed. Society of Exploration Geophysicists, Texas 7831, USA.
    
    """
    @property
    @abstractmethod 
    def is_valid (self): 
        """ Assert whether EDI is valid."""
        pass 
        
    @staticmethod 
    def _assert_edi (file: str ,
                     deep: bool  =True
                     )-> bool : 
        """ Assert EDI- file .
        
        :param file: str - path-like object 
        :param deep: bool - Open the file and assert whether it is a valid EDI
            if ``False``, just control the EDI extension.
            
        :return: bool- ``True`` if EDI is valid and ``False`` otherwise. 
        """
        msg = (" Unrecognized SEG EDI-file. Follow the paper of"
               " [Wight, D.E., Drive, B., 1988.]"
               " <https://www.mtnet.info/docs/seg_mt_emap_1987.pdf>"
               " to build a correct EDI- file."
                )
        flag = False 
        if file  is None : 
            raise  FileHandlingError("NoneType can not be checked. Please"
                                     " provide the right file.") 
        if not os.path.isfile(file): 
            raise FileNotFoundError(f"{file!r} is not file. Expect a Path-like"
                                    " object to EDI-file.")
        if isinstance(file  , str): 
            flag = os.path.splitext(file )[-1].replace('.', '')
            
        if not deep: 
            if flag =='edi': 
                return True 
            # Open the file now 
            if flag !='edi': 
                raise EDIError("Commonly SEG-EDI file must have extension *.edi."
                               f" Got {flag!r}. Set 'deep=True' to check whether"
                               " the file contents match an expected EDI contents."
                               )
        try :
            with open (file, 'r', encoding ='utf8') as f : 
                    edi_data =f.readlines()
        except PermissionError:
            msg =''.join(['https://stackoverflow.com/questions/36434764/',
                          'permissionerror-errno-13-permission-denied'])
            raise PermissionError("This Error occurs because you try to access"
                                  " a file from Python without having the necessary"
                                  " permissions. Please read the following guide"
                                  f" {msg!r} to fix that issue")
        if flag =='edi': 
            if (_EDI[0] not in  edi_data[0]) or  (
                    _EDI[-1] not in  edi_data[-1]): 
                
                raise EDIError(msg)
            flag =True 
        else : raise EDIError(msg)
        
        return flag 
     
class P:
    """
    Data properties are values that are hidden to avoid modifications alongside 
    the packages. Its was used for assertion, comparison etceteara. These are 
    enumerated below into a property objects.
    
    Is a property class that handles the |ERP| and |VES| attributes. Along 
    the :mod:`~.methods.electrical`, it deals with the electrical dipole 
    arrangements, the data classsification and assert whether it is able to 
    be read by the scripts. It is a lind of "asserter". Accept data or reject 
    data provided by the used indicated the way to sanitize it before feeding 
    to the algorithm.

    .. |ERP| replace:: Electrical resistivity profiling 
    
    Parameters  
    -----------
    
    **frcolortags**: Stands for flow rate colors tags. Values are  
        '#CED9EF','#9EB3DD', '#3B70F2', '#0A4CEF'.
                    
    **ididctags**: Stands for the list of index set in dictionary which encompasses 
        key and values of all different prefixes.
                
    **isation**: List of prefixes used for indexing the stations in the |ERP|. 

    **ieasting**: List of prefixes used for indexing the easting coordinates array. 

    **inorthing**: List of prefixes used to index the northing coordinates. 
     
    **iresistivity** List of prefix used for indexing the apparent resistivity 
        values in the |ERP| data collected during the survey. 

    **isren**: Is the list of heads columns during the data collections. Any data 
        head |ERP| data provided should be converted into 
        the following arangement:
                    
        +----------+-------------+-----------+-----------+
        |station   | resistivity | easting   | northing  | 
        +----------+-------------+-----------+-----------+
            
   **isrll**: Is the list of heads columns during the data collections. Any data 
        head |ERP| data provided should be converted into 
        the following arangement:
                   
        +----------+-------------+-------------+----------+
        |station   | resistivity | longitude   | latitude | 
        +----------+-------------+-------------+----------+
            
    **P**: Typing class for fectching the properties. 
        
    Examples 
    ---------
    >>> from watex.property import P 
    >>> P.idicttags 
    ... <property at 0x1ec1f2c3ae0>
    >>> P().idictags 
    ... 
    {'station': ['pk', 'sta', 'pos'], 'longitude': ['east', 'x', 'long', 'lon'],
     'latitude': ['north', 'y', 'lat'], 'resistivity': ['rho', 'app', 'res']}
    >>> {k:v for k, v in  P.__dict__.items() if '__' not in k}
    ... {'_station': ['pk', 'sta', 'pos'],
         '_easting': ['east', 'x', 'long'],
         '_northing': ['north', 'y', 'lat'],
         '_resistivity': ['rho', 'app', 'res'],
         'frcolortags': <property at 0x1ec1f2fee00>,
         'idicttags': <property at 0x1ec1f2c3ae0>,
         'istation': <property at 0x1ec1f2c3ea0>,
         'ieasting': <property at 0x1ec1f2c39f0>,
         'inorthing': <property at 0x1ec1f2c3c70>,
         'iresistivity': <property at 0x1ec1f2c3e00>,
         'isenr': <property at 0x1ec1f2c3db0>}
    >>> P().isrll 
    ... ['station','resistivity','longitude','latitude']
    >>> from watex.property import P 
    >>> pObj = P() 
    >>> P.idictags 
    ... <property at 0x15b2248a450>
    >>> pObj.idicttags 
    ... {'station': ['pk', 'sta', 'pos'],
    ...     'resistivity': ['rho', 'app', 'res', 'se', 'sounding.values'],
    ...     'longitude': ['long', 'lon'],
    ...     'latitude': ['lat'],
    ...     'easting': ['east', 'x'],
    ...     'northing': ['north', 'y']}
    >>> rphead = ['res', 'x', 'y', '']
    >>> pObj (rphead) # sanitize the given resistivity profiling head data.
    ... ['resistivity', 'easting', 'northing']
    >>> rphead = ['lat', 'x', 'rho', '']
    ... ['latitude', 'easting', 'resistivity']
    >>> rphead= ['pos', 'x', 'lon', 'north', 'latitud', 'app.res' ]
    >>> pObj (rphead)
    ... ['station', 'easting', 'longitude', 'northing', 'latitude', 'resistivity'] 
    >>> # --> for sounding head assertion 
    >>> vshead=['ab', 's', 'rho', 'potential']
    >>> pObj (vshead, kind ='ves')
    ... ['AB', 'resistivity'] # in the list of vshead, 
    ... # only 'AB' and 'resistivity' columns are recognized. 
    """
    
    station_prefix   = [
        'pk','sta','pos'
    ]
    easting_prefix   =[
        'east','x',
                ]
    northing_prefix = [
        'north','y',
    ]
    lon_prefix   =[
        'long', 'lon'
                ]
    
    lat_prefix = [
        'lat'
    ]
    
    resistivity_prefix = [
        'rho','app','res', 'se', 'sounding.values'
    ]
    erp_headll= [
        'station', 'resistivity',  'longitude','latitude',
    ]
    erp_headen= [
        'station', 'resistivity',  'easting','northing',
    ]
    ves_head =['AB', 'MN', 'rhoa']
    
    param_options = [
        ['bore', 'for'], 
        ['x','east'], 
        ['y', 'north'], 
        ['pow', 'puiss', 'pa'], 
        ['magn', 'amp', 'ma'], 
        ['shape', 'form'], 
        ['type'], 
        ['sfi', 'if'], 
        ['lat'], 
        ['lon'], 
        ['lwi', 'wi'], 
        ['ohms', 'surf'], 
        ['geol'], 
        ['flow', 'deb']
    ]
    param_ids =[
        'id', 
        'east', 
        'north', 
        'power', 
        'magnitude', 
        'shape', 
        'type', 
        'sfi', 
        'lat', 
        'lon', 
        'lwi', 
        'ohmS', 
        'geol', 
        'flow'
    ]
    
    ves_props = dict (_AB= ['ab', 'ab/2', 'current.electrodes',
                            'depth', 'thickness'],
                      _MN=['mn', 'mn/2', 'potential.electrodes', 'mnspacing'],
                      )
    
    all_prefixes = { f'_{k}':v for k, v in zip (
        erp_headll + erp_headen[2:] , [
            station_prefix,
            resistivity_prefix,
            lon_prefix,
            lat_prefix, 
            easting_prefix, 
            northing_prefix,
            northing_prefix, 
        ]
        )}
    all_prefixes = {**all_prefixes , **ves_props} 
    
    def __init__( self, hl =None ) :
        self.hl = hl
        for key , value in self.all_prefixes.items() : 
            self.__setattr__( key , value)
            
    
    def _check_header_item (self, it , kind ='erp'): 
        """ Check whether the item exists in the property dictionnary.
        Use param `kind` to select the type of header that the data must 
        collected: 
            `kind` = ``erp`` -> for Electrical Resistivity Profiling  
            `kind` = ``ves`` - > for Vertical Electrical Sounding 
        """
            
        dict_ = self.idictcpr if kind =='ves' else self.idicttags
        for k, val in dict_.items(): 
            for s in val : 
                if str(it).lower().find(s)>=0: 
                    return k 
        return  
                
    def __call__(self, hl: list = None , kind :str  ='erp'):
        """ Rename the given header to hold the  properties 
        header values. 
        
        Call function could  return ``None`` whether the 
        given header item in `hl` does not match any item in property 
        headers. 
        
        :param hl: list or array, 
            list of the given headers. 
        :param kind: str 
            Type of data fed into the algorithm. Can be ``ves`` for Vertical 
            Electrical Sounding  and  ``erp`` for Electrical Resistivity Profiling . 
            
        :Example: 
            >>> from watex.property import P 
            >>> test_v= ['pos', 'easting', 'north', 'rhoa', 'lat', 'longitud']
            >>> pobj = P(test_v)
            >>> pobj ()
            ... ['station', 'easting', 'northing', 'resistivity',
                 'latitude', 'longitude']
            >>> test_v2 = test_v + ['straa', 'nourmai', 'opirn'] 
            >>> pobj (test_v2)
            ... ['station', 'easting', 'northing', 'resistivity', 
                 'latitude', 'longitude']
        """
        
        v_ =list()
        
        self.hl = hl or self.hl 
        if self.hl is not None: 
            self.hl = [self.hl] if isinstance(self.hl, str ) else self.hl
            if hasattr(self.hl, '__iter__'):
                for item in self.hl : 
                    v_.append( self._check_header_item(item, kind)) 
                v_=list(filter((None).__ne__, v_))
                return None if len (v_) ==0 else v_
            
    @property 
    def frcolortags (self): 
        """ set the dictionnary"""
        return  dict ((f'fr{k}', f'#{v}') for k, v in zip(
                        range(4), ('CED9EF','9EB3DD', '3B70F2', '0A4CEF' )
                        )
        )
    @property 
    def idicttags (self): 
        """ Is the collection of data properties """ 
        return  dict ( (k, v) for k, v in zip(
            self.isrll + self.isren[2:],
              [self.istation, self.iresistivity, self.ilon, 
                self.ilat, self.ieasting, self.inorthing ])
                      )
    @property 
    def istation(self) : 
        """ Use prefix to identify station location positions """
        return self._station 
    
    @property 
    def ilon (self): 
        """ Use prefix to identify longitude coordinates if given in the
        dataset. """
        return self._longitude 
    
    @property 
    def ilat(self): 
        """ Use prefix to identify latitude coordinates if given in the
        dataset. """
        return self._latitude
    @property 
    def ieasting  (self): 
        """ Use prefix to identify easting coordinates if given in the
        dataset. """
        return self._easting 
    
    @property 
    def inorthing(self): 
        """ Use prefix to identify northing coordinates if given in the
        dataset. """
        return self._northing
    
    @property 
    def iresistivity(self): 
        """ Use prefix to identify the resistivity values in the dataset"""
        return self._resistivity 
    
    @property 
    def isrll(self): 
        """ `SRLL` is the abbreviation of `S`for ``Stations``,`R`` for 
        resistivity, `L` for ``Longitude`` and `L` for ``Latitude``. 
        `SRLL` is the expected columns in Electrical resistivity profiling.
        Indeed, it keeps the traditional collections sheets
        during the survey. """
        return self.erp_headll
    
    @property 
    def isren(self): 
        """ `SREN` is the abbreviation of `S`for ``Stations``,`R``for 
        resistivity, `E` for ``easting`` and `N` for ``northing``. 
        `SREN` is the expected columns in Electrical resistivity profiling.
        Indeed, it keeps the traditional collections sheets
        during the survey. """
        return self.erp_headen
    @property 
    def icpr (self): 
        """ Keep only the Vertical Electrical Sounding header data ..."""
        return [k.replace('_', '') 
                for k in self.ves_props.keys() ] +['resistivity']
    
    @property 
    def idictcpr (self): 
        """ cpr stands for current-potentials and resistivity. They compose the
        main property values when collected the vertical electrical sounding 
        data."""
        return {f'{k.replace("_", "")}': v  for k , v in {
            **self.ves_props, **{'resistivity': self.iresistivity}}.items()}
                

class BagoueNotes: 
    r"""
    A contest class about the `Bagoue dataset`. 
    
    The dataset comes from Bagoue region, located in WestAfrica and lies
    between longitudes 6° and 7° W and latitudes 9° and 11° N in the north of 
    Cote d’Ivoire.
    
    The average FR observed in this area fluctuates between 1
    and 3 m3/h. Refer to the link of case story paper in the `repository docs`_
    to visualize the location map of the study area with the geographical 
    distribution of the various boreholes in the region. The geophysical data 
    and boreholesdata were collected from National  Office of Drinking Water(ONEP) 
    and West-Africa International Drilling  Company (FORACO-CI) during  the 
    Presidential Emergency Program (PPU) in 2012-2013 and the National Drinking 
    Water Supply Program (PNAEP) in 2014.
    The data are firstly composed of Electrical resistivity profile (ERP) data
    collected from geophysical survey lines with various arrays such as
    Schlumberger, gradient rectangle and Wenner :math:`\alpha` or :math:`\beta` 
    and the Vertical electricalsounding (VES) data carried out on the selected anomalies.
    The configuration used during the ERP is Schlumberger with distance of
    :math:`AB = 200m \quad \text{and} \quad  MN =20m`.
    
    The class gives some details about the test dataset used throughout the 
    `WATex`_ packages. It is a guidance for the user to get any details about
    the data preprocessed in order to wuick implement or testing the method.
    Some examples to fetching infos and data are illustrated below: 
        
    Examples 
    -----------

    >>> from watex.datasets import fetch_data
    >>> bag_records = fetch_data('original').get('DESCR')
    ... 'https://doi.org/10.5281/zenodo.5571534: bagoue-original'
    >>> data_contests =fetch_data('original').get('dataset-contest') 
    ... {'__documentation:': '`watex.property.BagoueNotes.__doc__`',
    ...     '__area': 'https://en.wikipedia.org/wiki/Ivory_Coast',
    ...     '__casehistory': 'https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2021WR031623',
    ...     '__wikipages': 'https://github.com/WEgeophysics/watex/wiki',
    ...     '__citations': ('https://doi.org/10.1029/2021wr031623',
    ...      ' https://doi.org/10.5281/zenodo.5529368')}
    >>> #-->  fetching X, y dat
    >>> # get the list of tags before 
    >>> tags=fetch_data('original').get('tags')
    ... ('Bagoue original', ...,'Bagoue prepared sets', 'Bagoue untouched test sets')
    >>> len(tags)
    ... 11 
    >>> --> fetch the preprocessing sets of data 
    >>> X, y = fetch_data('preprocessing')
    >>> X.shape , y.shape 
    ... ((344, 8), (344,)) 
    >>> list(X.columns) 
    ... ['power', 'magnitude', 'sfi', 'ohmS', 'lwi', 'shape', 'type', 'geol']
    >>> X, y = fetch_data('prepared') # data are vectorized and onehotencoded 
    ... ((344, 18), (344,))
    >>> X, y = fetch_data('test sets')
    >>> X.shape , y.shape
    ... ((87, 12), (87,))
        
    
    .. _repository docs: https://github.com/WEgeophysics/watex#documentation>
    
    """
    bagkeys = ('num',
                'name',
                'east',
                'north',
                'power', 
                'magnitude',
                'shape', 
                 'type', 
                'geol',
                'ohmS',
                'flow'
                    )
    bagvalues = [
            'Numbering the data-erp-ves-boreholes',
            'Borehole code', 
            'easting :UTM:29P-30N-WGS84', 
            'northing :UTM:29P-30N-WGS84', 
             'anomaly `power` or anomaly width in meter(m).'
             '__ref :doc:`watex.utils.exmath.compute_power`', 
            'anomaly `magnitude` or `height` in Ω.m '
            '__ref :doc:`watex.utils.exmath.compute_magnitude`', 
            'anomaly `standard fracturing index`, no unit<=sqrt(2)'
            '__ref :doc:`watex.utils.exmath.compute_sfi`', 
             'anomaly `shape`. Can be `V`W`M`K`L`U`H`'
             '__ref :doc:`watex.utils.exmath.get_shape`', 
              'anomaly `shape`. Can be `EC`CP`CB2P`NC`__'
                'ref :doc:`watex.utils.exmath.get_type`', 
            'most dominant geology structure of the area where'
                ' the erp or ves (geophysical survey) is done.', 
            'Ohmic surface compute on sounding curve in '
                'relationship with VES1D inversion (Ω.m2)'
                    '__ref :doc:`watex.methods.electrical.VerticalSounding`',
             'flow rate value of drilling in m3/h'
                    ]
    
   
    bagattr_infos ={
        key:val  for key, val in zip(bagkeys, bagvalues)
                    }


class Config: 
    
    """ Container of property elements. 
    
    Out of bag to keep unmodificable elements. Trick to encapsulate all the 
    element that are not be allow to be modified.
    
    """
    
    @property 
    def arraytype (self):
        """ Different array from |ERP| configuration. 
    
         Array-configuration  can be added as the development progresses. 
         
        """
        return {
        1 : (
            ['Schlumberger','AB>> MN','slbg'], 
            'S'
            ), 
        2 : (
            ['Wenner','AB=MN'], 
             'W'
             ), 
        3: (
            ['Dipole-dipole','dd','AB<BM>MN','MN<NA>AB'],
            'DD'
            ), 
        4: (
            ['Gradient-rectangular','[AB]MN', 'MN[AB]','[AB]'],
            'GR'
            )
        }
    @property
    def parsers(self ): 
        """ Readable format that can be read and parse the data  """
        return {
                 ".csv" : pd.read_csv, 
                 ".xlsx": pd.read_excel,
                 ".json": pd.read_json,
                 ".html": pd.read_html,
                 ".sql" : pd.read_sql, 
                 ".xml" : pd.read_xml , 
                 ".fwf" : pd.read_fwf, 
                 ".pkl" : pd.read_pickle, 
                 ".sas" : pd.read_sas, 
                 ".spss": pd.read_spss, 
                 }
        
    @staticmethod 
    def arrangement(a: int | str ): 
        """ Assert whether the given arrangement is correct. 
        
        :param a: int, float, str - Type of given electrical arrangement. 
        
        :returns:
            - The correct arrangement name 
            - ``0`` which means ``False`` or a wrong given arrangements.   
        """
        
        for k, v in Config().arraytype.items(): 
            if a == k  or str(a).lower().strip() in ','.join (
                    v[0]).lower() or a ==v[1]: 
                return  v[0][0].lower()
            
        return 0
    
    @property 
    def geo_rocks_properties(self ):
        """ Get some sample of the geological rocks. """
        return {
             "basement rocks" :            [1e99,1e6 ],
             "igneous rocks":              [1e6, 1e3], 
             "duricrust"   :               [5.1e3 , 5.1e2],
             "gravel/sand" :               [1e4  , 7.943e0],
             "conglomerate"    :           [1e4  , 8.913e1],
             "dolomite/limestone" :        [1e5 ,  1e3],
            "permafrost"  :                [1e5  , 4.169e2],
             "metamorphic rocks" :         [5.1e2 , 1e1],
             "tills"  :                    [8.1e2 , 8.512e1],
             "standstone conglomerate" :   [1e4 , 8.318e1],
             "lignite/coal":               [7.762e2 , 1e1],
             "shale"   :                   [5.012e1 , 3.20e1],
             "clay"   :                    [1e2 ,  5.012e1],
             "saprolite" :                 [6.310e2 , 3.020e1],
             "sedimentary rocks":          [1e4 , 1e0],
             "fresh water"  :              [3.1e2 ,1e0],
             "salt water"   :              [1e0 , 1.41e0],
             "massive sulphide" :          [1e0   ,  1e-2],
             "sea water"     :             [1.231e-1 ,1e-1],
             "ore minerals"  :             [1e0   , 1e-4],
             "graphite"    :               [3.1623e-2, 3.162e-3]
                
                }
    
    @property 
    def rockpatterns(self): 
        """Default geological rocks patterns. 
        
        pattern are not exhaustiv, can be added and changed. This pattern
        randomly choosen its not exatly match the rocks geological patterns 
        as described with the conventional geological swatches relate to 
        the USGS(US Geological Survey ) swatches- references and FGDC 
        (Digital cartographic Standard for Geological  Map Symbolisation 
         -FGDCgeostdTM11A2_A-37-01cs2.eps)
        
        The following symbols can be used to create a matplotlib pattern. 
        
        make _pattern:{'/', '\', '|', '-', '+', 'x', 'o', 'O', '.', '*'}
                       
            /   - diagonal hatching
            \   - back diagonal
            |   - vertical
            -   - horizontal
            +   - crossed
            x   - crossed diagonal
            o   - small circle
            O   - large circle
            .   - dots
            *   - stars
        """
        return  {
             "basement rocks" :      ['.+++++.', (.25, .5, .5)],
             "igneous rocks":        ['.o.o.', (1., 1., 1.)], 
             "duricrust"   :         ['+.+',(1., .2, .36)],
             "gravel" :              ['oO',(.75,.86,.12)],
             "sand":                 ['....',(.23, .36, .45)],
             "conglomerate"    :     ['.O.', (.55, 0., .36)],
             "dolomite" :            ['.-.', (0., .75, .23)],
             "limestone" :           ['//.',(.52, .23, .125)],
            "permafrost"  :          ['o.', (.2, .26, .75)],
             "metamorphic rocks" :   ['*o.', (.2, .2, .3)],
             "tills"  :              ['-.', (.7, .6, .9)],
             "standstone ":          ['..', (.5, .6, .9)],
             "lignite coal":         ['+/.',(.5, .5, .4)],
             "coal":                 ['*.', (.8, .9, 0.)],
             "shale"   :             ['=', (0., 0., 0.7)],
             "clay"   :              ['=.',(.9, .8, 0.8)],
             "saprolite" :           ['*/',(.3, 1.2, .4)],
             "sedimentary rocks":    ['...',(.25, 0., .25)],
             "fresh water"  :        ['.-.',(0., 1.,.2)],
             "salt water"   :        ['o.-',(.2, 1., .2)],
             "massive sulphide" :    ['.+O',(1.,.5, .5 )],
             "sea water"     :       ['.--',(.0, 1., 0.)],
             "ore minerals"  :       ['--|',(.8, .2, .2)],
             "graphite"    :         ['.++.',(.2, .7, .7)],
             
             }
    

class References:
    """
    References information for a citation.

    Holds the following information:
        
    ================  ==========  =============================================
    Attributes         Type        Explanation
    ================  ==========  =============================================
    author            string      Author names
    title             string      Title of article, or publication
    journal           string      Name of journal
    doi               string      DOI number 
    year              int         year published
    ================  ==========  =============================================

    More attributes can be added by inputing a key word dictionary
    
    Examples
    ---------
    >>> from watex.property import References
    >>> refobj = References(
        **{'volume':18, 'pages':'234--214', 
        'title':'watex :A machine learning research for hydrogeophysic' ,
        'journal':'Computers and Geosciences', 
        'year':'2021', 'author':'DMaryE'}
        )
    >>> refobj.journal
    Out[21]: 'Computers and Geosciences'
    """
    def __init__(
        self, 
        author=None, 
        title=None, 
        journal=None, 
        volume=None, 
        doi=None, 
        year=None,  
        **kws
        ):
        self.author=author 
        self.title=title 
        self.journal=journal 
        self.volume=volume 
        self.doi=doi 
        self.year=year 
   
        for key in list(kws.keys()):
            setattr(self, key, kws[key])


class Copyright:
    """
    Information of copyright, mainly about the use of data can use
    the data. Be sure to read over the conditions_of_use.

    Holds the following informations:

    =================  ===========  ===========================================
    Attributes         Type         Explanation
    =================  ===========  ===========================================
    References          References  citation of published work using these data
    conditions_of_use   string      conditions of use of data used for testing 
                                    program
    release_status      string      release status [ open | public |proprietary]
    =================  ===========  ===========================================

    More attributes can be added by inputing a key word dictionary
    
    Examples
    ----------
    >>> from watex.property import Copyright 
    >>> copbj =Copyright(**{'owner':'University of AI applications',
    ...             'contact':'WATER4ALL'})
    >>> copbj.contact 
    Out[20]: 'WATER4ALL
    
    """
    cuse =( 
        "All Data used for software demonstration mostly located in "
        " data directory <data/> cannot be used for commercial and " 
        " distributive purposes. They can not be distributed to a third"
        " party. However, they can be used for understanding the program."
        " Some available ERP and VES raw data can be found on the record"
        " <'10.5281/zenodo.5571534'>. Whereas EDI-data e.g. EMAP/MT data,"
        " can be collected at http://ds.iris.edu/ds/tags/magnetotelluric-data/."
        " The metadata from both sites are available free of charge and may"
        " be copied freely, duplicated and further distributed provided"
        " these data are cited as the reference."
        )
    def __init__(
        self, 
        release_status=None, 
        additional_info=None, 
        conditions_of_use=None, 
        **kws
        ):
        self.release_status=release_status
        self.additional_info=additional_info
        self.conditions_of_use=conditions_of_use or self.cuse 
        self.References=References()
        for key in list(kws.keys()):
            setattr(self, key, kws[key])


class Person:
    """
    Information for a person

    ================  ==========  =============================================
    Attributes         Type        Explanation
    ================  ==========  =============================================
    email             string      email of person
    name              string      name of person
    organization      string      name of person's organization
    organization_url  string      organizations web address
    ================  ==========  =============================================

    More attributes can be added by inputing a key word dictionary
    
    Examples 
    ----------
    >>> from watex.property import Person
    >>> person =Person(**{'name':'ABA', 'email':'aba@water4all.ai.org',
    ...                  'phone':'00225-0769980706', 
    ...          'organization':'WATER4ALL'})
    >>> person.name
    Out[23]: 'ABA
    >>> person.organization
    Out[25]: 'WATER4ALL'
    """

    def __init__(
        self, 
        email=None, 
        name=None, 
        organization=None, 
        organization_url=None, 
        **kws
        ):
        self.email=email 
        self.name=name 
        self.organization=organization
        self.organization_url=organization_url

        for key in list(kws.keys()):
            setattr(self, key, kws[key])


class Software:
    """
    software info 

    ================= =========== =============================================
    Attributes         Type        Explanation
    ================= =========== =============================================
    name                string      name of software 
    version             string      version of sotware 
    Author              string      Author of software
    release             string      latest version release
    ================= =========== =============================================
    
    More attributes can be added by inputing a key word dictionary

    Examples 
    ----------
    >>> from watex.property import Software
    >>> Software(**{'release':'0.11.23'})

    """
    def __init__(
        self,
        name=None, 
        version=None, 
        release=None, 
        **kws
        ):
        self.name=name 
        self.version=version 
        self.release=release 
        self.Author=Person()
        
        for key in kws:
            setattr(self, key, kws[key]) 
            
                

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
   