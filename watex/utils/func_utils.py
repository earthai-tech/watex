# -*- coding: utf-8 -*-
# Copyright © 2021  Kouadio K.Laurent, Wed Jul  7 22:23:02 2021 hz
# This module is modified and is part of the pyCSAMT utils package,
#  which is released under a LGPL- licence.
# originallly created on Sun Sep 13 09:24:00 2020
# @author: ~alias @Daniel03

####################### import modules #######################
import os 
import shutil 
import warnings
import inspect
import numpy as np 
import matplotlib.pyplot as plt
# from copy import deepcopy
import  watex.utils.gis_tools as gis

from watex.utils._watexlog import watexlog
_logger = watexlog.get_watex_logger(__name__)


try:
    import scipy

    scipy_version = [int(ss) for ss in scipy.__version__.split('.')]
    if scipy_version[0] == 0:
        if scipy_version[1] < 14:
            warnings.warn('Note: need scipy version 0.14.0 or higher or interpolation '
                          'might not work.', ImportWarning)
            _logger.warning('Note: need scipy version 0.14.0 or higher or interpolation '
                            'might not work.')
    import scipy.interpolate as spi

    interp_import = True

except ImportError:  # pragma: no cover
    warnings.warn('Could not find scipy.interpolate, cannot use method interpolate'
                  'check installation you can get scipy from scipy.org.')
    _logger.warning('Could not find scipy.interpolate, cannot use method interpolate'
                    'check installation you can get scipy from scipy.org.')
    interp_import = False

###################### end import module ################################### 


def format_notes(text:str , cover_str:str='~', inline=70, **kws): 
    """ Format note 
    :param text: Text to be formated 
    :param cover_str: type of ``str`` to surround the text 
    :param inline: Nomber of character before going in liine 
    :param margin_space: Must be <1 and expressed in %. The empty distance 
                        between the first index to the inline text 
    :Example: 
        
        >>> from watex.utils import func_utils as func 
        >>> text ='Automatic Option is set to ``True``.'\
            ' Composite estimator building is triggered.' 
        >>>  func.format_notes(text= text ,
        ...                       inline = 70, margin_space = 0.05)
    
    """
    
    headnotes =kws.pop('headernotes', 'notes')
    margin_ratio = kws.pop('margin_space', 0.2 )
    margin = int(margin_ratio * inline)
    init_=0 
    new_textList= []
    if len(text) <= (inline - margin): 
        new_textList = text 
    else : 
        for kk, char in enumerate (text): 
            if kk % (inline - margin)==0 and kk !=0: 
                new_textList.append(text[init_:kk])
                init_ =kk 
            if kk ==  len(text)-1: 
                new_textList.append(text[init_:])
  
    print('!', headnotes.upper(), ':')
    print('{}'.format(cover_str * 70)) 
    for k in new_textList:
        fmtin_str ='{'+ '0:>{}'.format(margin) +'}'
        print('{0}{1:>2}{2:<51}'.format(fmtin_str.format(cover_str), '', k))
        
    print('{0}{1:>51}'.format(' '* (margin -1), cover_str * (70 -margin+1 ))) 
    
    
def concat_array_from_list(list_of_array, concat_axis=0):
    """
    Small function to concatenate a list with array contents 
    
    Parameters 
    -----------
        * list_of_array : list 
                contains a list for array data. the concatenation is possible 
                if an index array have the same size 
        
    Returns 
    -------
        array_like 
            numpy concatenated data 
        
    :Example: 
        
        >>> import numpy as np 
        >>>  np.random.seed(0)
        >>> ass=np.random.randn(10)
        >>> ass2=np.linspace(0,15,12)
        >>>  ass=ass.reshape((ass.shape[0],1))
        >>>  ass2=ass2.reshape((ass2.shape[0],1))
        >>> or_list=[ass,ass2]
        >>> ss_check_error=concat_array_from_list(list_of_array=or_list,
        ...                                          concat_axis=0)
        >>>  secont test :
        >>>  ass=np.linspace(0,15,14)
        >>> ass2=np.random.randn(14)
        >>> ass=ass.reshape((ass.shape[0],1))
        >>> ass2=ass2.reshape((ass2.shape[0],1))
        >>> or_list=[ass,ass2]
        >>>  ss=concat_array_from_list(list_of_array=or_list, concat_axis=0)
        >>> ss=concat_array_from_list(list_of_array=or_list, concat_axis=1)
        >>> ss
        >>> ss.shape 
    """
    #first attemp when the len of list is ==1 :
    
    if len(list_of_array)==1:
        if type(list_of_array[0])==np.ndarray:
            output_array=list_of_array[0]
            if output_array.ndim==1:
                if concat_axis==0 :
                    output_array=output_array.reshape((1,output_array.shape[0]))
                else :
                    output_array=output_array.reshape((output_array.shape[0],1))
            return output_array
        
        elif type(list_of_array[0])==list:
            output_array=np.array(list_of_array[0])
            if concat_axis==0 :
                output_array=output_array.reshape((1,output_array.shape[0]))
            else :
                output_array=output_array.reshape((output_array.shape[0],1))
            return output_array
    
    # check the size of array in the liste when the len of list is >=2
    
    for ii,elt in enumerate(list_of_array) :
        if type(elt)==list:
            elt=np.array(elt)
        if elt is None :
            pass
        elif elt.ndim ==1 :
            if concat_axis==0 :
                elt=elt.reshape((1,elt.shape[0]))
            else :
                elt=elt.reshape((elt.shape[0],1))
        list_of_array[ii]=elt
 

    output_array=list_of_array[0]
    for ii in list_of_array[1:]:
        output_array=np.concatenate((output_array,ii), axis=concat_axis)
        
    return output_array


def sort_array_data(data,  sort_order =0,
              concatenate=False, concat_axis_order=0 ):
    """
    Function to sort array data and concatenate 
    numpy.ndarray 
    
    Parameters
    ----------
        * data : numpy.ndarray 
                must be in simple array , list of array and 
                dictionary whom the value is numpy.ndarray 
                
        * sort_order : int, optional 
                index  of colum to sort data. The default is 0.
                
        * concatenate : Boolean , optional
                concatenate all array in the object.
                Must be the same dimentional if concatenate is set to True. 
                The *default* is False.
                
        * concat_axis_order : int, optional
                must the axis of concatenation  . The default is axis=0.

    Returns
    -------
        numpy.ndarray
           data , Either the simple sort data or 
           array sorted and concatenated .
    """
    
    if type(data)==list :
        for ss, val in data :
            val=val[val[:,sort_order].argsort(kind="mergesort")]
            data[ss]=val
            
        if concatenate: 
            data=concat_array_from_list(list_of_array=data,
                        concat_axis=concat_axis_order)
    elif type(data)==dict:
        
        for key, value in data.items(): 
            value=value[value[:,sort_order].argsort(kind="mergesort")]
            data[key]=value
            
        if concatenate==True :
            temp_list=list(data.values())
            data=concat_array_from_list(list_of_array=temp_list,
                                    concat_axis=concat_axis_order)
    else : 
        data=data[data[:,sort_order].argsort(kind="mergesort")]
        
    return data 
                   
def interpol_scipy (x_value, y_value,x_new,
                    kind="linear",plot=False, fill="extrapolate"):
    
    """
    function to interpolate data 
    
    Parameters 
    ------------
        * x_value : np.ndarray 
                    value on array data : original absciss 
                    
        * y_value : np.ndarray 
                    value on array data : original coordinates (slope)
                    
        * x_new  : np.ndarray 
                    new value of absciss you want to interpolate data 
                    
        * kind  : str 
                projection kind : 
                    maybe : "linear", "cubic"
                    
        * fill : str 
            kind of extraolation, if None , *spi will use constraint interpolation 
            can be "extrapolate" to fill_value.
            
        * plot : Boolean 
            Set to True to see a wiewer graph

    Returns 
    --------
        np.ndarray 
            y_new ,new function interplolate values .
            
    :Example: 
        
        >>> import numpy as np 
        >>>  fill="extrapolate"
        >>>  x=np.linspace(0,15,10)
        >>>  y=np.random.randn(10)
        >>>  x_=np.linspace(0,20,15)
        >>>  ss=interpol_Scipy(x_value=x, y_value=y, x_new=x_, kind="linear")
        >>>  ss
    """
    
    func_=spi.interp1d(x_value, y_value, kind=kind,fill_value=fill)
    y_new=func_(x_new)
    if plot :
        plt.plot(x_value, y_value,"o",x_new, y_new,"--")
        plt.legend(["data", "linear","cubic"],loc="best")
        plt.show()
    
    return y_new


def _set_depth_to_coeff(data, depth_column_index,coeff=1, depth_axis=0):
    
    """
    Parameters
    ----------
        * data : np.ndarray
            must be on array channel .
            
        * depth_column_index : int
            index of depth_column.
            
        * depth_axis : int, optional
            Precise kind of orientation of depth data(axis =0 or axis=1) 
            The *default* is 0.
            
        * coeff : float,
            the value you want to multiplie depth. 
            set depth to negative multiply by one. 
            The *default* is -1.

    Returns
    -------
        data : np.ndarray
            new data after set depth according to it value.
            
    :Example: 

        >>>  import numpy as np 
        >>>  np.random.seed(4)
        >>>  data=np.random.rand(4,3)
        >>>  data=data*(-1)
        >>>  print("data\n",data)
        >>>  data[:,1]=data[:,1]*(-1)
        >>>  data[data<0]
        >>>  print("data2\n",data)
    """
    
    if depth_axis==0:
        data[:,depth_column_index]=data[:,depth_column_index]*coeff
    if depth_axis==1:
        data[depth_column_index,:]=data[depth_column_index,:]*coeff  

    return data
            


def broke_array_to_(arrayData, keyIndex=0, broken_type="dict"):
    """
    broke data array into different value with their same key 

    Parameters
    ----------
        * arrayData :np.array
            data array .
            
        * keyIndex : int 
            index of column to create dict key 

    Returns
    -------
        dict 
           dico_brok ,dictionnary of array.
    """
    
    vcounts_temp,counts_temp=np.unique(arrayData[:,keyIndex], return_counts=True)
    vcounts_temp_max=vcounts_temp.max()

    dico_brok={}
    lis_brok=[]
    index=0
    deb=0
    for rowlines in arrayData :
        if rowlines[0] == vcounts_temp_max:
            value=arrayData[index:,::]
            if broken_type=="dict":
                dico_brok["{0}".format(rowlines[0])]=value
                break
            elif broken_type=="list":
                lis_brok.append(value)
                break
        elif rowlines[0] !=arrayData[index+1,keyIndex]:
            value=arrayData[deb:index+1,::]
            if broken_type=="dict":
                dico_brok["{0}".format(rowlines[0])]=value
            elif broken_type=="list":
                lis_brok.append(value)
            deb=index+1
        index=index+1
    if broken_type=="dict":
        return dico_brok
    elif broken_type=="list":
        return lis_brok
    

def take_firstValue_offDepth(data_array,
                             filter_order=1):
    """
    Parameters
    ----------
        * data_array : np.array 
                array of the data .
        * filter_order : int , optional
                the column you want to filter. The default is 1.

    Returns
    -------
        array_like
            return array of the data filtered.
   
    :Example: 
        
        >>>  import numpy as np 
        >>>  list8=[[4,2,0.1],[8,2,0.7],[10,1,0.18],[4,3,0.1],
        ...        [7,2,1.2],[10,3,0.5],[10,1,0.5],[8.2,0,1.9],
        ...        [10,7,0.5],[10,1,0.5],[2,0,1.4],[5,4,0.5],
        ...        [10,2,0.7],[7,2,1.078],[10,2,3.5],[10,8,1.9]]
        >>>  test=np.array(list8)
        >>>   print(np_test)
        >>>  ss=take_firstValue_offDepth(data_array =np_test, filter_order=1)
        >>>   ss=averageData(np_array=np_test,filter_order=1,
        >>>                 axis_average=0, astype="int")
        >>> print(ss)
    """
    
    listofArray=[]#data_array[0,:]]
    data_array=data_array[data_array[:,filter_order].argsort(kind="mergesort")]
    values, counts =np.unique(data_array[:,filter_order], return_counts=True)
    
    for ii, rowline in enumerate(data_array ): 
    
        if rowline[filter_order]==values[-1]:
            listofArray.append(data_array[ii])
            break 
        elif rowline[filter_order] !=data_array[ii-1][filter_order]:
            listofArray.append(data_array[ii])
        
        
    array =concat_array_from_list(list_of_array=listofArray, concat_axis=0)
    array=array[array[:,filter_order].argsort(kind="mergesort")]
    listofArray=[]

    return array 

def dump_comma(input_car, max_value=2, carType='mixed'):
    """
    Parameters
    ----------
        * input_car : str,
            Input character.
        * max_value : int, optional
            The default is 2.
            
        * carType: str 
            Type of character , you want to entry
                 
    Returns
    -------
        Tuple of input character
            must be return tuple of float value, or string value
      
    .. note:: carType  may be as arguments parameters like ['value','val',"numeric",
              "num", "num","float","int"] or  for pure character like 
                ["car","character","ch","char","str", "mix", "mixed","merge","mer",
                "both","num&val","val&num&"]
                if not , can  not possible to convert to float or integer.
                the *defaut* is mixed 
                
    :Example: 
        
        >>> import numpy as np
        >>>  ss=dump_comma(input_car=",car,box", max_value=3, 
        ...      carType="str")
        >>>  print(ss)
        ... ('0', 'car', 'box')
    """
    
    
    
        # dump "," at the end of 
    flag=0
    
    if input_car[-1]==",":
        input_car=input_car[:-1]
    if input_car[0]==",":
        input_car="0,"+ input_car[1:]
        
    if carType.lower() in ['value','val',"numeric",
                           "num", "num","float","int"]:
        input_car=eval(input_car)
        
    elif carType.lower() in ["car","character","ch","char","str",
                             "mix", "mixed","merge","mer",
                             "both","num&val","val&num&"]:
        
        input_car=input_car.strip(",").split(",")
        flag=1
        

    if np.iterable(input_car)==False :
        inputlist=[input_car,0]

    elif np.iterable(input_car) is True :

        inputlist=list(input_car)

    input_car=inputlist[:max_value]

    if flag==1 :
        if len(inputlist)==1 :
            return(inputlist[0])
    
    return tuple(input_car)

                    
def build_wellData (add_azimuth=False, utm_zone="49N",
                    report_path=None,add_geochemistry_sample=False):
    """
    Parameters
    ----------
        * add_azimuth : Bool, optional
                compute azimuth if add_azimut is set to True. 
                The default is False.
             
        *  utm_zone : Str, optional
                 WGS84 utm_projection. set your zone if add_azimuth is 
                 turn to True. 
                 The default is "49N".
             
        * report_path : str, optional
                path to save your _well_report. The default is None.
                its match the current work directory 
            
        * add_geochemistry_sample: bool
                add_sample_data.Set to True if you want to add_mannually 
                Geochimistry data.
                default is False.

    Raises
    ------
        Exception
            manage the dimentionaly of ndarrays .
        OSError
            when report_path is not found in your O.S.

    Returns
    -------
        str
            name of location of well .
        np.ndarray
             WellData , data of build Wells .
        np.ndarray
           GeolData , data of build geology.

    :Example: 
        
        >>>  import numpy as np 
        >>>  import os, shutil
        >>>  import warnings,
        >>>  form _utils.avgpylog import AvgPyLog
        >>>  well=build_wellData (add_azimuth=True, utm_zone="49N")
        >>>  print("nameof locations\n:",well[0])
        >>>  print("CollarData\n:",well[1])
        >>>  print("GeolData\n:", well[2])
        ...  nameof locations
        ...  Shimen
        ...  CollarData
        ...  [['S01' '477205.6935' '2830978.218' '987.25' '-90' '0.0' 'Shi01'
        ...   'Wdanxl0']
        ...   ['S18' '477915.4355' '2830555.927' '974.4' '-90' '2.111' 'Shi18'
        ...   'Wdanxl0']]
        ...  GeolData
        ...  [['S01' '0.0' '240.2' 'granite']
        ...   ['S01' '240.2' '256.4' ' basalte']
        ...   ['S01' '256.4' '580.0' ' granite']
        ...   ['S01' '580.0' '987.25' 'rock']
        ...   ['S18' '0.0' '110.3' 'sand']
        ...   ['S18' '110.3' '520.2' 'agrilite']
        ...    ['S18' '520.2' '631.3' ' granite']
        ...   ['S18' '631.3' '974.4' ' rock']]
        ...   Shimen_wellReports_
    """
    reg_lines=[]
    wellSites,ftgeo,hole_list,Geolist=[],[],[],[]
    
    text=["Enter the name of Location:",
                      "well_name :",
                      "Coordinates (Easting, Northing)_UTM_{0} : ".format(utm_zone),
                      "Hole Buttom  and dip values (Bottom, dip):" ,
                      "Layers-thickness levels in (meters):",
                      "Geology-layers or <stratigraphy> names (Top_To_Buttom):", 
                      "{0:-^70}".format(' Report '),
                      
                      "DH_Hole,DH_Easting, DH_Northing, DH_Buttom,"\
                      " DH_Dip,DH_Azimuth, DH_PlanDepth,DH_Descrip",
                      
                      "GeolData :",
                      "WellData:",
                      "DH_Hole, DH_From, DH_To, Rock",
                      "SampleData",
                      "DH_Hole, DH_From,DH_To, Sample",
                      "{0:-^70}".format(' InputData '),
                      ]
    
    
    name_of_location =input("Enter the name of Location:")
    reg_lines.append(''.join(text[0]+'{0:>18}'.format(name_of_location)+'\n'))
    reg_lines.append('\n')
    reg_lines.append(text[13]+'\n')
    
    comp=-1
    while 1 :
        DH_Hole=input("Enter the well_name :")
        if DH_Hole=="" or DH_Hole=="end":
            Geol=concat_array_from_list(list_of_array=Geolist,
                                            concat_axis=0)
            break
         
        print("Enter the coordinates (Easting, Northing) : ", end="")
        DH_East_North=input()
        DH_East_North=dump_comma(input_car=DH_East_North,
                                 max_value=2, carType='value')
        print("Enter the Hole Bottom value and dip (Bottom, dip):", end='')
        dh_botdip=input()
        dh_botdip=dump_comma(input_car=dh_botdip,
                                 max_value=2, carType='value')
        #check  the dip of the well 
        if float(dh_botdip[1])==0.:
            dh_botdip[1]=(90.)
        elif float(dh_botdip[0])==0.:
            raise Exception(
            "The curent bottom has a value 0.0 . Must put the "
            "bottom of the well as deep as possible !"
                            )
            
        hole_list.append(DH_Hole)

        wellSites.append((DH_Hole, DH_East_North[0],DH_East_North[1],
                          dh_botdip[0],dh_botdip[1] ))

        #DH_Hole (ID)	DH_East	DH_North	DH_RH	DH_Dip	DH_Azimuth	DH_Top	DH_Bottom	DH_PlanDepth	DH_Decr	Mask 
        reg_lines.append("".join(text[1]+'{0:>7}'.format(DH_Hole))+"\n")
        reg_lines.append("".join(text[2])+"\n")
        reg_lines.append(",".join(['{0:>14}'.format(str(ii))
                                   for ii in list(DH_East_North)])+"\n")
        reg_lines.append("".join(text[3])+"\n")
        reg_lines.append(",".join(['{0:>7}'.format(str(ii))
                                   for ii in list(dh_botdip)])+"\n")
        
        comp+=-1
        while DH_Hole :
            # print("Enter the layer thickness (From_, _To, geology):",end='')
            if comp==-1 :
                Geol=concat_array_from_list(list_of_array=ftgeo,
                                            concat_axis=0)
                
                ftgeo=[]        #  initialize temporary list 
                
                break
            # comp=comp+1
            print("Enter the layers-thickness levels in (meters):", end='')
            dh_from_in=input()
            
            if dh_from_in=="" or dh_from_in=="end":
                break

            dh_from=eval(dh_from_in)
            
            dh_from_ar=np.array(dh_from)
            dh_from_ar=dh_from_ar.reshape((dh_from_ar.shape[0],1)) 
            # check the last input bottom : 
            if dh_from_ar[-1] >= float(dh_botdip[0]):
                _logger.info("The input bottom of well {{0}}, is {{1}}."
                             "It's less last layer thickess: {{2}}."
                             "we add maximum bottom at 1.023km depth.".
                             format(DH_Hole,dh_botdip[0],
                                    dh_from_ar[-1]))
                dh_botdip[0]=(1023.)
                wellSites[-1][3]=dh_botdip[0]  # last append of wellSites 
            #find Dh_to through give dh_from
            
            dh_to=dh_from_ar[1:]
            dh_to=np.append(dh_to,dh_botdip[0])
            dh_to=dh_to.reshape((dh_to.shape[0],1))
            
            print("Enter the geology-layers names (From _To):",end="")
            rock=input()
            rock=rock.strip(",").split(",") # strip in the case where ","appear at the end 
            rock_ar=np.array(rock)
            rock_ar=rock_ar.reshape((rock_ar.shape[0],1))

            try :
                if rock_ar.shape[0]==dh_from_ar.shape[0]:
                    drill_names=np.full((rock_ar.shape[0],1),DH_Hole)
                fromtogeo=np.concatenate((drill_names,dh_from_ar,
                                              dh_to, rock_ar),axis=1)
            except IndexError:
                _logger.warn("np.ndarry sizeError:Check 'geologie', 'Dh_From', and "\
                             "'Dh_To' arrays size properly . It seems one size is "\
                                 " too longeR than another. ")
                warnings.warn(" All the arrays size must match propertly!")
                
            ftgeo.append(fromtogeo)
            comp=-1
            
            reg_lines.append("".join(text[4])+"\n")
            reg_lines.append(",".join(['{0:>7}'.format(str(jj))
                                       for jj in list(dh_from)])+"\n")
            reg_lines.append("".join(text[5])+"\n")
            reg_lines.append(",".join(['{0:>12}'.format(jj) 
                                       for jj in rock])+"\n") # rock already in list 
            # reg_lines.append("".join(rock+"\n"))
            
        reg_lines.append("\n")
        
        Geolist.append(Geol)    


    name_of_location=name_of_location.capitalize()
    #set on numpy array
    
    for ii , value in enumerate(wellSites):
        value=np.array(list(value))
        wellSites[ii]=value
    #create a wellsites array 
    
    wellSites=concat_array_from_list(wellSites,concat_axis=0)
    DH_Hole=wellSites[:,0]
    DH_PlanDepth=np.zeros((wellSites.shape[0],),dtype="<U8")
    DH_Decr=np.full((wellSites.shape[0],),"Wdanxl0",dtype="<U9")
    
    
    lenloc=len(name_of_location)
    for ii , row in enumerate(DH_Hole):
         # in order to keep all the well name location
        DH_PlanDepth[ii]=np.array(name_of_location[:-int(lenloc/2)]+row[-2:])

    Geol[:,1]=np.array([np.float(ii) for ii in Geol[:,1]])
    Geol[:,2]=np.array([np.float(ii) for ii in Geol[:,2]])
    
    if add_azimuth==False:
        DH_Azimuth=np.full((wellSites.shape[0]),0)
    elif add_azimuth == True :
        DH_Azimuth=compute_azimuth(easting = np.array([np.float(ii) 
                                                       for ii in wellSites[:,1]]),
                                   northing =np.array([np.float(ii)
                                                       for ii in wellSites[:,2]]),
                                   utm_zone=utm_zone)

        
    DH_Azimuth=DH_Azimuth.reshape((DH_Azimuth.shape[0],1))
    
    
    WellData=np.concatenate((wellSites,DH_Azimuth,
                             DH_PlanDepth.reshape((DH_PlanDepth.shape[0],1)),
                             DH_Decr.reshape((DH_Decr.shape[0],1))), axis=1)
    GeolData=Geol.copy()

    #-----write Report---
    
    reg_lines.append(text[6]+'\n')
    # reg_lines.append(text[7]+"\n")
    
    reg_lines.append(text[9]+'\n')
    reg_lines.append("".join(['{0:>12}'.format(ss) for 
                              ss in text[7].split(",")]) +'\n')
    for rowline in WellData :
        reg_lines.append(''.join(["{0:>12}".format(ss) for
                                  ss in rowline.tolist()])+"\n")
        
    reg_lines.append(text[8]+"\n")
    reg_lines.append("".join(['{0:>12}'.format(ss) for ss 
                              in text[10].split(",")]) +'\n')
    for ii , row in enumerate(GeolData):
        reg_lines.append(''.join(["{0:>12}".format(ss) for
                                  ss in row.tolist()])+"\n")
        
    if add_geochemistry_sample==True:
        SampleData=build_geochemistry_sample()
        reg_lines.append(text[11]+'\n')
        reg_lines.append("".join(['{0:>12}'.format(ss) for ss 
                                  in text[12].split(",")]) +'\n')
        for ii , row in enumerate(SampleData):
            reg_lines.append(''.join(["{0:>12}".format(ss) for
                                      ss in row.tolist()])+"\n")
    else :
        SampleData=None        
        

    with open("{0}_wellReport_".format(name_of_location),"w") as fid:
        # for ii in reg_lines :
        fid.writelines(reg_lines)
    fid.close()
    
    #---end write report---
    
    if report_path is None:
        report_path=os.getcwd()
    elif report_path is not None :
        if os.path.exists(report_path):
            shutil.move((os.path.join(os.getcwd(),"{0}_wellReport_".\
                                      format(name_of_location))),report_path)
        else :
            raise OSError (
                "The path does not exit.Try to put the right path")
            warnings.warn (
            "ignore","the report_path doesn't match properly.Try to fix it !")
        
    
    
    return (name_of_location, WellData , GeolData, SampleData)
        
        
def compute_azimuth(easting, northing, utm_zone="49N", extrapolate=False):
    
    """
    Parameters
    ----------
        * easting : np.ndarray
                Easting value of coordinates _UTM_WGS84 
             
        * northing : np.ndarray
                Northing value of coordinates._UTM_WGS84
            
        * utm_zone : str, optional
                the utm_zone . if None try to get is through 
                gis.get_utm_zone(latitude, longitude). 
                latitude and longitude must be on degree decimals.
                The default is "49N".
        * extrapolate : bool , 
                for other purpose , user can extrapolate azimuth value ,
                in order to get the sizesize as 
                the easting and northing size. The the value will 
                repositionate at each point data were collected. 
                    Default is False as originally azimuth computation . 

    Returns
    -------
        np.ndarray
            azimuth.
        
    :Example: 
        
        >>> import numpy as np
        >>> import gis_tools as gis
        >>>  easting=[477205.6935,477261.7258,477336.4355,477373.7903,477448.5,
        ...  477532.5484,477588.5806,477616.5968]
        >>>   northing=[2830978.218, 2830944.879,2830900.427, 2830878.202,2830833.75,
        ...                  2830783.742,2830750.403,2830733.734]
        >>>  test=compute_azimuth(easting=np.array(easting), 
        ...                      northing=np.array(northing), utm_zone="49N")
        >>>  print(test)
    """
    #---**** method to compute azimuth****----
    
    reference_ellipsoid=23
    
    lat,long=gis.utm_to_ll(reference_ellipsoid=reference_ellipsoid,
                                northing=northing, easting=easting, zone=utm_zone)
    
    #i, idx, ic_=0,0,pi/180
    azimuth=0
    
    i,ic_=0,np.pi /180
    
    while i < lat.shape[0]:
        xl=np.cos(lat[i]*ic_)*np.sin(lat[i+1]*ic_) - np.sin(lat[i]*ic_)\
            *np.cos(lat[i+1]*ic_)*np.cos((long[i+1]-long[i])*ic_)
        yl=np.sin((long[i+1]-long[i])*ic_)*np.cos(lat[i+1])
        azim=np.arctan2(yl,xl)
        azimuth=np.append(azimuth, azim)
        i=i+1
        if i==lat.shape[0]-1 :
            # azimuth.append(0)
            break
    
    
    if extrapolate is True : 
        # interpolate azimuth to find the azimuth to first station considered to 0.
    
        ff=spi.interp1d(x=np.arange(1,azimuth.size),
                                   y=azimuth[1:], fill_value='extrapolate')
        y_new , azim = ff(0),np.ones_like(azimuth)
        azim[0], azim[1:] = y_new , azimuth[1:]
    else : 
        azim=azimuth[1:] # substract the zero value added for computation as origin.
    #convert to degree : modulo 45degree
    azim = np.apply_along_axis(lambda zz : zz * 180/np.pi , 0, azim) 
    
    
    return np.around(azim,3)
        
def build_geochemistry_sample():
    """
    Build geochemistry_sample_data
    
    Raises
    ------
        Process to build geochemistry sample data manually .

    Returns
    -------
       np.ndarray
          Sample ,Geochemistry sample Data.

    :Example:
        
        >>> geoch=build_geochemistry_sample()
        >>> print(geoch)
        ... sampleData
        ... [['S0X4' '0' '254.0' 'PUP']
        ...     ['S0X4' '254' '521.0' 'mg']
        ...     ['S0X4' '521' '625.0' 'tut']
        ...     ['S0X4' '625' '984.0' 'suj']
        ...     ['S0X2' '0' '19.0' 'pup']
        ...     ['S0X2' '19' '425.0' 'hut']
        ...     ['S0X2' '425' '510.0' 'mgt']
        ...     ['S0X2' '510' '923.2' 'pyt']]
    """

    tempsamp,SampleList=[],[]
    comp=-1
    while 1:
        print('Enter Hole Name or <Enter/end> to stop:',end='')
        holeName=input()
        if holeName=="" or holeName.lower() in ["stop","end","enter",
                                                "finish","close"]:
            Sample=concat_array_from_list(list_of_array=SampleList,
                                          concat_axis=0)
            break
        comp=comp+1
        samp_buttom=np.float(input("Enter the buttom of the sampling (m):"))
        
        while holeName:
            if comp==-1:
                samP=concat_array_from_list(list_of_array=tempsamp,concat_axis=0)
                tempsamp=[]
                break
            print("Enter the sampling levels:",end='')
            samplevel=input()    
            samplevel=dump_comma(input_car=samplevel, max_value=12, 
                            carType='value')
            samp_ar=np.array(list(samplevel))
            samp_ar=samp_ar.reshape((samp_ar.shape[0],1))
            
            dh_to=samp_ar[1:]
            dh_to=np.append(dh_to,samp_buttom)
            dh_to=dh_to.reshape((dh_to.shape[0],1))
            
            print("Enter the samples' names:",end='')
            sampName=input()
            sampName=dump_comma(input_car=sampName, max_value=samp_ar.shape[0], 
                            carType='mixed')
            sampName_ar=np.array(list(sampName))
            sampName_ar=sampName_ar.reshape((sampName_ar.shape[0],1))
            try :
                holes_names=np.full((sampName_ar.shape[0],1),holeName)
                samfromto=np.concatenate((
                    holes_names,samp_ar,dh_to,sampName_ar),axis=1)
                
            except Exception as e :
                raise ("IndexError!, arrrays sample_DH_From:{0} &DH_To :{1}&"\
                       " 'Sample':{2} doesn't not match proprerrly.{3}".\
                           format(samp_ar.shape,dh_to.shape,sampName_ar.shape, e))
                warnings.warn("IndexError !, dimentional problem."\
                              " please check np.ndarrays.shape.")
                _logger.warn("IndexError !, dimentional problem."\
                              " please check np.ndarrays.shape.")
            tempsamp.append(samfromto)
            comp=-1
        SampleList.append(samP)
        print("\n")
        
    return Sample



def parse_wellData(filename=None, include_azimuth=False,
                   utm_zone="49N"):
    """
    Function to parse well information in*csv file 

    Parameters
    ----------
        * filename : str, optional
                full path to parser file, The default is None.
                
        * include_azimuth: bool , 
            Way to compute azimuth automatically 
            
        * utm_zone : str, 
            set coordinate _utm_WGS84. Defaut is 49N

    Raises
    ------
        FileNotFoundError
            if typical file deoesnt match the *csv file.

    Returns
    -------
       location:  str
            Name of location .
            
       WellData : np.ndarray
              Specificy the collar Data .
              
        GeoData : np.ndarray
             specify the geology data .
             
        SampleData : TYPE
            geochemistry sample Data.


    :Example: 
        
        >>> import numpy as np
        >>> dir_=r"F:\OneDrive\Python\CodesExercices\ex_avgfiles\modules"
        >>> parse_=parse_wellData(filename='Drill&GeologydataT.csv')
        >>> print("NameOflocation:\n",parse_[0])
        >>> print("WellData:\n",parse_[1])
        >>> print("GeoData:\n",parse_[2])
        >>> print("Sample:\n",parse_[3])
    """
    
    
    
    identity=["DH_Hole (ID)","DH_East","DH_North","DH_Dip",
              "Elevation" ,'DH_Azimuth',"DH_Top","DH_Bottom",
              "DH_PlanDepth","DH_Decr","Mask "]
    
    car=",".join([ss for ss in identity])
    # print(car)
    
    #ckeck the if it is the correct file
    _flag=0
    if filename is None :
        filename=[file for file in os.listdir(os.getcwd()) \
                  if (os.path.isfile(file)and file.endswith(".csv"))]
        # print(filename)
        if np.iterable (filename):
            for file in filename :
                with open (file,"r", encoding="utf-8") as f :
                    data=f.readlines()
                    _flag=1
        else :
            _logger.error('The {0} doesnt not match the *csv file'\
                          ' You must convert file on *.csv format'.format(filename))
            warnings.error("The input file is wrong ! only *.csv file "\
                          " can be parsed.")
    elif filename is not None :
        assert filename.endswith(".csv"), "The input file {0} is not in *.csv format"
        with open (filename,'r', encoding='utf-8') as f:
            data=f.readlines()   
            _flag=1
            
    if _flag==1 :
        try :
            # print(data[0])
            head=data[0].strip("'\ufeff").strip('\n').split(',')[:-1]
            head=_nonevalue_checker(list_of_value=head,
                                    value_to_delete='')
            chk=[1 for ii ,value in enumerate( head) if value ==identity[ii]]
            # data[0]=head
            if not all(chk):
                _logger.error('The {0} doesnt not match the correct file'\
                              ' to parse drill data'.format(filename))
                warnings.warn("The input file is wrong ! must "\
                              "input the correct file to be parsed.")
        except Exception as e :
            raise FileNotFoundError("The *csv file does no match the well file",e)
            
        # process to parse all data 
        # coll,geol,samp=[],[],[]

    for ss, elm in enumerate (data):
        elm=elm.split(',')
        for ii, value in enumerate(elm):
            if value=='' or value=='\n':
                elm=elm[:ii]
        data[ss]=elm
        
    data[0]=head

    [data.pop(jj)for jj, val in enumerate(data) if val==[]]

    if data[-1]==[]:
        data=data[:-1]
        
    ## final check ###
    safeData=_nonelist_checker(data=data, _checker=True ,
                  list_to_delete=['\n'])
    data=safeData[2]
    # identify collar , geology dans sample data 
    comp=0
    for ss , elm in enumerate(data):
        if elm[0].lower()=='geology':
            coll=data[:ss]
            comp=ss
        if elm[0].lower() =="sample":
            geol=data[comp+1:ss]
            samp=data[ss+1:]
            
    # build numpy data array 
    collar_list=coll[1:]
    # print(collar_list)
    collar_ar=np.zeros((len(collar_list), len(identity)),dtype='<U12')
    for ii, row in enumerate(collar_ar):
        collar_ar[ii:,:len(collar_list[ii])]= np.array(collar_list[ii])
    
    bottom_ar=collar_ar[:,7]
    # print(bottom_ar)
        
    geol_list=geol[1:]
    geol_ar=_order_well (data=geol_list,bottom_value=bottom_ar)
    samp_list=samp[1:]
    samp_ar=_order_well (data=samp_list,bottom_value=bottom_ar)
    
    name_of_location =filename[:-3]
    # find Description 
    DH_PlanDepth=collar_ar[:,8]
    DH_Decr=collar_ar[:,9]
    
    lenloc=len(name_of_location)
    
 
    for ss , singleArray in enumerate (DH_PlanDepth):
        # print(singleArray)
        if singleArray == ''or singleArray is None :
            singleArray=name_of_location[:-int(lenloc/2)]+collar_ar[ss,0][-2:]
            DH_PlanDepth[ss]=singleArray
            if DH_Decr[ss] ==''or DH_Decr[ss] is None :
                DH_Decr[ss] = "wdanx"+collar_ar[ss,0][0]+\
                    name_of_location.lower()[0]+collar_ar[ss,0][-1]
    
    collar_ar[:,8]=DH_PlanDepth
    collar_ar[:,9]=DH_Decr
    
    if include_azimuth==False:
        DH_Azimuth=np.full((collar_ar.shape[0]),0)
    elif include_azimuth== True:
        DH_Azimuth=compute_azimuth(easting = np.array([np.float(ii) 
                                                       for ii in collar_ar[:,1]]),
                                  northing = np.array([np.float(ii) 
                                                       for ii in collar_ar[:,2]]),
                                  utm_zone=utm_zone, extrapolate=True)
    
    collar_ar[:,5]=DH_Azimuth
    

    name_of_location=name_of_location.capitalize()
    WellData,GeoData,SampleData=collar_ar,geol_ar, samp_ar 

    return (name_of_location, WellData,GeoData,SampleData)
                   

            
def _nonelist_checker(data, _checker=False ,
                      list_to_delete=['\n']):
    """
    Function to delete a special item on list in data.
    Any item you want to delete is acceptable as long as item is on a list.
    
    Parameters
    ----------
        * data : list
             container of list. Data must contain others list.
             the element to delete should be on list.
             
        *  _checker : bool, optional
              The default is False.
             
        * list_to_delete : TYPE, optional
                The default is ['\n'].

    Returns
    -------
        _occ : int
            number of occurence.
        num_turn : int
            number of turns to elimate the value.
        data : list
           data safeted exempt of the value we want to delete.
        
    :Example: 
        
        >>> import numpy as np 
        >>> listtest =[['DH_Hole', 'Thick01', 'Thick02', 'Thick03',
        ...   'Thick04', 'sample02', 'sample03'], 
        ...   ['sample04'], [], ['\n'], 
        ...  ['S01', '98.62776918', '204.7500461', '420.0266651'], ['prt'],
        ...  ['pup', 'pzs'],[], ['papate04', '\n'], 
        ...  ['S02', '174.4293956'], [], ['400.12', '974.8945704'],
        ...  ['pup', 'prt', 'pup', 'pzs', '', '\n'],
        ...  ['saple07'], [], '',  ['sample04'], ['\n'],
        ...  [''], [313.9043882], [''], [''], ['2'], [''], ['2'], [''], [''], ['\n'], 
        ...  [''], ['968.82'], [], [],[], [''],[ 0.36], [''], ['\n']]
        >>> ss=_nonelist_checker(data=listtest, _checker=True,
        ...                        list_to_delete=[''])
        >>> print(ss)
    """
    
    _occ,num_turn=0,0
    
    
    if _checker is False:
        _occ=0
        return _occ, num_turn, data 
    
   
    while _checker is True :
        if list_to_delete in data :
            for indix , elem_chker in enumerate(data):
                if list_to_delete == elem_chker :
                    _occ=_occ+1
                    del data[indix]
                    if data[-1]==list_to_delete:
                        _occ +=1
                        # data=data[:-1]
                        data.pop()
        elif list_to_delete not in data :
            _checker =False
        num_turn+=1

        
    return _occ,num_turn,data
        
def _order_well (data,**kwargs):
    """
    Function to reorganize value , depth rock and depth-sample 
    the program controls  the input depth value and compare it . 
    with the bottom. It will pay attention that bottom depth must
    be greater or egual of any other value. In the same time ,
    the program check if value entered are sorted on ascending order . 
    well must go deep ( less value to great value). Negative values of
    depths are not acceptable.
    
    Parameters
    ----------
        * data : list,
                data contains list of well thickness and rock description .
    
        * bottom_value  : np.ndarray float
                value of bottom . it may the basement extrapolation.
                default is 1.023 km

    Returns
    -------
         np.ndarray
            data, data aranged to [DH_Hole, DH_From, DH_To, Rock.] arrays
        
    :Example:
        
        >>> import numpy as np
        >>> listtest =[['DH_Hole', 'Thick01', 'Thick02', 'Thick03',
        ...           'Thick04','Rock01', 'Rock02', 'Rock03', 'Rock04'],
        >>> ['S01', '0.0', '98.62776918', '204.7500461','420.0266651',
        ...     'GRT', 'ATRK', 'GRT', 'ROCK'],
        >>> ['S02', '174.4293956', '313.9043882', '400.12', '974.8945704',
        ...     'GRT', 'ATRK', 'GRT', 'ROCK']]
        >>> print(listtest[1:])
        >>> ss=_order_well(listtest[1:])
        >>> print(ss)
    """
    
    this_function_name=inspect.getframeinfo(inspect.currentframe())[2]

    bottomgeo=kwargs.pop("bottom_value", None)
    # bottomsamp=kwargs.pop("sample_bottom_value",None)
    # if type(bottomgeo) is np.ndarray :_flag=1
    # else : _flag=0
    
    temp=[]

    _logger.info ("You pass by {0} function! Thin now , everything is ok. ".
                  format(this_function_name))
    
    for jj, value in enumerate (data):
        thickness_len,dial1,dial2=intell_index(datalist=value[1:]) #call intell_index
        value=np.array(value)

        dh_from=value[1:thickness_len+1]
        dh_to =np.zeros((thickness_len,), dtype="<U12")
                # ---- check the last value given        
        max_given_bottom=np.max(np.array([np.float(ii) for ii in dh_from])) 
                # ----it may be less than bottom value                                                                .
        dh_geo=value[thickness_len+1:]
        dh_to=dh_from[1:]
        # if _flag==0 :
        if bottomgeo[jj] =="" or bottomgeo[jj]==None :
            bottomgeoidx=1023.
            if max_given_bottom > bottomgeoidx:
                _logger.warn("value {0} is greater than the Bottom depth {1}".
                             format(max_given_bottom ,bottomgeoidx))
                warnings.warn (
                    "Given values have a value greater than the depth !")
                
            dh_to=np.append(dh_to,bottomgeoidx)
        else: #elif :_flag==1 :         # check the bottom , any values given 
                                        # must be less or egual to depth not greater.
            if max_given_bottom> np.float(bottomgeo[jj]):
                _logger.warn("value {0} is greater than the Bottom depth {1}".
                             format(max_given_bottom ,bottomgeo[jj]))
                warnings.warn ("Given values have a value greater than the depth !")
            dh_to=np.append(dh_to,bottomgeo[jj])
            
        dh_hole=np.full((thickness_len),value[0])
        
        temp.append(np.concatenate((dh_hole.reshape((dh_hole.shape[0],1)),
                                    dh_from.reshape((dh_from.shape[0],1)),
                                    dh_to.reshape((dh_to.shape[0],1)),
                                    dh_geo.reshape((dh_geo.shape[0],1))),axis=1 ))
        
    data=concat_array_from_list(list_of_array=temp, concat_axis=0)
    
    return data
                                
        
def intell_index (datalist,assembly_dials =False):
    """
    function to search index to differency value to string element like 
    geological rocks and geologicals samples. It check that value are sorted
    in ascending order.

    Parameters
    ----------
        * datalist : list
                list of element : may contain value and rocks or sample .
        * assembly_dials : list, optional
                separate on two list : values and rocks or samples. 
                The default is False.

    Returns
    -------
        index: int
             index of breaking up.
        first_dial: list , 
           first sclice of value part 
        secund_dial: list , 
          second slice of rocks or sample part.
        assembly : list 
             list of first_dial and second_dial
    
    :Example: 
        
        >>> import numpy as np
        >>> listtest =[['DH_Hole', 'Thick01', 'Thick02', 'Thick03',
        ...           'Thick04','Rock01', 'Rock02', 'Rock03', 'Rock04'],
        ...           ['S01', '0.0', '98.62776918', '204.7500461','420.0266651','520', 'GRT', 
        ...            'ATRK', 'GRT', 'ROCK','GRANODIORITE'],
        ...           ['S02', '174.4293956', '313.9043882','974.8945704', 'GRT', 'ATRK', 'GRT']]
        >>> listtest2=[listtest[1][1:],listtest[2][1:]]
        >>> for ii in listtest2 :
        >>> op=intell_index(datalist=ii)
        >>> print("index:\n",op [0])
        >>> print('firstDials :\n',op [1])
        >>> print('secondDials:\n',op [2])
    """
    # assembly_dials=[]
    max_=0              # way to check whether values are in sort (ascending =True) order 
                        # because we go to deep (first value must be less than the next none)
    for ii, value in enumerate (datalist):
        try : 
            thick=float(value)
            if thick >= max_:
                max_=thick 
            else :
                _logger.warning(
                    "the input value {0} is less than the previous one."
                " Please enter value greater than {1}.".format(thick, max_) )
                warnings.warn(
                    "Value {1} must be greater than the previous value {0}."\
                    " Must change on your input data.".format(thick,max_))
        except :
            # pass
            indexi=ii
            break
        
    first_dial=datalist[:indexi]
    second_dial =datalist[indexi:]

    if assembly_dials:
        
        assembly_dials=[first_dial,second_dial]
        
        return indexi, assembly_dials
        
    return indexi, first_dial, second_dial

def _nonevalue_checker (list_of_value, value_to_delete=None):
    """
    Function similar to _nonelist_checker. the function deletes the specific 
    value on the list whatever the number of repetition of value_to_delete.
    The difference with none list checker is value to delete 
    may not necessary be on list.
    
    Parameters
    ----------
        * list_of_value : list
                list to check.
        * value_to_delete : TYPE, optional
            specific value to delete. The default is ''.

    Returns
    -------
        list
            list_of_value , safe list without the deleted value .
    
    :Example: 
        
        >>> import numpy as np 
        >>> test=['DH_Hole (ID)', 'DH_East', 'DH_North',
        ...          'DH_Dip', 'Elevation ', 'DH_Azimuth', 
        ...            'DH_Top', 'DH_Bottom', 'DH_PlanDepth', 'DH_Decr', 
        ...            'Mask', '', '', '', '']
        >>> test0= _nonevalue_checker (list_of_value=test)
        >>> print(test0)
    """
    if value_to_delete is None :
        value_to_delete  =''
    
    if type (list_of_value) is not list :
        list_of_value=list(list_of_value)
        
    start_point=1
    while start_point ==1 :
        if value_to_delete in list_of_value :
            [list_of_value.pop(ii) for  ii, elm in\
             enumerate (list_of_value) if elm==value_to_delete]
        elif value_to_delete not in list_of_value :
            start_point=0 # not necessary , just for secure the loop. 
            break           # be sure one case or onother , it will break
    return list_of_value 
        
def _strip_item(item_to_clean, item=None, multi_space=12):
    """
    Function to strip item around string values.  if the item to clean is None or 
    item-to clean is "''", function will return None value

    Parameters
    ----------
        * item_to_clean : list or np.ndarray of string 
                 List to strip item.
        * cleaner : str , optional
                item to clean , it may change according the use. The default is ''.
        * multi_space : int, optional
                degree of repetition may find around the item. The default is 12.

    Returns
    -------
        list or ndarray
            item_to_clean , cleaned item 
            
    :Example: 
        
     >>> import numpy as np
     >>> new_data=_strip_item (item_to_clean=np.array(
         ['      ss_data','    pati   ']))
     >>>  print(np.array(['      ss_data','    pati   ']))
     ... print(new_data)

    """
    if item==None :item = ' '
    
    cleaner =[(''+ ii*'{0}'.format(item)) for ii in range(multi_space)]
    
    if type(item_to_clean ) != list :#or type(item_to_clean ) !=np.ndarray:
        if type(item_to_clean ) !=np.ndarray:
            item_to_clean=[item_to_clean]
    if item_to_clean in cleaner or item_to_clean ==['']:
        warnings.warn ('No data found in <item_to_clean :{}> We gonna return None.')
        return None 

  
    try : 
        multi_space=int(multi_space)
    except : 
        raise TypeError('argument <multplier> must be'\
                        ' an integer not {0}'.format(type(multi_space)))
    
    for jj, ss in enumerate(item_to_clean) : 
        for space in cleaner:
            if space in ss :
                new_ss=ss.strip(space)
                item_to_clean[jj]=new_ss
    
    return item_to_clean
    
def _cross_eraser (data , to_del, deep_cleaner =False):
    """
    Function to delete some item present in another list. It may cheCk deeper 

    Parameters
    ----------
        * data : list
                Main data user want to filter.
        * to_del : list
                list of item you want to delete present on the main data.
        * deep_cleaner : bool, optional
                Way to deeply check. Sometimes the values are uncleaned and 
            capitalizeed . this way must not find their safety correspondace 
            then the algorth must clean item and to match all at
            the same time before eraisng.
            The *default* is False.

    Returns
    -------
        list
         data , list erased.

    :Example: 
        
        >>> data =['Z.mwgt','Z.pwgt','Freq',' Tx.Amp','E.mag','   E.phz',
        ...          '   B.mag','   B.phz','   Z.mag', '   Zphz  ']
        >>> data2=['   B.phz','   Z.mag',]
        ...     remain_data =cross_eraser(data=data, to_del=data2, 
        ...                              deep_cleaner=True)
        >>> print(remain_data)
    """
    
    data , to_del=_strip_item(item_to_clean=data), _strip_item(
        item_to_clean=to_del)
    if deep_cleaner : 
        data, to_del =[ii.lower() for ii in data], [jj.lower() 
                                                    for jj in to_del]
        
    for index, item in enumerate(data): 
        while item in to_del :
            del data[index]
            if index==len(data)-1 :
                break

    return data 

def _remove_str_word (char, word_to_remove, deep_remove=False):
    """
    Small funnction to remove a word present on  astring character 
    whatever the number of times it will repeated.
    
    Parameters
    ----------
        * char : str
                may the the str phrases or sentences . main items.
        * word_to_remove : str
                specific word to remove.
        * deep_remove : bool, optional
                use the lower case to remove the word even the word is uppercased 
                of capitalized. The default is False.

    Returns
    -------
        str 
            char , new_char without the removed word .
        
    :Example: 
        
        >>> from pycsamt.utils  import func_utils as func
        >>> ch ='AMTAVG 7.76: "K1.fld", Dated 99-01-01,AMTAVG, 
        ...    Processed 11 Jul 17 AMTAVG'
        >>> ss=func._remove_str_word(char=ch, word_to_remove='AMTAVG', 
        ...                             deep_remove=False)
        >>> print(ss)
    """
    if type(char) is not str : char =str(char)
    if type(word_to_remove) is not str : word_to_remove=str(word_to_remove)
    
    if deep_remove == True :
        word_to_remove, char =word_to_remove.lower(),char.lower()

    if word_to_remove not in char :
        return char

    while word_to_remove in char : 
        if word_to_remove not in char : 
            break 
        index_wr = char.find(word_to_remove)
        remain_len=index_wr+len(word_to_remove)
        char=char[:index_wr]+char[remain_len:]

    return char

def stn_check_split_type(data_lines): 
    """
    Read data_line and check for data line the presence of 
    split_type < ',' or ' ', or any other marks.>
    Threshold is assume to be third of total data length.
    
    :params data_lines: list of data to parse . 
    :type data_lines: list 
 
    :returns: The split _type
    :rtype: str
    
    :Example: 
        
        >>> from pycsamt.utils  import func_utils as func
        >>> path =  os.path.join(os.environ["pyCSAMT"], 
                          'csamtpy','data', K6.stn)
        >>> with open (path, 'r', encoding='utf8') as f : 
        ...                     data= f.readlines()
        >>>  print(func.stn_check_split_type(data_lines=data))
            
    """

    split_type =[',', ':',' ',';' ]
    data_to_read =[]
    if isinstance(data_lines, np.ndarray): # change the data if data is not dtype string elements.
        if data_lines.dtype in ['float', 'int', 'complex']: 
            data_lines=data_lines.astype('<U12')
        data_lines= data_lines.tolist()
        
    if isinstance(data_lines, list):
        for ii, item in enumerate(data_lines[:int(len(data_lines)/3)]):
             data_to_read.append(item)
             data_to_read=[''.join([str(item) for item in data_to_read])] # be sure the list is str item . 

    elif isinstance(data_lines, str): data_to_read=[str(data_lines)]
    
    for jj, sep  in enumerate(split_type) :
        if data_to_read[0].find(sep) > 0 :
            if data_to_read[0].count(sep) >= 2 * len(data_lines)/3:
                if sep == ' ': return  None  # use None more conventional 
                else : return sep 

def minimum_parser_to_write_edi (edilines, parser = '='):
    """
    This fonction validates edifile for writing , string with egal.we assume that 
    dictionnary in list will be for definemeasurment E and H fied. 
    
    :param edilines: list of item to parse 
    :type edilines: list 
    
    :param parser: the egal is use  to parser edifile .
                    can be changed, default is `=`
    :type parser: str 
  
    """
    if isinstance(edilines,list):
        if isinstance(edilines , tuple) : edilines =list(edilines)
        else :raise TypeError('<Edilines> Must be on list')
    for ii, lines in enumerate(edilines) :
        if isinstance(lines, dict):continue 
        elif lines.find('=') <0 : 
            raise 'None <"="> found on this item<{0}> of '
            ' the edilines list. list can not'\
            ' be parsed.Please put egal between key and value '.format(
                edilines[ii])
    
    return edilines 
            

def round_dipole_length(value, round_value =5.): 
    """ 
    small function to graduate dipole length 5 to 5. Goes to be reality and 
    simple computation .
    
    :param value: value of dipole length 
    :type value: float 
    
    :returns: value of dipole length rounded 5 to 5 
    :rtype: float
    """ 
    mm = value % round_value 
    if mm < 3 :return np.around(value - mm)
    elif mm >= 3 and mm < 7 :return np.around(value -mm +round_value) 
    else:return np.around(value - mm +10.)
    



    

    
    
    
    
    
    
    

    
        