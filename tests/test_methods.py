# -*- coding: utf-8 -*-

"""
Created on Mon Jul  5 13:27:27 2021

Description:
    Test to core module . Containers of modules  :mod:`~.methods.erp.ERP` and 
    :mod:`~.methods.ves.VES`
    Test ouputfiles from rewriting and generating files , which includes:
    Reference input data are from ERP_DATA_DIR and VES_DATA_DIR
   
@author: @Daniel03

"""
import os
# import datetime
import pandas as pd 
import  unittest 
import pytest
from watex.utils import naive_imputer
from watex.datasets import (
    make_erp ,
    make_ves ,
    load_hlogs, 
    load_edis 
    )
from watex.methods import (
    ResistivityProfiling, VerticalSounding, 
    DCProfiling, DCSounding, MXS, 
    AqSection, AqGroup, 
    Logging , EM , Processing, 
   )
from watex.methods.erp import ERP 
from tests import  ( 
    ERP_PATH, TEST_TEMP_DIR,  
    make_temp_dir 
    ) 

from tests import erp_test_location_name 
from tests.__init__ import reset_matplotlib, watexlog, diff_files

#*** Get the data **** 
HDATA = load_hlogs(key ='*', drop_observations =True ).frame 

class TestERP(unittest.TestCase):
    """
    Test electrical resistivity profile  and compute geo-lectrical features 
    as followings : 
        - type 
        - shape 
        - sfi 
        - power 
        - magnitude
        - anr
        - select_best_point
        - select_best_value
    """
    dipole_length = 10. 

    @classmethod 
    def setUpClass(cls):
        """
        Reset building matplotlib plot and generate tempdir inputfiles 
        
        """
        reset_matplotlib()
        cls._temp_dir = make_temp_dir(cls.__name__)

    def setUp(self): 
        
        if not os.path.isdir(TEST_TEMP_DIR):
            print('--> outdir not exist , set to None !')
            watexlog.get_watex_logger().error('Outdir does not exist !')
        
    @pytest.mark.skip(reason='Test succeeded on Windox env. With Python 3.7'
                      'but required latest version of pandas library on '
                      'Linux env.')  
    def test_geo_params(self):
        """
        Test geo-electricals features computations 
        
        """
        geoCounter=0
        for option in [True, False]: 
            if option is False : pos_boundaries = (90, 130)
            else : pos_boundaries  =None
            anomaly_obj =ERP(erp_fn = os.path.join(ERP_PATH,
                                                   erp_test_location_name), 
                         auto=option, dipole_length =self.dipole_length, 
                         posMinMax= pos_boundaries, turn_on =True)
            
            for anObj  in [anomaly_obj.best_type, anomaly_obj.best_shape]:
                self.assertEqual(type(anObj),
                                str,'Type and Shape of selected anomaly'\
                                'must be a str object not {0}'.
                                format(type(anObj)))
                geoCounter +=1
            for anObj in [anomaly_obj.select_best_point_,
                          anomaly_obj.select_best_value_, 
                          anomaly_obj.best_magnitude, 
                          anomaly_obj.best_power, 
                          anomaly_obj.best_sfi, 
                          anomaly_obj.best_anr]: 
                try : 
                    
                    self.assertEqual(type(float(anObj)), 
                                    float, 'ValueError, must be `float`'\
                                        ' or integrer value.')
                except : 
                    watexlog().get_watex_logger().error(
                        'Something wrong happen when computing '
                        'geo-electrical features.')
                else: 
                    geoCounter +=1
                    
            self.assertEqual(geoCounter,8, 'Parameters count shoud be'
                             '8 not {0}'.format(geoCounter)) 
            geoCounter =0 
            
class TestResistivityProfiling (unittest.TestCase): 
    
    constraints = {"S10":'Prohibited site', 
                   "S50": "Marsh zone", 
                   "S70": "Heritage site"
                   }
    erp_data = make_erp( n_stations =100 , seed =123 , max_rho = 1e5 ,
                        min_rho = 1e1, as_frame =True)
    auto_detection = True 

    def test_summary (self ): 
        
        # automatic detection 
        erpobj = ResistivityProfiling(auto= self.auto_detection 
                                      ).fit(self.erp_data )
        erpobj_c = ResistivityProfiling(
            auto=self.auto_detection, constraints=self.constraints
            ).fit(self.erp_data ) 
        
        # sstation detection  
        erpobj_s_c = ResistivityProfiling(
            station= 'S25', constraints=self.constraints, coerce=True, 
                                        ).fit(self.erp_data ) 
        
        erpobj_s= ResistivityProfiling(station= 'S25', ).fit(self.erp_data )
        
        # get data from table 
        erpobj.summary() ; erpobj_c.summary()
        erpobj_s.summary(); erpobj_s_c.summary() 
        
        # compute DC data from the same datasets 
        dc_res = DCProfiling(stations = 'S25').fit(self.erp_data )
        # assert whether both values are the same 
        
        for param  in ("sfi", 'power', 'magnitude', 'shape', 'type'): 
            self.assertAlmostEqual(getattr (erpobj_s, param +'_' ),
                                   getattr (dc_res.line1, f"{param}_")
                                   ) 


class TestVerticalSounding (unittest.TestCase): 
    """ Make test for |VES| """
    search = 20 
    ves_data = make_ves (samples =50 , max_depth= 200 , order ='+',
                         max_rho=1e5, seed = 123).frame 
    
    def test_summary (self): 
        """ Compute Parameters with simple run """
        
        vesobj = VerticalSounding(search = self.search ).fit(self.ves_data ) 
        dcobj = DCSounding(search = self.search ).fit(self.ves_data )
        self.assertAlmostEquals(vesobj.ohmic_area_, dcobj.site1.ohmic_area_)
        
        dcobj.summary(return_table =True)
        vesobj.summary(return_table= True ) 
        
        self.assertAlmostEqual(dcobj.nareas_, vesobj.nareas_ )
   

class TestMXS (unittest.TestCase ): 
    "Test Mixture Learning Strategy (MXS) "
    
    # aggreate the two boreholes data  the two boreholes data 
    # --> shape = ( 218, 17)
    hdata = pd.concat ([ load_hlogs().frame  
                        + load_hlogs(key ='h2601').frame ]) 
    # drop remark column 
    hdata.drop (columns ='remark', inplace =True )
    # fit NaN values using naive transformer 
    hdata = naive_imputer(hdata, mode = 'bi-impute')
    
    mxsobj = MXS(kname ='k' ).fit(hdata )
    
    def test_mxs (self  ): 
        """ Predict NGA labels and MXS  """
        self.mxsobj.predictNGA(n_components= 3 )
        self.mxsobj.makeyMXS(categorize_k=True, default_func=True)
        
    def test_label_similarity (self ): 
        # refit the data 
        mxs = MXS (kname ='k', aqname ='aquifer_group').fit(self.hdata)
        sim = mxs.labelSimilarity() 
        
        print("label similarity groups=", sim )
        
        
class TestAqGroup(unittest.TestCase): 
    """ Test aquifer Group sections """
    
    def test_findgroup (self): 
         
        hg = AqGroup (kname ='k', aqname='aquifer_group').fit(HDATA ) 
        print( hg.findGroups () ) 
        
class TestqASection (unittest.TestCase): 
    """ Compute the section of aquifer"""
    section = AqSection (aqname ='aquifer_group', kname ='k',
                         zname ='depth_top').fit(HDATA) 

    def test_findsection (self): 
        self.assertListEqual(self.section.findSection () , [93.1, 340.35]) 
        
class TestLogging (unittest.TestCase): 
    
    h = load_hlogs(key ='h2601') 
    log = Logging(kname ='k', zname='depth_top' ).fit(
        h.frame[h.feature_names])
    
    @classmethod 
    def setUpClass(cls):
        """
        Reset building matplotlib plot and generate tempdir inputfiles 
        
        """
        reset_matplotlib()
        cls._temp_dir = make_temp_dir(cls.__name__)

    def setUp(self): 
        
        if not os.path.isdir(TEST_TEMP_DIR):
            print('--> outdir not exist , set to None !')
            watexlog.get_watex_logger().error('Outdir does not exist !')
            
    def test_log (self): 
        self.log.plot() 
        
class TestEM (unittest.TestCase): 
    # output the edis data as array_like 1d 
    edi_data = load_edis (key='edi' , return_data =True )
    
    emobj = EM ().fit(edi_data )
        
    def test_make2d (self): 
        """ make2D blocks """
        self.emobj.make2d() 
        print("emobj.freqs_.max()=", self.emobj.freqs_.max())
        print("emobj.refreq_=", self.emobj.refreq_)
        
        self.assertAlmostEqual( self.emobj.freqs_.max() , self.emobj.refreq_)
        
        
class TestProcessing (unittest.TestCase): 
    # output the edis data as array_like 1d 
    edi_data = load_edis (key='edi' , return_data =True, samples = 17 )
    pobj = Processing().fit(edi_data)
    
    def test_ama (self ): 
        
        self.pobj.ama () ; self.pobj.flma () ; self.pobj.ama () 
        
    def test_qc (self): 
        """ Compute the quality control """ 
        
        c, _ = self.pobj.qc ( tol = .6 )
        print(f"QC= {c * 100}%"  ) 
        
    def test_zrestore (self): 
        """ Restore Impedance data """
        self.pobj.zrestore () 
        
    def test_skew (self ): 
        
        for meth in ("swift", 'bahr'): 
            self.pobj.skew (method =meth) 
            
            
def compare_diff_files(refout, refexp):
    """
    Compare diff files like expected files and output files generated after 
    runnning scripts.
    
    :param refout: 
        
        list of reference output files generated after runing scripts
        
    :type refout: list 
    
    :param refexp: recreated expected files for comparison 
    :param refexp: list 

    """
    for outfile , expfile in zip(sorted(refout), 
                                   sorted(refexp)):
            unittest.TestCase.assertTrue(os.path.isfile(outfile),
                                "Ref output data file does not exist,"
                                "nothing to compare with"
                                )
            
            print(("Comparing", outfile, "and", expfile))
            
            is_identical, msg = diff_files(outfile, expfile, ignores=['Date/Time:'])
            print(msg)
            unittest.TestCase.assertTrue(is_identical, 
                            "The output file is not the same with the baseline file.")
    


if __name__=='__main__':

    unittest.main()
    
    
                
    
    
    
    
    
    
    
    
    
    
    
