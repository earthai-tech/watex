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
import copy 
import pandas as pd 
import  unittest 
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
    Logging , EM , EMAP, 
   )
from tests import  ( 
    TEST_TEMP_DIR,  
    make_temp_dir 
    ) 

    
from tests.__init__ import reset_matplotlib, diff_files

class TestElectrical (unittest.TestCase): 
    # constraints elaborations 
    constraints = {"S10":'Prohibited site', 
                   "S50": "Marsh zone", 
                   "S70": "Heritage site"
                   }
    # electrical resistivity data generating 
    erp_data = make_erp( n_stations =100 , seed =123 , max_rho = 1e5 ,
                        min_rho = 1e1, as_frame =True)
    auto_detection = True 
    # vertical electrical data generating 
    search = 20 
    ves_data = make_ves (samples =50 , max_depth= 200 , order ='+',
                         max_rho=1e5, seed = 123).frame 

    def test_ResistivityProfiling (self ): 
        
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

    def test_VerticalSounding (self): 
        """ Make test for |VES| , Compute Parameters with simple run """
        
        vesobj = VerticalSounding(search = self.search ).fit(self.ves_data ) 
        dcobj = DCSounding(search = self.search ).fit(self.ves_data )
        self.assertAlmostEqual(vesobj.ohmic_area_, dcobj.site1.ohmic_area_)
        
        dcobj.summary(return_table =True)
        vesobj.summary(return_table= True ) 
        
        self.assertAlmostEqual(dcobj.nareas_, vesobj.nareas_ )
   
class TestHydro (unittest.TestCase ): 
    """ Test Hydrogeological module"""
    
    #*** Get the data **** 
    HDATA = load_hlogs(key ='h502 h2601', drop_observations =True ).frame 
    

    def test_mxs (self  ): 
        """ Test Mixture Learning Strategy (MXS). 
        Predict NGA labels and MXS  """
        # aggreate the two boreholes data  the two boreholes data 
        # --> shape = ( 218, 17)
        hdata = pd.concat ([ load_hlogs().frame  
                            + load_hlogs(key ='h2601').frame ]) 
        # drop remark column 
        hdata.drop (columns ='remark', inplace =True )
        # fit NaN values using naive transformer 
        hdata = naive_imputer(hdata, mode = 'bi-impute')
        
        mxsobj = MXS(kname ='k' ).fit(hdata )
        
        mxsobj.predictNGA(n_components= 3 )
        mxsobj.makeyMXS(categorize_k=True, default_func=True)
        
    def test_label_similarity (self ): 
        # refit the data 
        mxs = MXS (kname ='k', aqname ='aquifer_group').fit(self.HDATA)
        sim = mxs.labelSimilarity() 
        
        print("label similarity groups=", sim )
 

    def test_AqGroup (self): 
        """ Test aquifer Group sections """
        hg = AqGroup (kname ='k', aqname='aquifer_group').fit(self.HDATA ) 
        print( hg.findGroups () ) 
        
    def test_AqSection (self): 
        """ Compute the section of aquifer"""
        section = AqSection (aqname ='aquifer_group', kname ='k',
                         zname ='depth_top').fit(self.HDATA) 
        self.assertEqual(len(section.findSection ()) , 2) 
        
        
    def test_Logging (self): 
        
        h = load_hlogs(key ='h2601') 
        log = Logging(kname ='k', zname='depth_top' ).fit(
                h.frame[h.feature_names])
        log.plot() 
        
class TestEM (unittest.TestCase): 
    # output the edis data as array_like 1d 
    edi_data = load_edis (key='edi' , return_data =True  )
    emobj = EM ().fit(edi_data )
        
    @classmethod 
    def setUpClass(cls):
        """
        Reset building matplotlib plot and generate tempdir inputfiles 
        
        """
        reset_matplotlib()
        cls._temp_dir = make_temp_dir(cls.__name__)


    def test_make2d (self): 
        """ make2D blocks """
        self.emobj.make2d() 
        print("emobj.freqs_.max()=", self.emobj.freqs_.max())
        print("emobj.refreq_=", self.emobj.refreq_)
        
        self.assertAlmostEqual( self.emobj.freqs_.max() , self.emobj.refreq_)
        
    def test_rewrite (self): 
        # rewrite EDI with 7 seven of edi from a clone edi_class 

        edi_sample = self.edi_data [:7]  
        self.emobj_cloned = copy.deepcopy(self.emobj)
        self.emobj_cloned.fit(edi_sample )
        self.emobj_cloned.rewrite(by='station', prefix='PS', 
                           savepath = os.path.join( 
                               TEST_TEMP_DIR, self.__class__.__name__) 
                           )
        # fix bug in :meth:`watex.methods.em.EM.rewrite`. remove 'todms' in 
        # :func:`watex.utils.exmath.scalePosition` since the latter does not 
        # longer convert data to D:MM:SS. 
        
    def test_getreference_frequency (self): 
        """ check the reference frequency"""
        # this is a naive approach since our EDI data used for the test 
        # is already preprocessed data and missing weak signal are alreay 
        # removed. Thus the reference frequency should obviously equals 
        # to the max frequency. 
        self.assertAlmostEqual(self.emobj.getreferencefrequency(), 
                               self.emobj.freq_array.max ()) 
        

class TestEMAP (unittest.TestCase): 
    # output the edis data as array_like 1d 
    edi_data = load_edis (key='edi' , return_data =True, samples = 30 )
    pobj = EMAP().fit(edi_data)
    
    def test_EMAP (self ): 
        
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
    


# if __name__=='__main__':
#     unittest.main()
    
    
                
    
    
    
    
    
    
    
    
    
    
    
