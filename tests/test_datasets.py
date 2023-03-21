# -*- coding: utf-8 -*-

import pandas as pd 
import  unittest 
import watex as wx  

class TestDatasets (unittest.TestCase) :
    """ Innner datasets tests. """
    def test_load_hlogs (self ): 
        """ Test Hydro-geological datasets
        Add new features: 
            - Able  to drop observation ``remark`` into the log data 
            - collect all data and aggregate them by passing  '*'  to the 
              parameter `key`. 
        Here is an example. 
        
        """
        # --- older methods 
        # fetch both data 
        h502 = wx.fetch_data('hlogs').frame 
        h2601 = wx.fetch_data('hlogs', key ='h2601').frame 
        # concat data 
        hdata_o = pd.concat ([h502, h2601])
        print( "data size:", len(hdata_o) )
        print( "mumber of columns=", len( hdata_o.columns )) 
        print( "columns\n:", hdata_o.columns )

        # the observation ("remark") columns sometimes does not 
        # contain a meaning full detail \so it can be dropped as
        hdata_o. drop (columns = 'remark', inplace = True )

        print( "columns\n:", hdata_o.columns )

        #-- in the newest version , both ( aggregation and drop ) can be 
        # performed in a single action by setting the ``key="*"` for concatenation 
        # or ``drop_observations`` param top ``True``. here is an example. 
        # - Aggregation 
        hdata = wx.fetch_data("hlogs", key ='*').frame 
        print( "data size:", len(hdata) ) 

        print( "mumber of columns=", len( hdata.columns )) 
        # - Drop observation of h502 borehole
        hdata = wx.fetch_data("hlogs", drop_observations =True ).frame 
        print( "Does observation still exist?", "remark" in hdata.columns) 

        # or --Do both actions 
        hdata = wx.fetch_data("hlogs", key='*', drop_observations =True ).frame 

        print( "Does observation still exist ?", "remark" in hdata.columns)
        print("show new data_size:", len(hdata ))
        
        self.assertEqual(len(hdata_o), len(hdata))

    def test_fetch_data (self ) :
        """ Test the boilerplate function """
        erp_data = wx.fetch_data ('tankesse', as_frame = True )
        ves_data = wx.fetch_data('semien', index_rhoa =2 )
        hlog_data = wx.fetch_data("hlogs" , key ='*').frame 
        edi_data = wx.fetch_data("edis", key='*').frame 
        results = [ 'resistivity' in d.columns 
                   for d in ( erp_data , ves_data, hlog_data, )
                   ]
        self.assertEqual(set (results ), {True} )
        self.assertEqual("edi" in edi_data.columns, True )

        
       
# if __name__=='__main__':

#     unittest.main()