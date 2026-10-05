import unittest
import tempfile,csv
import numpy as np
from ball_geometry import flight,project,fit_flights,sample_segment
from tracknet import preprocess,heatmap_detection,filter_track
from geometry import calibrate
from ball_exports import write_ball_csv

class LearnedBallTests(unittest.TestCase):
    def test_csv_distinguishes_observed_estimated_and_unknown(self):
        rows=[{'t':1,'ball':None},{'t':2,'ball':{'pixel':[100,200],'model_score':.7}},
              {'t':3,'ball':None,'reconstruction':{'xyz':[1,2,.3],'velocity':[4,5,6]}}]
        with tempfile.TemporaryDirectory() as directory:
            path=write_ball_csv({'clip':{'start':1},'ball_track':rows},directory)
            with path.open() as stream:export=list(csv.DictReader(stream))
        self.assertEqual([r['depth_status'] for r in export],['unknown','2d_only','estimated'])
        self.assertEqual(export[0]['height_m'],'')
        self.assertEqual(export[1]['model_score'],'0.7')
        self.assertEqual(export[2]['height_m'],'0.3')
        self.assertEqual(export[2]['pixel_x'],'')

    def test_rgb_triplet_order_and_normalization(self):
        images=[np.full((8,8,3),[10,20,30],dtype='uint8'),np.full((8,8,3),[40,50,60],dtype='uint8'),np.full((8,8,3),[70,80,90],dtype='uint8')]
        tensor=preprocess(images)
        self.assertEqual(tensor.shape,(9,288,512))
        np.testing.assert_allclose(tensor[:,0,0],np.array([30,20,10,60,50,40,90,80,70])/255,atol=1e-7)

    def test_heatmap_unknown_and_pixel_scaling(self):
        h=np.zeros((288,512),np.float32)
        self.assertIsNone(heatmap_detection(h,1280,720))
        h[100:102,200:202]=.8
        d=heatmap_detection(h,1280,720)
        np.testing.assert_allclose(d['pixel'],[502.5,252.5])
        self.assertIsNone(d['height'])

    def test_temporal_filter_rejects_spike_and_stationary_paddle(self):
        rows=[{'t':i/30,'ball':{'pixel':[100+i*10,200]}} for i in range(12)]
        rows[5]['ball']['pixel']=[800,300]
        self.assertEqual(filter_track(rows,1280),1)
        self.assertIsNone(rows[5]['ball'])
        self.assertTrue(all(r['ball'] for r in rows if r is not rows[5]))
        stationary=[{'t':i/30,'ball':{'pixel':[600+i*.2,220]}} for i in range(24)]
        self.assertGreater(filter_track(stationary,1280),15)

class BallFlightTests(unittest.TestCase):
    def setUp(self):
        _,self.K,self.R,self.C,_=calibrate(1280,720,[[474,210],[818,210],[1090,536],[206,536]])

    def test_drag_flight_endpoint_and_gravity_limit(self):
        p0=np.array([1,12,.6]);p1=np.array([5,1,.037]);duration=.8
        pts,vel=flight(p0,p1,duration,.4,[0,.4,.8])
        np.testing.assert_allclose(pts[[0,-1]],[p0,p1],atol=1e-8)
        ballistic,_=flight(p0,p1,duration,1e-5,.4)
        np.testing.assert_allclose(ballistic,(p0+p1)/2-np.array([0,0,-9.81])*duration**2/8,atol=1e-4)
        self.assertGreater(pts[1,2],.6)

    def test_synthetic_track_recovers_flight_with_bounce_and_contact_prior(self):
        p0=np.array([2,13,.65]);p1=np.array([5,1,.037]);times=np.linspace(10,10.8,25)
        xyz,_=flight(p0,p1,.8,.5,times-10);pixels=project(xyz,self.K,self.R,self.C)
        track=[{'t':float(t),'ball':{'pixel':p.tolist()}} for t,p in zip(times,pixels)]
        events=[{'t':10.,'kind':'contact','height_prior_m':.65},{'t':10.8,'kind':'bounce','height_prior_m':.037}]
        calibration={k:v.tolist() for k,v in [('K',self.K),('R',self.R),('C',self.C)]}
        segments=fit_flights(track,events,calibration)
        self.assertEqual(len(segments),1)
        np.testing.assert_allclose(segments[0]['p0'],p0,atol=.02)
        np.testing.assert_allclose(segments[0]['p1'],p1,atol=.02)
        self.assertLess(segments[0]['reprojection_rmse_px'],.1)

    def test_no_extrapolation_or_long_gap_fill(self):
        segments=[{'start':1,'end':2,'p0':[1,10,.5],'p1':[4,1,.037],'drag_per_second':.5,
                   'support_times':[1,1.05,1.9,2],'reprojection_rmse_px':2}]
        self.assertIsNone(sample_segment(segments,.99))
        self.assertIsNone(sample_segment(segments,2.01))
        self.assertIsNone(sample_segment(segments,1.5))
        self.assertIsNone(sample_segment(segments,1.1)) # unsupported long gap, even near one edge
        self.assertIsNotNone(sample_segment(segments,1))

if __name__=='__main__':unittest.main()
