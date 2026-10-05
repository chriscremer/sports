import copy
import unittest

import numpy as np

from ball_geometry import flight, project, sample_segment
from geometry import calibrate
from reconstruct_comparison import reconstruct_model, supported_event_groups, detection_display_rows


class ComparisonReconstructionTests(unittest.TestCase):
    def calibration(self):
        H,K,R,C,_=calibrate(1280,720,[[474,210],[818,210],[1090,536],[206,536]])
        return dict(clip=dict(width=1280,height=720),calibration={key:value.tolist() for key,value in [('H',H),('K',K),('R',R),('C',C)]})

    def test_ground_projection_keeps_height_unknown_and_respects_source_scale(self):
        analysis=self.calibration()
        H=np.asarray(analysis['calibration']['H'])
        pixel=np.linalg.inv(H)@np.array([2.,7.,1.]);pixel=pixel[:2]/pixel[2]*1.5
        model=dict(frames=[dict(t=0,source_t=55,candidates=[dict(pixel=pixel.tolist(),score=.8)]),dict(t=1/30,source_t=55+1/30,candidates=[])])
        rows=detection_display_rows(model,dict(width=1920,height=1080),analysis,[])
        np.testing.assert_allclose(rows[0]['projection']['ground_xy'],[2,7],atol=1e-5)
        self.assertIsNone(rows[0]['projection']['height'])
        self.assertIsNone(rows[0]['projection']['fit_residual_1080p_px'])
        self.assertIsNone(rows[1]['projection'])

    def test_unmatched_candidate_remains_visible_beside_a_supported_flight(self):
        analysis=self.calibration();calibration=analysis['calibration']
        segments=[dict(start=10,end=10.8,p0=[2,13,.65],p1=[5,1,.037],drag_per_second=.5,support_times=[10+i/30 for i in range(25)],reprojection_rmse_px=0)]
        xyz=sample_segment(segments,10.4)['xyz']
        pixel=project(xyz,*[np.asarray(calibration[key]) for key in ['K','R','C']])*1.5
        def display(p):
            return detection_display_rows(dict(frames=[dict(t=.4,source_t=10.4,candidates=[dict(pixel=p.tolist(),score=.8)])]),dict(width=1920,height=1080),analysis,segments)[0]['projection']
        matched=display(pixel);unmatched=display(pixel+np.array([100.,0]))
        self.assertFalse(matched['show_with_fit'])
        self.assertTrue(unmatched['show_with_fit'])
        self.assertGreater(unmatched['fit_residual_1080p_px'],99)
        self.assertIsNone(unmatched['height'])

    def test_unsupported_flight_does_not_remove_supported_neighbors(self):
        events=[dict(t=t,kind='contact',height_prior_m=.5) for t in [0,1,2,3]]
        track=[dict(t=float(t),ball=dict(pixel=[100,100])) for t in list(np.linspace(.1,.9,6))+[1.5]+list(np.linspace(2.1,2.9,6))]
        groups, skipped=supported_event_groups(track,events)
        self.assertEqual([[e['t'] for e in group] for group in groups],[[0,1],[2,3]])
        self.assertEqual([(r['start'],r['end'],r['observations']) for r in skipped],[(1,2,1)])

    def test_1080p_fit_maps_to_calibration_and_preserves_inputs(self):
        _,K,R,C,_=calibrate(1280,720,[[474,210],[818,210],[1090,536],[206,536]])
        p0=np.array([2,13,.65]);p1=np.array([5,1,.037]);times=np.linspace(10,10.8,25)
        xyz,_=flight(p0,p1,.8,.5,times-10)
        pixels=project(xyz,K,R,C)*1.5
        model=dict(frames=[dict(source_t=float(t),candidates=[dict(pixel=p.tolist())]) for t,p in zip(times,pixels)])
        clip=dict(width=1920,height=1080)
        original=dict(clip=dict(width=1280,height=720),calibration={k:v.tolist() for k,v in [('K',K),('R',R),('C',C)]},ball_segments=[dict(p0=[99,99,99])])
        review=dict(events=[dict(t=10.,kind='contact',height_prior_m=.65),dict(t=10.8,kind='bounce',height_prior_m=.037)])
        untouched=copy.deepcopy((model,original,review))
        result=reconstruct_model(model,clip,original,review)
        self.assertEqual(len(result['segments']),1)
        np.testing.assert_allclose(result['segments'][0]['p0'],p0,atol=.02)
        np.testing.assert_allclose(result['segments'][0]['p1'],p1,atol=.02)
        self.assertEqual((model,original,review),untouched)
        self.assertIsNone(sample_segment(result['segments'],9.9))
        self.assertIsNone(sample_segment(result['segments'],10.9))

    def test_missing_review_or_detections_does_not_invent_height(self):
        model=dict(frames=[dict(source_t=t,candidates=[]) for t in np.linspace(10,11,10)])
        original=dict(clip=dict(width=1280,height=720),calibration={})
        clip=dict(width=1920,height=1080)
        review=dict(events=[dict(t=10,kind='contact',height_prior_m=.5),dict(t=11,kind='bounce',height_prior_m=.037)])
        for events in [None,review]:
            result=reconstruct_model(model,clip,original,events)
            self.assertEqual(result['segments'],[])
            self.assertEqual(result['reconstructed_frames'],0)
            self.assertIsNone(result['fit_rmse_1080p_px'])


if __name__=='__main__': unittest.main()
