import unittest
import numpy as np
from geometry import calibrate
from tracking import bridge_short_gaps

class ContinuityTests(unittest.TestCase):
    def setUp(self):
        _,self.K,self.R,self.C,_=calibrate(1280,720,[[474,210],[818,210],[1090,536],[206,536]])
    def player(self,x=2,pid=1,team='far'):
        return {'id':pid,'team':team,'xy':[x,3],'confidence':.8,'keypoints':[[560,135,.9]]*17}
    def bridge(self,frames):return bridge_short_gaps(frames,self.K,self.R,self.C)
    def test_short_gap_is_bracketed_and_labelled(self):
        frames=[{'t':0,'players':[self.player()]},{'t':.1,'players':[]},{'t':.2,'players':[self.player(2.2)]}]
        self.assertEqual(self.bridge(frames),1)
        p=frames[1]['players'][0];self.assertEqual(p['tracking_status'],'interpolated');self.assertIsNone(p['confidence']);self.assertEqual(p['support_times'],[0,.2]);np.testing.assert_allclose(p['xy'],[2.1,3])
    def test_long_gaps_and_clip_endpoints_stay_empty(self):
        frames=[{'t':0,'players':[]},{'t':.1,'players':[self.player()]},{'t':.4,'players':[]},{'t':.7,'players':[self.player()]},{'t':.8,'players':[]}]
        self.assertEqual(self.bridge(frames),0)
        for i in [0,2,4]:self.assertEqual(frames[i]['players'],[])
    def test_identity_and_team_changes_are_not_bridged(self):
        for p in [self.player(pid=2),self.player(team='near')]:
            frames=[{'t':0,'players':[self.player()]},{'t':.1,'players':[]},{'t':.2,'players':[p]}]
            self.assertEqual(self.bridge(frames),0)
    def test_implausible_motion_is_not_bridged(self):
        frames=[{'t':0,'players':[self.player()]},{'t':.1,'players':[]},{'t':.2,'players':[self.player(6)]}]
        self.assertEqual(self.bridge(frames),0)
    def test_fill_never_adds_a_third_teammate(self):
        frames=[{'t':0,'players':[self.player()]},{'t':.1,'players':[self.player(pid=2),self.player(pid=3)]},{'t':.2,'players':[self.player()]}]
        self.assertEqual(self.bridge(frames),0)

if __name__=='__main__':unittest.main()
