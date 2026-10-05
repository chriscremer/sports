import unittest
import numpy as np
from geometry import calibrate,ground,lift_pose

class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.corners=[[474,210],[818,210],[1090,536],[206,536]]
        self.H,self.K,self.R,self.C,_=calibrate(1280,720,self.corners)
    def test_court_corners_map_to_metres(self):
        expected=[[0,0],[6.096,0],[6.096,13.4112],[0,13.4112]]
        for pixel,world in zip(self.corners,expected):np.testing.assert_allclose(ground(self.H,pixel),world,atol=1e-5)
    def test_upright_height_and_visible_pose_reprojection(self):
        foot=ground(self.H,[560,245])
        joints=[[560,135,.99],[560,245,.99],[0,0,.05]]
        points=lift_pose(self.K,self.R,self.C,foot,joints)
        self.assertGreater(self.C[2],0)
        self.assertGreater(points[0][2],1)
        self.assertLess(points[0][2],2.5)
        self.assertIsNone(points[2])
        for p,k in zip(points[:2],joints[:2]):
            cam=self.R@(np.array(p)-self.C);pixel=self.K@cam;pixel=pixel[:2]/pixel[2]
            np.testing.assert_allclose(pixel,k[:2],atol=.02)

if __name__=='__main__':unittest.main()
