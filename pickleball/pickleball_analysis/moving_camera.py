"""Register sideline camera frames to visually marked lines on the court floor."""
import cv2
import numpy as np

from geometry import LENGTH, WIDTH, calibrate


KITCHEN = [[0, 4.572], [WIDTH, 4.572], [WIDTH, LENGTH - 4.572], [0, LENGTH - 4.572]]


class MovingCourt:
    def __init__(self, capture, references, width, height):
        self.width = width
        self.height = height
        self.scale = .75
        self.detector = cv2.SIFT_create(nfeatures=850, contrastThreshold=.012)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        self.references = []
        for ref in references:
            capture.set(cv2.CAP_PROP_POS_MSEC, 1000 * ref['time'])
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"Could not read reference at {ref['time']}")
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            positions, descriptors = self.features(frame)
            anchors = np.asarray(ref['kitchen_corners_normalized'], dtype=np.float32) * [width, height]
            self.references.append((ref['time'], positions, descriptors, anchors.astype(np.float32)))

    def features(self, frame):
        small = cv2.resize(frame, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        # The floor is blue. Close holes around the lines and white court lettering.
        mask = cv2.inRange(hsv, (87, 65, 30), (137, 255, 255))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8))
        mask[:int(mask.shape[0] * .26)] = 0
        keys, descriptors = self.detector.detectAndCompute(gray, mask)
        return np.asarray([key.pt for key in keys], dtype=np.float32) / self.scale, descriptors

    def locate(self, frame, time):
        positions, descriptors = self.features(frame)
        if descriptors is None or len(descriptors) < 10:
            return None
        matches_by_reference = sorted(self.references, key=lambda ref: abs(ref[0] - time))
        for _, reference, reference_descriptors, anchors in matches_by_reference[:3]:
            if reference_descriptors is None:
                continue
            neighbors = self.matcher.knnMatch(reference_descriptors, descriptors, k=2)
            matches = [first for pair in neighbors if len(pair) == 2
                       for first, second in [pair] if first.distance < .70 * second.distance]
            if len(matches) < 12:
                continue
            source = np.array([reference[match.queryIdx] for match in matches])
            target = np.array([positions[match.trainIdx] for match in matches])
            transform, inliers = cv2.findHomography(source, target, cv2.RANSAC, 2.8)
            if transform is None or inliers.sum() < 12:
                continue
            supported = target[inliers.ravel() == 1]
            if cv2.contourArea(cv2.convexHull(supported)) < .012 * self.width * self.height:
                continue
            corners = cv2.perspectiveTransform(anchors[None], transform)[0]
            if not cv2.isContourConvex(corners) or cv2.contourArea(corners) < .025 * self.width * self.height:
                continue
            if (corners.min(axis=0) < [-.25 * self.width, .22 * self.height]).any():
                continue
            if (corners.max(axis=0) > [1.25 * self.width, 1.12 * self.height]).any():
                continue
            return calibrate(self.width, self.height, corners, KITCHEN)
        return None
