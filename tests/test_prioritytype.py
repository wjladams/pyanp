from unittest import TestCase
from pyanp.prioritizer import PriorityType, priority_type_default
import numpy as np
import pandas as pd

class TestPriorityType(TestCase):
    def test_crud(self):
        p = priority_type_default()
        self.assertEqual(p, PriorityType.RAW)

    def test_lists(self):
        lvals = [1, 3, 2, 4]
        raw = PriorityType.RAW.apply(lvals)
        # Apply always returns a copy
        self.assertFalse(raw is lvals)
        # Apply always returns the same type
        self.assertTrue(isinstance(raw, type(lvals)))
        # Raw means do nothing
        np.testing.assert_array_equal(lvals, raw)

        normalize = PriorityType.NORMALIZE.apply(lvals)
        # Apply always returns a copy
        self.assertFalse(normalize is lvals)
        # Apply always returns the same type
        self.assertTrue(isinstance(normalize, type(lvals)))
        # Raw means do nothing
        np.testing.assert_array_equal(normalize, np.array(lvals)/10.0)

        ideal = PriorityType.IDEALIZE.apply(lvals)
        # Apply always returns a copy
        self.assertFalse(ideal is lvals)
        # Apply always returns thy same type
        self.assertTrue(isinstance(ideal, type(lvals)))
        # Raw means do nothing
        np.testing.assert_array_equal(ideal, np.array(lvals)/4.0)

    def test_nparray(self):
        lvals = np.array([1, 3, 2, 4])
        raw = PriorityType.RAW.apply(lvals)
        # Apply always returns a copy
        self.assertFalse(raw is lvals)
        # Apply always returns the same type
        self.assertTrue(isinstance(raw, type(lvals)))
        # Raw means do nothing
        np.testing.assert_array_equal(lvals, raw)

        normalize = PriorityType.NORMALIZE.apply(lvals)
        # Apply always returns a copy
        self.assertFalse(normalize is lvals)
        # Apply always returns the same type
        self.assertTrue(isinstance(normalize, type(lvals)))
        # Raw means do nothing
        np.testing.assert_array_equal(normalize, np.array(lvals)/10.0)

        ideal = PriorityType.IDEALIZE.apply(lvals)
        # Apply always returns a copy
        self.assertFalse(ideal is lvals)
        # Apply always returns thy same type
        self.assertTrue(isinstance(ideal, type(lvals)))
        # Raw means do nothing
        np.testing.assert_array_equal(ideal, np.array(lvals)/4.0)

    def test_series(self):
        lvals = pd.Series(data=[1, 3, 2, 4], index=['Bill', 'John', 'Dan', 'Keith'])
        raw = PriorityType.RAW.apply(lvals)
        # Apply always returns a copy
        self.assertFalse(raw is lvals)
        # Apply always returns the same type
        self.assertTrue(isinstance(raw, type(lvals)))
        # Raw means do nothing
        np.testing.assert_array_equal(lvals, raw)

        normalize = PriorityType.NORMALIZE.apply(lvals)
        # Apply always returns a copy
        self.assertFalse(normalize is lvals)
        # Apply always returns the same type
        self.assertTrue(isinstance(normalize, type(lvals)))
        # Raw means do nothing
        np.testing.assert_array_equal(normalize, np.array(lvals)/10.0)

        ideal = PriorityType.IDEALIZE.apply(lvals)
        # Apply always returns a copy
        self.assertFalse(ideal is lvals)
        # Apply always returns thy same type
        self.assertTrue(isinstance(ideal, type(lvals)))
        # Raw means do nothing
        np.testing.assert_array_equal(ideal, np.array(lvals)/4.0)

