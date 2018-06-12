from unittest import TestCase

from pyanp.rating import *
from numpy.testing import assert_array_equal, assert_allclose
class TestRating(TestCase):

    def test_crud(self):
        rating = Rating()
        alts = ["a1", "a2", "a3"]
        rating.add_alt(alts)
        self.assertEqual(3, rating.nalts())
        unames = ("Bill", "Lee", "Paul", "Frank")
        users=rating.add_user(unames)
        self.assertEqual(4, rating.nusers())
        rating.vote_column("a1", ["H", "M", "L", "M"])
        rating.vote_column("a2", ["good", "good", "bad", "good"])
        ps = rating.priority()
        assert_allclose([0.5, 0.625, 0.0], ps)

    def test_manual_setting(self):
        rating = Rating()
        alts = ["a1", "a2", "a3"]
        rating.add_alt(alts)
        self.assertEqual(3, rating.nalts())
        unames = ("Bill", "Lee", "Paul", "Frank")
        users=rating.add_user(unames)
        rating.set_word_eval({
            "hi":(1, "high", "h"),
            "med":(0.3, "medium", "m", "md"),
            "low":(0.1, "lo", "l", "bad"),
            "stupid":(0.005, "stupd", "stu", "st", "s")
        })
        rating.vote_column("a1", ["H", "M", "L", "M"])
        rating.vote_column("a2", ["good", "good", "bad", "good"])
        rating.vote_column("a3", ["stupid", "st", "bad", "good"])
        ps = rating.priority()
        print(ps)
        print(rating.vote_values(alt_name="a1"))
