from unittest import TestCase

from pyanp.rating import *

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
        ps = rating.priority()
        print(ps)