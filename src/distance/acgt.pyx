cdef class ACGTContent(object):
  """
    Distance simply based on the amount of a, c, g, t content in the strings.
    (Should be stored as transitions from the root node).
  """

  cdef list characters

  def __init__(self, characters=['A', 'C', 'G', 'T']):
    self.characters = characters


  cpdef double distance(self, left_vlmc, right_vlmc):
    # Assume this is the alphabet, only relevant case for us.
    cdef dict left_tree = left_vlmc.tree, right_tree = right_vlmc.tree

    cdef double distance = sum([abs(left_tree[""][char_] - right_tree[""][char_]) for char_ in self.characters])
    return distance
