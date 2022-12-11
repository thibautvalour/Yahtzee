import numpy as np


class Hash():
    def __init__(self, dice_max_value):
        self.dice_max_value = dice_max_value

    def initialize(self):

        # The value i is 1 if the combination created by i is deleteable
        self.is_combinaison_removable = np.zeros(
            self.dice_max_value**self.dice_max_value, dtype=np.uint8)
        for i1 in range(self.dice_max_value):
            for i2 in range(self.dice_max_value-i1):
                for i3 in range(self.dice_max_value-i1-i2):
                    for i4 in range(self.dice_max_value-i1-i2-i3):
                        for i5 in range(self.dice_max_value-i1-i2-i3-i4):
                            for i6 in range(self.dice_max_value-i1-i2-i3-i4-i5):
                                self.is_combinaison_removable[i1*self.dice_max_value**0+i2*self.dice_max_value**1+i3*self.dice_max_value **
                                                              2+i4*self.dice_max_value**3+i5*self.dice_max_value**4+i6*self.dice_max_value**5] = 1

        # The value in i is the rank of the i-th deletable combination
        self.index_available_combinaison = np.zeros(
            self.is_combinaison_removable.sum(), dtype=np.int32)
        j = 0
        for i in range(self.dice_max_value**self.dice_max_value):
            if self.is_combinaison_removable[i] == 1:
                self.index_available_combinaison[j] = i
                j += 1

        # To the value associates 0 if the combination formed by i is not deletable,
        # otherwise associates the non-zero rank
        self.combinaison_remove = np.zeros(
            self.dice_max_value**self.dice_max_value, dtype=np.int32)
        j = 0
        for i in range(self.dice_max_value**self.dice_max_value):
            if self.index_available_combinaison[j] == i:
                self.combinaison_remove[i] = j
                j += 1
                if i >= self.index_available_combinaison.max():
                    break

    def determine_all_index_available_moves(self, l):
        """
        l is an array of size 6, is the sum of dice
        """
        available_moves = []
        for i1 in range(l[0]+1):
            for i2 in range(l[1]+1):
                for i3 in range(l[2]+1):
                    for i4 in range(l[3]+1):
                        for i5 in range(l[4]+1):
                            for i6 in range(l[5]+1):
                                available_moves.append(i1*self.dice_max_value**0+i2*self.dice_max_value**1+i3*self.dice_max_value **
                                                       2+i4*self.dice_max_value**3+i5*self.dice_max_value**4+i6*self.dice_max_value**5)
        available_moves = self.combinaison_remove[np.array(available_moves)]
        available_moves_one_hot = np.zeros(
            self.index_available_combinaison.size, dtype=np.float32)
        available_moves_one_hot[available_moves] = 1
        return available_moves_one_hot

    def hash_function(self, l):
        """
        l is an array of size (n,6)
        """
        return np.array([self.determine_all_index_available_moves(l[i]) for i in range(l.shape[0])])

    def representation(self, n):
        if n < self.dice_max_value:
            return [n]
        return [n % self.dice_max_value]+self.representation(n//self.dice_max_value)

    def reverse_hash_function(self, n0):
        """
        n0 is the index of the move played
        """
        value = np.array(self.representation(
            self.index_available_combinaison[n0]))
        deleted_dice = np.zeros(self.dice_max_value)
        deleted_dice[:value.size] = value
        return deleted_dice.astype(np.int32)
