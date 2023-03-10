"""
pauli_string.py - Define PauliString and LinearCombinaisonPauliString

Copyright 2020-2021 Maxime Dion <maxime.dion@usherbrooke.ca>
This file has been modified by <Your,Name> during the
QSciTech-QuantumBC virtual workshop on gate-based quantum computing.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
from typing import Union
from numpy.typing import NDArray


class PauliString:

    def __init__(self, z_bits: NDArray[np.bool_], x_bits: NDArray[np.bool_]):
        """
        Describe a Pauli string as 2 arrays of booleans.
        The PauliString represents (-1j)**(z_bits*x_bits) Z**z_bits X**x_bits.

        Args:
            z_bits (np.ndarray<bool>): True where a Z Pauli is applied.
            x_bits (np.ndarray<bool>): True where a X Pauli is applied.

        Raises:
            ValueError: [description]
        """

        if len(z_bits) != len(x_bits):
            raise ValueError('z_bits and x_bits must have the same number of elements')
        self.z_bits = z_bits
        self.x_bits = x_bits

    def __str__(self) -> str:
        """
        String representation of the PauliString.

        Returns:
            str: String of I, Z, X and Y.
        """

        pauli_labels = 'IZXY'
        pauli_choices = (self.z_bits + 2*self.x_bits).astype(int)
        out = ''
        for i in reversed(pauli_choices):
            out += pauli_labels[i]
        return out

    def __len__(self) -> int:
        """
        Number of Pauli in the PauliString.
        Also the number of qubits.

        Returns:
            int: Length of the PauliString, also number of qubits.
        """

        return len(self.z_bits)

    def __mul__(self, other: Union['PauliString', complex, float, int]) -> Union[tuple['PauliString', complex],
                                                                                 'LinearCombinaisonPauliString']:
        """
        Allow the use of '*' with other PauliString or with a coef (numeric).
        NOTE: If this isn't working, restart the kernel.
        Args:
            other (PauliString): Will compute the product 
            or
            other (float): [description]

        Returns:
            PauliString, complex: When other is a PauliString
            or
            LinearCombinaisonPauliString : When other is numeric
        """
        if isinstance(other, PauliString):
            return self.mul_pauli_string(other)
        else:
            return self.mul_coef(other)

    def __rmul__(self, other: Union['PauliString', complex, float, int]) -> Union[tuple['PauliString', float],
                                                                                  'LinearCombinaisonPauliString']:
        """
        Same as __mul__. Allow the use of '*' with a preceding coef (numeric) Like in 0.5 * PauliString

        Args:
            other (PauliString): Will compute the product 
            or
            other (float): [description]

        Returns:
            PauliString, complex: When other is a PauliString
            or
            LinearCombinaisonPauliString : When other is numeric
        """

        return self.__mul__(other)

    @classmethod
    def from_zx_bits(cls, zx_bits: NDArray[np.bool_]) -> 'PauliString':
        """
        Construct a PauliString from a single array<bool> of len 2n.

        Args:
            zx_bits (np.array<bool>): An array of bools. First n bits specify the Zs. Second half specify the Xs.

        Returns:
            PauliString: The Pauli string specified by the 'zx_bits'.
        """

        temp=np.split(zx_bits,2)
        z_bits = temp[0]
        x_bits = temp[1]
        
        return cls(z_bits, x_bits)

    @classmethod
    def from_str(cls, pauli_str: str) -> 'PauliString':
        """
        Construct a PauliString from a str (as returned by __str__).

        Args:
            pauli_str (str): String of length n made of 'I', 'X', 'Y' and 'Z'.

        Returns:
            PauliString: The Pauli string specified by the 'pauli_str'.
        """
        len_str = len(pauli_str)
        z_bits = np.array([0] * len_str, dtype=bool)
        x_bits = np.array([0] * len_str, dtype=bool)
        
        ii=len_str-1
        for char in pauli_str:
            if char == 'I' or char == 'X':
                z_bits[ii] = 0
            else:
                z_bits[ii] = 1
            ii = ii-1

        ii=len_str-1
        for char in pauli_str:
            if char == 'I' or char == 'Z':
                x_bits[ii] = 0
            else:
                x_bits[ii] = 1
            ii = ii-1
        
        return cls(z_bits, x_bits)

    def to_zx_bits(self) -> NDArray[np.bool_]:
        """
        Return the zx_bits representation of the PauliString.
        Useful to compare PauliString together.

        Returns:
            np.array<bool>: zx_bits representation of the PauliString of length 2n
        """
        pauli_str = str(self)
        
        z_bits = np.array([0] * len(self), dtype=bool)
        x_bits = np.array([0] * len(self), dtype=bool)
        
        ii=len(self)-1
        for char in pauli_str:
            if char == 'I' or char == 'X':
                z_bits[ii] = 0
            else:
                z_bits[ii] = 1
            ii = ii-1

        ii=len(self)-1
        for char in pauli_str:
            if char == 'I' or char == 'Z':
                x_bits[ii] = 0
            else:
                x_bits[ii] = 1
            ii = ii-1
            
        zx_bits = np.concatenate((z_bits, x_bits), axis=None)
        return zx_bits

    def to_xz_bits(self) -> NDArray[np.bool_]:
        """
        Return the xz_bits representation of the PauliString.
        Useful to check commutativity.

        Returns:
            np.array<bool>: xz_bits representation of the PauliString of length 2n
        """
        pauli_str = str(self)
        
        z_bits = np.array([0] * len(self), dtype=bool)
        x_bits = np.array([0] * len(self), dtype=bool)
        
        ii=len(self)-1
        for char in pauli_str:
            if char == 'I' or char == 'X':
                z_bits[ii] = 0
            else:
                z_bits[ii] = 1
            ii = ii-1

        ii=len(self)-1
        for char in pauli_str:
            if char == 'I' or char == 'Z':
                x_bits[ii] = 0
            else:
                x_bits[ii] = 1
            ii = ii-1
            
        xz_bits = np.concatenate((x_bits, z_bits), axis=None)
        return xz_bits

    def mul_pauli_string(self, other: 'PauliString') -> 'PauliString':
        """
        Product with an 'other' PauliString.

        Args:
            other (PauliString): An other PauliString.

        Raises:
            ValueError: If the other PauliString is not of the same length.

        Returns:
            PauliString, complex: The resulting PauliString and the product phase.
        """
        
        if len(self) != len(other):
            raise ValueError('PauliString must be of the same length')
        
        pauli_string_1 = str(self)
        pauli_string_2 = str(other)
        zx_bits_A = PauliString.to_zx_bits(self)
        zx_bits_B = PauliString.to_zx_bits(other)
        
        temp_A=np.split(zx_bits_A,2)
        z_bits_A = temp_A[0]
        x_bits_A = temp_A[1]
        
        temp_B=np.split(zx_bits_B,2)
        z_bits_B = temp_B[0]
        x_bits_B = temp_B[1]
        
        new_z_bits = np.logical_xor(z_bits_A,z_bits_B)
        new_x_bits = np.logical_xor(x_bits_A,x_bits_B)
        
        w = 0
        for jj in enumerate(pauli_string_1):
            ii = jj[0]
            if pauli_string_1[ii] == 'Y':
                if pauli_string_2[ii] == 'X':
                    w=w+1
                if pauli_string_2[ii] == 'Z':
                    w=w-1
            if pauli_string_1[ii] == 'Z':
                if pauli_string_2[ii] == 'Y':
                    w=w+1
                if pauli_string_2[ii] == 'X':
                    w=w-1
            if pauli_string_1[ii] == 'X':
                if pauli_string_2[ii] == 'Z':
                    w=w+1
                if pauli_string_2[ii] == 'Y':
                    w=w-1
        phase = (-1j)**w
        
     
        return self.__class__(new_z_bits, new_x_bits), phase

    def mul_coef(self, coef: Union[int, float, complex]) -> 'LinearCombinaisonPauliString':
        """
        Build a LCPS from a PauliString (self) and a numeric (coef).

        Args:
            coef (int, float or complex): A numeric coefficient.

        Returns:
            LinearCombinaisonPauliString: A LCPS with only one PauliString and coef.
        """
        coefs = pauli_strings = None
        coefs = np.array([coef],dtype = complex) 
        pauli_strings = np.array([self],dtype = PauliString)

        return LinearCombinaisonPauliString(coefs, pauli_strings)

    def ids(self) -> NDArray[np.bool_]:
        """
        Position of Identity in the PauliString.

        Returns:
            np.array<bool>: True where both z_bits and x_bits are False.
        """

        ids = np.array([0] * len(self), dtype=bool)
        
        ii=len(self)-1
        for char in str(self):
            if char == 'I':
                ids[ii] = 1
            ii=ii-1
        
        return ids

    def copy(self) -> 'PauliString':
        """
        Build a copy of the PauliString.

        Returns:
            PauliString: A copy.
        """

        return PauliString(self.z_bits.copy(), self.x_bits.copy())

    def to_matrix(self) -> NDArray[np.complex128]:
        """
        Build the matrix representation of the PauliString using the Kroenecker product.

        Returns:
            np.array<complex>: A 2**n side square matrix.
        """

        I_MAT = np.array([[1, 0], [0, 1]])
        X_MAT = np.array([[0, 1], [1, 0]])
        Y_MAT = np.array([[0, -1j], [1j, 0]])
        Z_MAT = np.array([[1, 0], [0, -1]])

        matrix = None
        matrix = np.ones((1,1),dtype = np.complex128)
        x = self.x_bits
        z = self.z_bits

        i = len(x)-1

        for ii in range(len(x)):
            if x[i] & z[i]:
                matrix = np.kron(matrix, Y_MAT)
            elif x[i] & ~z[i]:
                matrix = np.kron(matrix, X_MAT)
            elif ~x[i] & z[i]:
                matrix = np.kron(matrix, Z_MAT)
            elif ~x[i] & ~z[i]:
                matrix = np.kron(matrix, I_MAT)  
            i = i-1
        
        return matrix


class LinearCombinaisonPauliString:
    def __init__(self, coefs: NDArray[np.complex128], pauli_strings: NDArray[PauliString]):
        """
        Describes a Linear Combinaison of Pauli Strings.

        Args:
            coefs (np.array): Coefficients multiplying the respective PauliStrings.
            pauli_strings (np.array<PauliString>): PauliStrings.

        Raises:
            ValueError: If the number of coefs is different from the number of PauliStrings.
            ValueError: If all PauliStrings are not of the same length.
        """

        if len(coefs) != len(pauli_strings):
            raise ValueError('Must provide a equal number of coefs and PauliString')

        n_qubits = len(pauli_strings[0])
        for pauli in pauli_strings:
            if len(pauli) != n_qubits:
                raise ValueError('All PauliString must be of same length')

        self.n_terms = len(pauli_strings)
        self.n_qubits = len(pauli_strings[0])

        self.coefs = np.array(coefs, dtype=complex)
        self.pauli_strings = np.array(pauli_strings, dtype=PauliString)
        
    def __str__(self) -> str:
        """
        String representation of the LinearCombinaisonPauliString.

        Returns:
            str: Descriptive string.
        """

        out = f'{self.n_terms:d} pauli strings for {self.n_qubits:d} qubits (Real, Imaginary)'
        for coef, pauli in zip(self.coefs, self.pauli_strings):
            out += '\n' + f'{str(pauli)} ({np.real(coef):+.5f},{np.imag(coef):+.5f})'
        return out

    def __getitem__(self, key: Union[int, slice]) -> 'LinearCombinaisonPauliString':
        """
        Return a subset of the LinearCombinaisonPauliString array-like.

        Args:
            key (int or slice): Elements to be returned.

        Returns:
            LinearCombinaisonPauliString: LCPS with the element specified in key.
        """
        
        if isinstance(key, slice):
            new_coefs = np.array(self.coefs[key])
            new_pauli_strings = self.pauli_strings[key]
        else:
            if isinstance(key, int):
                key = [key]
            new_coefs = self.coefs[key]
            new_pauli_strings = self.pauli_strings[key]

        return self.__class__(new_coefs, new_pauli_strings)

    def __len__(self) -> int:
        """
        Number of PauliStrings in the LCPS.

        Returns:
            int: Number of PauliStrings/coefs.
        """

        return len(self.pauli_strings)

    def __add__(self, other: 'LinearCombinaisonPauliString') -> 'LinearCombinaisonPauliString':
        """
        Allow the use of + to add two LCPS together.

        Args:
            other (LinearCombinaisonPauliString): An other LCPS.

        Returns:
            LinearCombinaisonPauliString: New LCPS of len = len(self) + len(other).
        """

        return self.add_pauli_string_linear_combinaison(other)

    def __mul__(self, other: 'LinearCombinaisonPauliString') -> 'LinearCombinaisonPauliString':
        """
        Allow the use of * with other LCPS or numeric value(s)
        NOTE: If this isn't working, restart the kernel.
        
        Args:
            other (LinearCombinaisonPauliString): An other LCPS

        Returns:
            LinearCombinaisonPauliString: New LCPS of len = len(self) * len(other)
            or
            LinearCombinaisonPauliString: New LCPS of same length with modified coefs
        """

        if isinstance(other, LinearCombinaisonPauliString):
            return self.mul_linear_combinaison_pauli_string(other)
        else:
            return self.mul_coef(other)

    def __rmul__(self, other: 'LinearCombinaisonPauliString') -> 'LinearCombinaisonPauliString':
        """
        Same as __mul__.
        Allow the use of '*' with a preceding coef (numeric).
        Like in 0.5 * LCPS.

        Args:
            other (LinearCombinaisonPauliString): An other LCPS.

        Returns:
            LinearCombinaisonPauliString: New LCPS of len = len(self) * len(other)
            or
            LinearCombinaisonPauliString: New LCPS of same length with modified coefs
        """

        return self.__mul__(other)

    def add_pauli_string_linear_combinaison(self, other: 'LinearCombinaisonPauliString') -> 'LinearCombinaisonPauliString':
        """
        Adding with an other LCPS. Merging the coefs and PauliStrings arrays.

        Args:
            other (LinearCombinaisonPauliString): An other LCPS.

        Raises:
            ValueError: If other is not an LCPS.
            ValueError: If the other LCPS has not the same number of qubits.

        Returns:
            LinearCombinaisonPauliString: New LCPS of len = len(self) + len(other).
        """

        if not isinstance(other, LinearCombinaisonPauliString):
            raise ValueError('Can only add with an other LCPS')

        if self.n_qubits != other.n_qubits:
            raise ValueError('Can only add with LCPS of identical number of qubits')

        new_coefs = new_pauli_strings = None

        ################################################################################################################
        #new_coefs = np.concatenate(self.coefs, other.coefs)
        new_coefs = np.concatenate((self.coefs, other.coefs))
        new_pauli_strings = np.concatenate((self.pauli_strings, other.pauli_strings))
        ################################################################################################################

        #raise NotImplementedError()

        return self.__class__(new_coefs, new_pauli_strings)

    def mul_linear_combinaison_pauli_string(self, other: 'LinearCombinaisonPauliString') -> 'LinearCombinaisonPauliString':
        """
        Multiply with an other LCPS.

        Args:
            other (LinearCombinaisonPauliString): An other LCPS.

        Raises:
            ValueError: If other is not an LCPS.
            ValueError: If the other LCPS has not the same number of qubits.

        Returns:
            LinearCombinaisonPauliString: New LCPS of len = len(self) * len(other).
        """

        if not isinstance(other, LinearCombinaisonPauliString):
            raise ValueError()

        if self.n_qubits != other.n_qubits:
            raise ValueError('Can only add with LCPS of identical number of qubits')

        new_coefs = np.zeros((len(self)*len(other),), dtype=np.complex128)
        new_pauli_strings = np.zeros((len(self)*len(other),), dtype=PauliString)
        
        kk=0
        ii=0
        for char1 in self.coefs:
            jj=0
            for char2 in other.coefs:
                new_pauli_strings[kk],phase = PauliString.mul_pauli_string(self.pauli_strings[ii], other.pauli_strings[jj])
                new_coefs[kk] = self.coefs[ii] * other.coefs[jj] * phase
                kk=kk+1
                jj=jj+1
            ii=ii+1
            
        return self.__class__(new_coefs, new_pauli_strings)

    def mul_coef(self, other: 'LinearCombinaisonPauliString') -> 'LinearCombinaisonPauliString':
        """
        Multiply the LCPS by a coef (numeric) or an array of the same length.

        Args:
            other (float, complex or np.array): One numeric factor or one factor per PauliString.

        Raises:
            ValueError: If other is np.array should be of the same length as the LCPS.

        Returns:
            [type]: [description]
        """

        new_coefs = new_pauli_strings = None

        ################################################################################################################
        # YOUR CODE HERE
        # TO COMPLETE (after lecture on mapping)
        # new_coefs =
        # new_pauli_strings = 
        ################################################################################################################
        
        raise NotImplementedError()

        return self.__class__(new_coefs, new_pauli_strings)

    def to_zx_bits(self) -> NDArray[np.bool_]:
        """
        Build an array that contains all the zx_bits for each PauliString.

        Returns:
            np.array<bool>: A 2d array of booleans where each line is the zx_bits of a PauliString.
        """

        zx_bits = np.zeros((len(self), 2*self.n_qubits), dtype=np.bool_)
        
        kk=0
        for val in self.pauli_strings:
            
            pauli_str = str(self.pauli_strings[kk])

            z_bits = np.array([0] * len(self.pauli_strings[kk]), dtype=bool)
            x_bits = np.array([0] * len(self.pauli_strings[kk]), dtype=bool)

            ii=len(self.pauli_strings[kk])-1
            for char in pauli_str:
                if char == 'I' or char == 'X':
                    z_bits[ii] = 0
                else:
                    z_bits[ii] = 1
                ii = ii-1

            ii=len(self.pauli_strings[kk])-1
            for char in pauli_str:
                if char == 'I' or char == 'Z':
                    x_bits[ii] = 0
                else:
                    x_bits[ii] = 1
                ii = ii-1

            zx_bits[kk] = np.concatenate((z_bits, x_bits), axis=None) 
            kk=kk+1
        
        return zx_bits

    def to_xz_bits(self) -> NDArray[np.bool_]:
        """
        Build an array that contains all the xz_bits for each PauliString.

        Returns:
            np.array<bool>: A 2d array of booleans where each line is the xz_bits of a PauliString.
        """

        xz_bits = np.zeros((len(self), 2*self.n_qubits), dtype=np.bool_)

        kk=0
        for val in self.pauli_strings:
            
            pauli_str = str(self.pauli_strings[kk])

            z_bits = np.array([0] * len(self.pauli_strings[kk]), dtype=bool)
            x_bits = np.array([0] * len(self.pauli_strings[kk]), dtype=bool)

            ii=len(self.pauli_strings[kk])-1
            for char in pauli_str:
                if char == 'I' or char == 'X':
                    z_bits[ii] = 0
                else:
                    z_bits[ii] = 1
                ii = ii-1

            ii=len(self.pauli_strings[kk])-1
            for char in pauli_str:
                if char == 'I' or char == 'Z':
                    x_bits[ii] = 0
                else:
                    x_bits[ii] = 1
                ii = ii-1

            xz_bits[kk] = np.concatenate((x_bits, z_bits), axis=None) 
            kk=kk+1
        
        return xz_bits

    def ids(self) -> NDArray[np.bool_]:
        """
        Build an array that identifies the position of all the I for each PauliString.

        Returns:
            np.array<bool>: A 2d array of booleans where each line identifies the I on a PauliString.
        """

        ids = np.zeros((len(self), self.n_qubits), dtype=np.bool_)

        kk=0
        for val in self.pauli_strings:
            ii=len(self.pauli_strings[kk])-1
            for char in str(self.pauli_strings[kk]):
                if char == 'I':
                    ids[kk,ii] = 1
                ii=ii-1
            kk=kk+1
            
        return ids

    def combine(self) -> 'LinearCombinaisonPauliString':
        """
        Finds unique PauliStrings in the LCPS and combines the coefs of identical PauliStrings.
        Reduces the length of the LCPS.

        Returns:
            LinearCombinaisonPauliString: LCPS with combined coefficients.
        """

        

        ################################################################################################################
        # YOUR CODE HERE
        # TO COMPLETE (after lecture on mapping)
        # hint : make use to_zx_bits and np.unique
        LCPS_zx_bits = self.to_zx_bits()
        new_LCPS_zx_bits, counts = np.unique(LCPS_zx_bits, return_counts=True,axis=0)
        
        new_coefs = np.zeros(len(counts), dtype=np.complex128)
        new_pauli_strings = np.zeros(len(counts), dtype=PauliString)
        
        for step, val in enumerate(counts):
            for stepp, value in enumerate(self.coefs):
                if (new_LCPS_zx_bits[step,:]==LCPS_zx_bits[stepp,:]).all():
                    new_coefs[step] = new_coefs[step] + value
        
        for step, val in enumerate(new_LCPS_zx_bits):
            new_pauli_strings[step] = PauliString.from_zx_bits(val)
        ##########################
        return self.__class__(new_coefs, new_pauli_strings)

    def apply_threshold(self, threshold: float = 1e-6) -> 'LinearCombinaisonPauliString':
        """
        Remove PauliStrings with coefficients smaller than threshold.

        Args:
            threshold (float, optional): PauliStrings with coef smaller than 'threshold' will be removed. 
                                         Defaults to 1e-6.

        Returns:
            LinearCombinaisonPauliString: LCPS without coefficients smaller than threshold.
        """

        new_coefs = new_pauli_strings = None

        ################################################################################################################
        # YOUR CODE HERE
        # TO COMPLETE (after lecture on mapping)
        # Hint : create a np.array<bool> and use this array to get the subset of the lcps where this array is True
        checker = np.zeros(len(self.coefs),dtype=bool)
        kk=0
        for step,val in enumerate(self.coefs):
            if threshold < abs(val):
                checker[step] = True

        new_length = np.count_nonzero(checker == True)
        new_coefs = np.zeros(new_length, dtype=np.complex128)
        new_pauli_strings = np.zeros(new_length, dtype=PauliString)
        
        kk = 0
        for step,val in enumerate(checker):
            if val == True:
                new_coefs[kk] = self.coefs[step]
                new_pauli_strings[kk] = self.pauli_strings[step]
                kk=kk+1
        ################################################################################################################
        #raise NotImplementedError()

        return self.__class__(new_coefs,new_pauli_strings)

    def divide_in_bitwise_commuting_cliques(self) -> list['LinearCombinaisonPauliString']:
        """
        Find bitwise commuting cliques in the LCPS.

        Returns:
            list<LinearCombinaisonPauliString>: List of LCPS where all elements of one LCPS bitwise commute with each
                                                other.
        """

        cliques = list()

        ################################################################################################################
        # YOUR CODE HERE
        # TO COMPLETE (after activity 3.2)
        # This one can be hard to implement
        # Use to_zx_bits
        # Transform all I into Z and look for unique PauliStrings
        ################################################################################################################

        raise NotImplementedError()

        return cliques

    def sort(self) -> 'LinearCombinaisonPauliString':
        """
        Sort the PauliStrings by order of the zx_bits.

        Returns:
            LinearCombinaisonPauliString: Sorted.
        """

        order = None
        
        zx = self.to_zx_bits()
        val = []

        for item1 in zx:
            n = 0
            valu = 0
            for item2 in item1:
                if item2 == False:
                    valu = valu + 2**n
                n = n+1
            val.append(valu)
        
        order = list(np.argsort(val))
        
        new_coefs = self.coefs[order]
        new_pauli_strings = self.pauli_strings[order]

        return self.__class__(new_coefs, new_pauli_strings)
    
    def to_matrix(self) -> NDArray[np.complex128]:
        """
        Build the total matrix representation of the LCPS.

        Returns:
            np.array<complex>: A 2**n side square matrix.
        """

        size = 2**self.n_qubits
        matrix = np.zeros((size, size), dtype=np.complex128)

        ################################################################################################################
        # YOUR CODE HERE (OPTIONAL)
        # TO COMPLETE (after lecture on mapping)
        # Hints : sum all the matrices of all PauliStrings weighted by their coef
        ################################################################################################################

        raise NotImplementedError()

        return matrix