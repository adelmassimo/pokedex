import csv
import cv2
import numpy as np
from matplotlib import pyplot as plt

TYPES = ['Grass', 'Poison', 'Fire', 'Flying', 'Dragon', 'Water', 'Bug', 'Normal', 'Electric', 'Ground', 'Fairy', 'Fighting', 'Psychic', 'Rock', 'Steel', 'Ice', 'Ghost', 'Dark']
NUM_TYPES = len(TYPES)
IMG_SHAPE = (96, 96, 3)

class Pokemon(object):
    """docstring for pokemon"""
    types = np.zeros( len(TYPES) )
    img = np.zeros( IMG_SHAPE )
    name = ''
    def __init__(self, name, img, type1, type2):
        self.types = np.zeros( len(TYPES) )
        self.name = name
        self.img = img
        if type1 != '': 
            self.types[TYPES.index(type1)] = 1
        if type2 != '':
            self.types[TYPES.index(type2)] = 1
        
class Pokedex(object):
    """docstring for Pokedex"""
    pokemons = []
    def load(self):
        with open('pokemon.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            first_row = True
            for row in csv_reader:
                if first_row:
                    print(f'Columns:{", ".join(row)}')
                    first_row = False
                else:
                    pkm = Pokemon(row[1], cv2.imread("sprites/pokemon/"+row[0]+".png"), row[2], row[3])
                    self.pokemons.append(pkm)

    def printPokemon(self, index):
        print( self.pokemons[index].name, self.pokemons[index].types)
        cv2.imshow('image', self.pokemons[index].img)
        cv2.waitKey(0)

    def trainSet(self, length):
        images = np.zeros( (length, 96, 96, 3) )
        labels = np.zeros( (length, NUM_TYPES) )
        for i in range(0, length):
            labels[i] = self.pokemons[0].types
            images[i] = self.pokemons[0].img
        return (labels, images)


    def testSet(self, length):
        images = np.zeros( (length, 96, 96, 3) )
        labels = np.zeros( (length, NUM_TYPES) )
        for i in range(0, length):
            labels[i] = self.pokemons[i].types
            images[i] = self.pokemons[500+i].img
        return (labels, images)

pokedex = Pokedex()
pokedex.load()

# pokedex.printPokemon(151)


        