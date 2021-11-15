# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 18:58:55 2021

@author: rapha
"""

#XSokoban levels bank from :
#https://www.cs.cornell.edu/andru/xsokoban.html


#MicroSokoban levels bank from :
#http://www.abelmartin.com/rj/sokobanJS/Skinner/David%20W.%20Skinner%20-%20Sokoban_files/Microban.txt

#fonction to get the level in raw

import numpy as np

TYPE_LOOKUP = {
    'wall': 0,
    'empty space': 1,
    'box target': 2,
    'box on target': 3,
    'box not on target': 4,
    'player': 5,
}


def XSokoban_lvl_to_raw(num_lvl:int) -> np.array:
        
    TYPE_LOOKUP_XSOKOBAN = {
        'wall': '#',
        'empty space': ' ',
        'box target': '.',
        'box on target': '*',
        'box not on target': '$',
        'player': '@',
    }
    
    file1 = open('levels/XSokoban/screen.'+str(num_lvl), 'r')
    Lines = file1.readlines()
    height, width = len(Lines),max([len(Lines[k]) for k in range(len(Lines))])-1
    board = np.zeros((height, width))
    k=0
    for line in Lines:
        L=[]
        for elt in line:
            for key,val in TYPE_LOOKUP_XSOKOBAN.items():
                if elt == val:
                    L.append(TYPE_LOOKUP[key])
        board[k]=L+[TYPE_LOOKUP['wall'] for k in range(width-len(L))]
        k+=1
    return board

def MicroSokoban_lvl_to_raw(num_lvl:int) -> np.array:
        
    TYPE_LOOKUP_XSOKOBAN = {
        'wall': '#',
        'empty space': ' ',
        'box target': '.',
        'box on target': '*',
        'box not on target': '$',
        'player': '@',
    }
    
    file1 = open('levels/MicroSokoban/screen.'+str(num_lvl), 'r')
    Lines = file1.readlines()
    height, width = len(Lines),max([len(Lines[k]) for k in range(len(Lines))])-1
    board = np.zeros((height, width))
    k=0
    for line in Lines:
        L=[]
        for elt in line:
            for key,val in TYPE_LOOKUP_XSOKOBAN.items():
                if elt == val:
                    L.append(TYPE_LOOKUP[key])
        board[k]=L+[TYPE_LOOKUP['wall'] for k in range(width-len(L))]
        k+=1
    return board