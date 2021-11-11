# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 18:58:55 2021

@author: rapha
"""

import numpy as np
# Les niveaux suivants ont été rajouté à la main et peuvent donc avoir des erreurs :
# 15/16/29/48/49/50
# Ce sont tous les niveaux où certaines boxes sont déjà sur une position cible

file1 = open('Xsokoban.txt', 'r')
Lines = file1.readlines()

#Liste comportant tous les niveaux de Xsokoban dans l'ordre indiqué ici :
#http://sokobano.de/wiki/index.php?title=Solver_Statistics_-_XSokoban_-_Thinking_Rabbit_%26_Various_Authors
#L'élément d'indice 0 de Liste_niveaux_Xsokoban est alors le niveau 1 et ainsi de suite

#La liste contient des array de (11,20) et de (17,20) dans le même format que env.room_state
#Les 51 premiers niveaux sont en (11,20) et les autres en (17,20)


Liste_niveaux_Xsokoban=[] 
k=0
for ligne in Lines:
  if 'name:' in ligne:
    Indice = Lines.index(ligne)+1
    Array_niv=np.zeros((11,20))
    if len(Liste_niveaux_Xsokoban)<51:
        i=0
        for k in range(Indice,Indice+11):
          L=[]
          for elt in Lines[k]:
            if elt in [' ','|','-'] :
              L.append(0)
            if elt == '.':
              L.append(1)
            if elt == '@':
              L.append(5)
            if elt == '^':
              L.append(2)
            if elt == '0':
              L.append(4)
            if elt == '8':
              L.append(3)
          Array_niv[i]=L+[0 for k in range(20-len(L))]
          i+=1
        Liste_niveaux_Xsokoban.append(Array_niv)
    else:
        Array_niv=np.zeros((17,20))
        i=0
        for k in range(Indice,Indice+17):
          L=[]
          for elt in Lines[k]:
            if elt in [' ','|','-'] :
              L.append(0)
            if elt == '.':
              L.append(1)
            if elt == '@':
              L.append(5)
            if elt == '^':
              L.append(2)
            if elt == '0':
              L.append(4)
            if elt == '8':
              L.append(3)
          Array_niv[i]=L+[0 for k in range(20-len(L))]
          i+=1
        Liste_niveaux_Xsokoban.append(Array_niv)