#! /usr/bin/env python

from lindenmayer import LSystem, Rule, StochasticRule
from draw import TurtleDrawer

import math

def main():
    # UNCOMMENT THIS TO PRODUCE THE 1ST L-SYSTEM
    # Pampas Grass
    drawer = TurtleDrawer(7, 1, 15)

    lsystem = LSystem('[X+X]',
                      {'X': 'F[BXA]F[BXA]',
                       'F': 'FA',
                       'A': StochasticRule(((0.6, 'F[+FX]'), (0.4, 'F[FX]'))),
                       'B': StochasticRule(((0.8, '+'),  (0.2, 'F')))
                       })

    description = lsystem.evaluate(6)

    # UNCOMMENT THIS TO PRODUCE THE 2ND L-SYSTEM
    # Wild hay
    # drawer = TurtleDrawer(5, 1, 15)
    #
    # lsystem = LSystem('X',
    #                   {
    #                       'X': 'FYYF[+FFXX][-FFXX]',
    #                       'Y': StochasticRule(((0.5, 'F'), (0.5, 'X')))
    #                   })
    #
    # description = lsystem.evaluate(6)

    # # UNCOMMENT THIS TO PRODUCE THE 3RD L-SYSTEM
    # Olive branch
    # drawer = TurtleDrawer(4, 1, 20)
    #
    # lsystem = LSystem('X',
    #                   {'X': 'F[YX][YX][XYX]FX',
    #                    'F': 'FF',
    #                    'Y': StochasticRule(((0.05, '+++'), (0.45, '+'), (0.45, '-'), (0.05, '---')))
    #                    })
    #
    # description = lsystem.evaluate(6)

    # uncomment if you want to see the string you're drawing
    # print(description)

    drawer.draw(description, offset=(0, -400))
    drawer.done()

if __name__ == '__main__':
    main()
