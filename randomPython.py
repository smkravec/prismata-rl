import random
import prismataengine as p

class PythonRandomPlayer(p.PrismataPlayerPython):
    def __init__(self):
        super().__init__(p.Players.One)
    
    def getAction(self, gamestate):
        actions = gamestate.getAbstractActions()
        action = random.choice(actions)
        return action
    
    def getMove(self, prismata_gamestate, move):
        gamestate=p.GameState(prismata_gamestate)
        saveActivePlayer = gamestate.activePlayer
        while gamestate.activePlayer == saveActivePlayer:
            abstractAction = self.getAction(gamestate)
            #print(type(abstractAction), abstractAction)
            actionPointer = gamestate.coerceAction(abstractAction)
            #print(type(actionPointer))
            move.append(p.unsafeIntToAction(actionPointer))
            gamestate.doAction(abstractAction)