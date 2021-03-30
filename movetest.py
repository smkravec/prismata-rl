from os import environ
# Only works with POSIX-style paths
environ["PRISMATA_INIT_AI_JSON_PATH"] = f"{'/'.join(__file__.split('/')[:-1])}/AI_config.txt"
from sys import exc_info
import traceback
import prismataengine as p
import randomPython
gamestate = p.GameState('''{
                     "whiteMana":"0HH",
                 "blackMana":"0HH",
                 "phase":"action",
                 "table":
                 [
                     {"cardName":"Drone", "color":0, "amount":6},
                     {"cardName":"Engineer", "color":0, "amount":2},
                     {"cardName":"Drone", "color":1, "amount":7},
                     {"cardName":"Engineer", "color":1, "amount":2}
                 ],
                 "cards":["Drone","Engineer","Blastforge","Steelsplitter"]
         }''', cards=4)


prp = randomPython.PythonRandomPlayer()
move = p.Move()
p.getMove(prp.clone(), gamestate._state, move)
print("SIDE EFFECT TIME")
print(move)
