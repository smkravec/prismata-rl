from os import environ
# Only works with POSIX-style paths
#environ["PRISMATA_INIT_AI_JSON_PATH"] = f"{'/'.join(__file__.split('/')[:-1])}/AI_config.txt"
from sys import exc_info
import traceback
import prismataengine
import randomPython
import NN_opponent
import json
#try:
#   prp = randomPython.PythonRandomPlayer()
#    prismataengine.addPlayer('00PythonRandomPlayer', prp)
#    prismataengine.PrismataGUIEngine().run()
#except Exception as e:
#    exc_type, exc_obj, exc_tb = exc_info()
#    print(traceback.format_exc())
#    print(f"guitest[{exc_tb.tb_lineno}]: {type(e).__name__} {e}")

with open('./NN_opponent.json') as f:
    nn_opponent_params = json.load(f)

try:
    opponent = NN_opponent.NN_opponent(**nn_opponent_params)
    prismataengine.addPlayer('00NNOpponent', opponent)
    prismataengine.PrismataGUIEngine().run()
except Exception as e:
    exc_type, exc_obj, exc_tb = exc_info()
    print(traceback.format_exc())
    print(f"guitest[{exc_tb.tb_lineno}]: {type(e).__name__} {e}")